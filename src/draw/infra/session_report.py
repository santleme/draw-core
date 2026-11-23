"""Relatorio de sessao: o resumo legivel de uma corrida do app.

A telemetria (telemetry.py) grava o retrato de 1 Hz num JSONL
rotativo: dado bruto, misturado entre sessoes, sem dizer QUAL
corrida era, com quais modelos, nem como a camera abriu. Quando
uma sessao ao vivo falha, reconstruir o que aconteceu dela exige
minerar esse arquivo — e a rotacao pode ter levado a parte que
importa. A sessao de 2026-11-02 a noite nem isso deixou: nenhum
artefato depois do fato dizia o que rodou.

Este modulo e o complemento: um JSON por corrida, em
<save_dir>/session-reports/, com cabecalho (modelos carregados
com sha, fonte, abertura da camera com o desfecho do supervisor,
porta da mao disponivel ou nao) e agregados da sessao inteira
(tempo ate a primeira folha confirmada, % do tempo com folha,
quedas, histogramas de estado e de modo do worker, medias de fps
e de custo por estagio). A escrita e atomica (tmp + replace) a
cada intervalo e no fechamento: mesmo se o app morrer no meio, o
relatorio parcial fica.

O custo sai do caminho critico como o da telemetria: update()
guarda o retrato mais recente sob cadeado; uma thread propria
amostra a 1 Hz e dobra os agregados.
"""

import itertools
import json
import os
import threading
import time
from pathlib import Path

_REPORT_SEQ = itertools.count()


class SessionReport:
    """Agregador de sessao: cabecalho fixo + agregados a 1 Hz."""

    def __init__(
        self,
        directory,
        header,
        sample_interval=1.0,
        write_interval=15.0,
        now=time.monotonic,
    ):
        """O cabecalho precisa de "inicio_epoch" (epoch do inicio).

        A validacao acontece na construcao: header sem a chave
        (KeyError) ou com valor nao numerico (TypeError/ValueError)
        estoura na cara do chamador, nao dentro da thread daemon onde
        morreria em silencio.
        """
        self._directory = Path(directory)
        self._header = dict(header)
        self._header["inicio_epoch"] = float(self._header["inicio_epoch"])
        self._sample_interval = float(sample_interval)
        self._write_interval = float(write_interval)
        self._now = now
        self._lock = threading.Lock()
        self._latest = {}
        self._stopped = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="draw-session-report", daemon=True
        )
        self._started_at = None
        self._samples = 0
        self._report_path = None
        self._write_lock = threading.Lock()
        self._write_seq = 0
        self._sheet_samples = 0
        self._first_sheet_s = None
        self._previous_sheet = False
        self._drops = 0
        self._states = {}
        self._worker_modes = {}
        self._hand_samples = 0
        self._sums = {}
        self._counts = {}
        self._maxes = {}
        self._error = None

    @property
    def path(self):
        # Nome calculado uma vez e congelado: a fracao de segundo e o
        # sufixo pid+contador impedem que duas sessoes no mesmo
        # segundo, no mesmo diretorio, se sobrescrevam. A escolha fica
        # sob cadeado: dois chamadores concorrentes nao podem consumir
        # o contador e publicar nomes diferentes.
        with self._lock:
            if self._report_path is None:
                epoch = self._header["inicio_epoch"]
                stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(epoch))
                frac = int((epoch % 1) * 1000)
                self._report_path = self._directory / (
                    f"session-{stamp}-{frac:03d}-{os.getpid()}-{next(_REPORT_SEQ)}.json"
                )
            return self._report_path

    def start(self):
        self._directory.mkdir(parents=True, exist_ok=True)
        self._started_at = self._now()
        self._thread.start()
        return self

    def update(self, **fields):
        """Guarda o retrato mais recente. Barato: um dict sob cadeado."""
        with self._lock:
            self._latest.update(fields)

    def _sample_once(self):
        # A dobra inteira fica sob o cadeado: os agregados e o retrato
        # andam juntos, e summary() le um conjunto consistente.
        with self._lock:
            record = dict(self._latest)
            elapsed = self._now() - self._started_at
            self._samples += 1
            present = bool(record.get("sheet_present"))
            if present:
                self._sheet_samples += 1
                if self._first_sheet_s is None:
                    self._first_sheet_s = round(elapsed, 1)
            elif self._previous_sheet:
                self._drops += 1
            self._previous_sheet = present
            state = record.get("state")
            if state is not None:
                self._states[state] = self._states.get(state, 0) + 1
            mode = record.get("worker_mode")
            if mode is not None:
                self._worker_modes[mode] = self._worker_modes.get(mode, 0) + 1
            if record.get("hand_present"):
                self._hand_samples += 1
            for key in ("loop_fps", "detect_ms", "pass_ms"):
                value = record.get(key)
                if isinstance(value, (int, float)):
                    self._sums[key] = self._sums.get(key, 0.0) + value
                    self._counts[key] = self._counts.get(key, 0) + 1
                    self._maxes[key] = max(self._maxes.get(key, value), value)

    def summary(self):
        def mean(key):
            count = self._counts.get(key, 0)
            return round(self._sums[key] / count, 2) if count else None

        # Leitura sob cadeado: sem ele, uma amostra em voo podia
        # mudar os dicts de agregados durante a montagem do resumo.
        with self._lock:
            total = self._samples
            return {
                **self._header,
                "duracao_s": (
                    round(self._now() - self._started_at, 1)
                    if self._started_at is not None
                    else None
                ),
                "amostras_s": total,
                "folha_presente_pct": (
                    round(100.0 * self._sheet_samples / total, 1) if total else None
                ),
                "primeira_folha_s": self._first_sheet_s,
                "quedas_de_folha": self._drops,
                "estados": dict(self._states),
                "modos_worker": dict(self._worker_modes),
                "mao_presente_pct": (
                    round(100.0 * self._hand_samples / total, 1) if total else None
                ),
                "loop_fps_medio": mean("loop_fps"),
                "detect_ms_medio": mean("detect_ms"),
                "detect_ms_max": self._maxes.get("detect_ms"),
                "pass_ms_medio": mean("pass_ms"),
                "pass_ms_max": self._maxes.get("pass_ms"),
            }

    def _write(self):
        try:
            self._directory.mkdir(parents=True, exist_ok=True)
            text = json.dumps(self.summary(), indent=1, ensure_ascii=False)
            with self._write_lock:
                # O escritor periodico e o close() podem se sobrepor:
                # cada escrita usa um temporario proprio e o lock
                # serializa — close() espera a escrita em voo em vez
                # de sobrepor o mesmo .tmp. latest.json tambem e
                # substituido por rename, nunca escrito no lugar.
                self._write_seq += 1
                tag = f"{os.getpid()}-{threading.get_ident()}-{self._write_seq}"
                tmp = self.path.with_name(f"{self.path.stem}.tmp-{tag}")
                latest_tmp = self._directory / f"latest.tmp-{tag}"
                try:
                    tmp.write_text(text, encoding="utf-8")
                    tmp.replace(self.path)
                    latest_tmp.write_text(text, encoding="utf-8")
                    latest_tmp.replace(self._directory / "latest.json")
                except OSError:
                    # Cada tentativa usa tag nova: sem a limpeza,
                    # falhas persistentes acumulam .tmp-* no dir.
                    tmp.unlink(missing_ok=True)
                    latest_tmp.unlink(missing_ok=True)
                    raise
        except Exception as error:  # noqa: BLE001 - vai para o laco
            with self._lock:
                self._error = error

    def _run(self):
        since_write = 0.0
        try:
            while not self._stopped.wait(self._sample_interval):
                self._sample_once()
                since_write += self._sample_interval
                if since_write >= self._write_interval:
                    since_write = 0.0
                    self._write()
        except Exception as error:  # noqa: BLE001 - vai para o laco
            # A thread daemon nao pode morrer em silencio: a excecao
            # que a parou fica visivel na propriedade error.
            with self._lock:
                self._error = error

    @property
    def error(self):
        """A excecao que parou o escritor, ou None."""
        with self._lock:
            return self._error

    def close(self):
        self._stopped.set()
        self._thread.join(timeout=2.0)
        self._write()
