"""CLI ``python -m lab.corpus`` para rotulagem, manifesto, replay e sweep."""

import argparse
import json
import sys
from pathlib import Path


def _parse_conds(items: list[str]) -> dict:
    out = {}
    for item in items:
        key, sep, val = item.partition("=")
        key, val = key.strip(), val.strip()
        if not sep or not key or not val:
            raise ValueError(f"--cond espera chave=valor: {item!r}")
        out[key] = val
    return out


def _parse_bundles(specs: list[str]) -> dict:
    bundles = {}
    for spec in specs:
        name, sep, raw = spec.partition(":")
        name = name.strip()
        if not sep or not name:
            raise ValueError(f"--bundle espera 'nome:flag1,flag2': {spec!r}")
        flags = [f.strip() for f in raw.split(",") if f.strip()]
        bundles[name] = flags
    return bundles


def _resolve_verify(mode: str):
    if mode == "none":
        return None
    if mode == "manifest":
        from draw.infra.manifest import verify_model

        return verify_model
    raise ValueError(f"--verify invalido: {mode!r}")


def _cmd_init(args) -> int:
    from lab.corpus.labels import init_clip

    conditions = _parse_conds(args.cond)
    provenance = {"source": args.source} if args.source else None
    out = init_clip(
        args.dir,
        fps=args.fps,
        orientation=args.orientation,
        desc=args.desc,
        conditions=conditions,
        provenance=provenance,
        issues=args.issue,
        overwrite=args.overwrite,
    )
    print(out)
    return 0


def _cmd_upgrade(args) -> int:
    from lab.corpus.labels import upgrade_clip

    out = upgrade_clip(args.dir, fps=args.fps)
    print(out)
    return 0


def _cmd_check(args) -> int:
    from lab.corpus.clip import validate_clip
    from lab.corpus.manifest import iter_clip_dirs

    target = Path(args.dir)
    dirs = [target] if (target / "clip.json").exists() else iter_clip_dirs(target)
    if not dirs:
        print(f"nenhum clipe encontrado em {target}", file=sys.stderr)
        return 2
    problems = []
    for clip_dir in dirs:
        for msg in validate_clip(clip_dir):
            problems.append(f"{clip_dir.name}: {msg}")
    for msg in problems:
        print(msg, file=sys.stderr)
    return 2 if problems else 0


def _cmd_manifest(args) -> int:
    from lab.corpus.manifest import build_manifest, write_manifest

    manifest = build_manifest(args.root)
    out = write_manifest(args.root, manifest)
    print(f"{len(manifest['clips'])} clipes registrados em {out}")
    return 0


def _cmd_verify(args) -> int:
    from lab.corpus.manifest import verify_manifest

    problems = verify_manifest(args.root)
    for msg in problems:
        print(msg, file=sys.stderr)
    return 2 if problems else 0


def _write_outputs(out_dir: Path, results, baseline: str, title: str, meta=None):
    from lab.corpus.metrics import RUN_META_NAME, write_results, write_run_meta
    from lab.corpus.report import render_markdown, render_summary

    out_dir.mkdir(parents=True, exist_ok=True)
    write_results(out_path := out_dir / "results.jsonl", results)
    write_run_meta(out_dir / RUN_META_NAME, {"baseline": baseline, **(meta or {})})
    md = render_markdown(results, baseline, title=title)
    (out_dir / "report.md").write_text(md, encoding="utf-8")
    summary = render_summary(results, baseline)
    print(json.dumps({"results": str(out_path), **summary}, indent=2))


def _cmd_run(args) -> int:
    from lab.corpus.manifest import verify_manifest
    from lab.corpus.run import run_corpus

    bundles = _parse_bundles(args.bundle)
    baseline = args.baseline or next(iter(bundles))
    verify = _resolve_verify(args.verify)
    results = run_corpus(
        args.root,
        bundles,
        verify_corpus=verify_manifest,
        only=args.only or None,
        verify=verify,
        progress=lambda clip, b, i, n: print(
            f"[{i}/{n}] {clip} :: {b}", file=sys.stderr
        ),
    )
    _write_outputs(
        Path(args.out),
        results,
        baseline,
        Path(args.root).name,
        meta={"bundles": list(bundles)},
    )
    return 0


def _cmd_sweep(args) -> int:
    from lab.corpus.manifest import verify_manifest
    from lab.corpus.sweep import grid_bundles, parse_grid, run_sweep

    base_flags = [f.strip() for f in args.base.split(",") if f.strip()]
    bundles = grid_bundles(parse_grid(args.grid), base_flags=base_flags)
    baseline = args.baseline or next(iter(bundles))
    verify = _resolve_verify(args.verify)
    results = run_sweep(
        args.root,
        args.grid,
        base_flags=base_flags,
        verify_corpus=verify_manifest,
        verify=verify,
        progress=lambda clip, b, i, n: print(
            f"[{i}/{n}] {clip} :: {b}", file=sys.stderr
        ),
    )
    _write_outputs(
        Path(args.out),
        results,
        baseline,
        f"sweep ({args.grid})",
        meta={"bundles": list(bundles), "grid": args.grid},
    )
    return 0


def _cmd_report(args) -> int:
    from lab.corpus.metrics import RUN_META_NAME, load_results, load_run_meta
    from lab.corpus.report import render_markdown

    results = load_results(args.results)
    meta = load_run_meta(Path(args.results).parent / RUN_META_NAME)
    baseline = (
        args.baseline
        or meta.get("baseline")
        or (results[0].bundle if results else "base")
    )
    print(render_markdown(results, baseline, title=Path(args.results).stem))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Monta o parser de subcomandos do corpus."""
    parser = argparse.ArgumentParser(
        prog="python -m lab.corpus",
        description="Corpus de clipes reais: schema, manifesto, replay e sweep.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="rotula um clipe novo (clip.json v2)")
    p_init.add_argument("dir")
    p_init.add_argument("--fps", type=float, required=True)
    p_init.add_argument("--orientation", default=None)
    p_init.add_argument("--desc", default="")
    p_init.add_argument("--cond", action="append", default=[])
    p_init.add_argument("--issue", action="append", type=int, default=[])
    p_init.add_argument("--source", default=None)
    p_init.add_argument("--overwrite", action="store_true")

    p_upg = sub.add_parser("upgrade", help="atualiza clip.json v1 -> v2")
    p_upg.add_argument("dir")
    p_upg.add_argument("--fps", type=float, default=None)

    p_chk = sub.add_parser("check", help="valida clipe(s) e contagem de frames")
    p_chk.add_argument("dir")

    p_man = sub.add_parser("manifest", help="(re)gera corpus.json com sha256")
    p_man.add_argument("root")

    p_ver = sub.add_parser("verify", help="confere corpus contra corpus.json")
    p_ver.add_argument("root")

    p_run = sub.add_parser("run", help="executa bundles sobre o corpus")
    p_run.add_argument("root")
    p_run.add_argument("--bundle", action="append", required=True)
    p_run.add_argument("--baseline", default=None)
    p_run.add_argument("--only", action="append", default=[])
    p_run.add_argument("--verify", choices=("manifest", "none"), default="manifest")
    p_run.add_argument("--out", default="corpus-out")

    p_swp = sub.add_parser("sweep", help="varre grade cartesiana de flags")
    p_swp.add_argument("root")
    p_swp.add_argument("--grid", required=True)
    p_swp.add_argument("--base", default="")
    p_swp.add_argument("--baseline", default=None)
    p_swp.add_argument("--verify", choices=("manifest", "none"), default="manifest")
    p_swp.add_argument("--out", default="corpus-sweep")

    p_rep = sub.add_parser("report", help="rerrenderiza Markdown de results.jsonl")
    p_rep.add_argument("results")
    p_rep.add_argument("--baseline", default=None)

    return parser


def main(argv=None) -> int:
    """Ponto de entrada de ``python -m lab.corpus``."""
    from lab.corpus.clip import ClipError
    from lab.corpus.labels import ClipInitError

    args = build_parser().parse_args(argv)
    handlers = {
        "init": _cmd_init,
        "upgrade": _cmd_upgrade,
        "check": _cmd_check,
        "manifest": _cmd_manifest,
        "verify": _cmd_verify,
        "run": _cmd_run,
        "sweep": _cmd_sweep,
        "report": _cmd_report,
    }
    try:
        return handlers[args.cmd](args)
    except (ClipInitError, ClipError, ValueError, OSError) as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
