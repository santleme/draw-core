import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(
    os.name != "posix" or not BASH, reason="requires a POSIX shell environment"
)

# Stub dos pythons de sistema (python3/python/py): a linha "-c" responde
# a sonda de versao do _find_python; "-3" cobre o `py -3` — o word-split
# de "py -3" no run.sh entrega o flag como $1.
_PY_STUB = (
    "#!/bin/sh\n"
    '[ "$1" = -3 ] && shift\n'
    '[ "$1" = -c ] && exit 0\n'
    'if [ "$3" != .venv ]; then\n'
    '    mkdir -p "$3/bin" && cp .venv/bin/python .venv/bin/draw "$3/bin/"\n'
    "fi\n"
)


@pytest.fixture
def launcher(tmp_path):
    shutil.copyfile(Path(__file__).resolve().parents[1] / "run.sh", tmp_path / "run.sh")
    tools = tmp_path / "tools"
    tools.mkdir()
    for name in ("dirname", "uname", "touch", "rm", "cp", "mkdir"):
        (tools / name).symlink_to(shutil.which(name))
    venv = tmp_path / ".venv"
    bin_dir = venv / "bin"
    bin_dir.mkdir(parents=True)
    python = bin_dir / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import sys, types\n"
        "from pathlib import Path\n"
        "prefix = Path(__file__).resolve().parents[1]\n"
        "if sys.argv[1] == '-c':\n"
        "    sys.prefix = str(prefix)\n"
        "    for name in ('cv2', 'onnxruntime', 'draw', 'draw.app'):\n"
        "        sys.modules[name] = types.ModuleType(name)\n"
        "    sys.modules['draw.app']._gui_backend_compiled = lambda: not (prefix / 'gui-broken').exists()\n"
        "    if not (prefix / 'ximgproc-missing').exists():\n"
        "        sys.modules['cv2'].ximgproc = object()\n"
        "    site = prefix / 'site'\n"
        "    def _dists():\n"
        "        out = []\n"
        "        for p in site.glob('*.dist-info'):\n"
        "            meta_file = p / 'METADATA'\n"
        "            name = p.name.rsplit('-', 1)[0]\n"
        "            if meta_file.exists():\n"
        "                name = next(\n"
        "                    line.split(':', 1)[1].strip()\n"
        "                    for line in meta_file.read_text().splitlines()\n"
        "                    if line.startswith('Name:')\n"
        "                )\n"
        "            record = p / 'RECORD'\n"
        "            files = (\n"
        "                [row.split(',', 1)[0] for row in record.read_text().splitlines()]\n"
        "                if record.exists() else []\n"
        "            )\n"
        "            out.append(types.SimpleNamespace(\n"
        "                metadata={'Name': name}, _path=p, files=files,\n"
        "                locate_file=lambda f, p=p: p.parent / f))\n"
        "        return out\n"
        "    meta = types.ModuleType('importlib.metadata')\n"
        "    meta.distributions = _dists\n"
        "    sys.modules['importlib.metadata'] = meta\n"
        "    sc = types.ModuleType('sysconfig')\n"
        "    sc.get_path = lambda key: str(site)\n"
        "    sys.modules['sysconfig'] = sc\n"
        "    if (prefix / 'hand-installed').exists():\n"
        "        sys.modules['mediapipe'] = types.ModuleType('mediapipe')\n"
        "    else:\n"
        "        sys.modules['mediapipe'] = None\n"
        "    exec(sys.argv[2])\n"
        "else:\n"
        "    site = prefix / 'site'\n"
        "    with open('commands', 'a') as log:\n"
        "        print(' '.join(sys.argv[1:]), file=log)\n"
        "    if sys.argv[-1] == '.[gui,hand]':\n"
        "        if not Path('hand-available').exists():\n"
        "            sys.exit(7)\n"
        "        (prefix / 'hand-installed').touch()\n"
        "    elif (\n"
        "        sys.argv[-1] != 'pip'\n"
        "        and sys.argv[-1] != './packages/draw-dral'\n"
        "        and not Path('gui-available').exists()\n"
        "    ):\n"
        "        sys.exit(7)\n"
        "    if '-e' in sys.argv:\n"
        "        for pkg in ('opencv-contrib-python-headless', 'opencv-contrib-python'):\n"
        "            dist = site / (pkg.replace('-', '_') + '-5.0.0.93.dist-info')\n"
        "            dist.mkdir(parents=True, exist_ok=True)\n"
        "            (dist / 'METADATA').write_text('Name: ' + pkg + '\\n')\n"
        "    if '-e' in sys.argv or sys.argv[-1].startswith('opencv-contrib-python'):\n"
        "        (prefix / 'gui-broken').unlink(missing_ok=True)\n"
        "    if '-e' in sys.argv and (prefix.parent / 'gui-stubborn').exists():\n"
        "        (prefix / 'gui-broken').touch()\n"
    )
    python.chmod(0o755)
    draw = bin_dir / "draw"
    draw.write_text('#!/bin/sh\nprintf "%s\\n" "$0" >> launches\n[ "$1" = --check ]\n')
    draw.chmod(0o755)
    python3 = tools / "python3"
    python3.write_text(_PY_STUB)
    python3.chmod(0o755)
    return tools


@pytest.mark.parametrize("use_uv", [False, True], ids=["pip", "uv"])
def test_failed_setup_never_uninstalls_opencv(tmp_path, launcher, use_uv):
    (tmp_path / ".venv" / "gui-broken").touch()
    if use_uv:
        uv = launcher / "uv"
        uv.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> commands\nexit 7\n')
        uv.chmod(0o755)
    result = subprocess.run(
        [BASH, "run.sh", "--check"],
        cwd=tmp_path,
        env={"PATH": str(launcher)},
        check=False,
        timeout=10,
    )
    commands = (tmp_path / "commands").read_text().splitlines()
    assert all("uninstall" not in command for command in commands)
    dral = "-m pip install -e ./packages/draw-dral"
    if use_uv:
        assert commands[-2:] == [_INSTALL_SYNC_HAND, _INSTALL_SYNC_GUI]
    else:
        # pip nao le tool.uv.sources: o membro entra antes de cada -e.
        assert commands[-4:] == [
            dral,
            "-m pip install -e .[gui,hand]",
            dral,
            "-m pip install -e .[gui]",
        ]
    assert result.returncode == 7
    assert not (tmp_path / ".venv" / ".no_hand").exists()


def test_gui_fallback_retries_hand_after_cooldown(tmp_path, launcher):
    (tmp_path / "gui-available").touch()
    (tmp_path / ".venv" / "gui-broken").touch()
    args = [BASH, "run.sh", "--check"]
    options = {
        "cwd": tmp_path,
        "env": {"PATH": str(launcher)},
        "check": True,
        "timeout": 10,
    }
    subprocess.run(args, **options)
    marker = tmp_path / ".venv" / ".no_hand"
    assert marker.is_file()
    commands = tmp_path / "commands"
    initial = commands.read_text()
    assert initial.splitlines()[-1] == "-m pip install -e .[gui]"

    subprocess.run(args, **options)
    assert commands.read_text() == initial

    os.utime(marker, (1, 1))
    (tmp_path / "hand-available").touch()
    subprocess.run(args, **options)
    assert (tmp_path / ".venv-hand" / "hand-installed").is_file()
    assert not (tmp_path / ".venv" / "hand-installed").exists()
    launches = (tmp_path / "launches").read_text().splitlines()
    assert [tmp_path / launch for launch in launches] == [
        tmp_path / ".venv/bin/draw",
        tmp_path / ".venv/bin/draw",
        tmp_path / ".venv-hand/bin/draw",
    ]


@pytest.mark.parametrize("use_uv", [False, True], ids=["pip", "uv"])
def test_expired_hand_fallback_still_launches_offline(tmp_path, launcher, use_uv):
    if use_uv:
        uv = launcher / "uv"
        uv.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> commands\nexit 7\n')
        uv.chmod(0o755)
    marker = tmp_path / ".venv" / ".no_hand"
    marker.touch()
    os.utime(marker, (1, 1))

    subprocess.run(
        [BASH, "run.sh", "--check"],
        cwd=tmp_path,
        env={"PATH": str(launcher)},
        check=True,
        timeout=10,
    )
    commands = (tmp_path / "commands").read_text()
    assert "uninstall" not in commands
    assert ".[gui]\n" not in commands
    assert (
        "sync --inexact --extra gui --no-install-package "
        "opencv-contrib-python-headless\n" not in commands
    )
    assert marker.stat().st_mtime > 1
    launched = (tmp_path / "launches").read_text().strip()
    assert tmp_path / launched == tmp_path / ".venv/bin/draw"


FAKE_UV = """#!/bin/sh
printf "%s\\n" "$*" >> commands
add_dist() {
    dir="$1/site/$(echo "$2" | tr - _)-5.0.0.93.dist-info"
    mkdir -p "$dir" && printf "Name: %s\\n" "$2" > "$dir/METADATA"
}
if [ "$1" = "sync" ]; then
    [ -n "$UV_PROJECT_ENVIRONMENT" ] || exit 1
    case " $* " in
        *" --locked "*) [ -f uv.lock ] || exit 1 ;;
    esac
    case " $* " in
        *" --offline "*) [ -f wheel-cached ] || exit 1 ;;
    esac
    prefix="$UV_PROJECT_ENVIRONMENT"
    if [ "$prefix" != .venv ]; then
        mkdir -p "$prefix/bin"
        cp .venv/bin/python .venv/bin/draw "$prefix/bin/"
    fi
    case " $* " in
        *" --extra hand "*) touch "$prefix/hand-installed" ;;
    esac
    case " $* " in
        *" --no-install-package opencv-contrib-python-headless "*) ;;
        *)
            if [ ! -d "$prefix/site/opencv_contrib_python_headless-5.0.0.93.dist-info" ]; then
                add_dist "$prefix" opencv-contrib-python-headless
                touch "$prefix/gui-broken"
            fi ;;
    esac
    case " $* " in
        *" --reinstall-package opencv-contrib-python "*)
            add_dist "$prefix" opencv-contrib-python
            rm -f "$prefix/gui-broken" "$prefix/ximgproc-missing" ;;
        *" --extra gui "*)
            if [ ! -d "$prefix/site/opencv_contrib_python-5.0.0.93.dist-info" ]; then
                add_dist "$prefix" opencv-contrib-python
                [ -e "$prefix/site/opencv_contrib_python_headless-5.0.0.93.dist-info" ] ||
                    rm -f "$prefix/gui-broken" "$prefix/ximgproc-missing"
            fi ;;
    esac
    exit 0
fi
exit 0
"""

MINI_PYPROJECT = """\
[project]
name = "opencv-provider-fixture"
version = "0.0.0"
requires-python = ">=3.11"
dependencies = ["opencv-contrib-python-headless>=4.8,<5.1"]

[project.optional-dependencies]
gui = ["opencv-contrib-python>=4.8,<5.1"]

[tool.uv]
package = false
"""

_CV2_GUI_CHECK = (
    "import re, sys, cv2\n"
    "m = re.search(r'^[ \\t]*GUI:[ \\t]*(\\S*)', cv2.getBuildInformation(), re.M)\n"
    "sys.exit(0 if m and m.group(1).upper() not in ('NONE','NO','OFF','FALSE','0') "
    "else 1)\n"
)

# Mesmo script do _drop_extra_cv2_providers no run.sh.
_CV2_DROP_EXTRAS = (
    "import shutil, sysconfig\n"
    "from pathlib import Path\n"
    "from importlib.metadata import distributions\n"
    "extra = {\n"
    "    'opencv-contrib-python-headless',\n"
    "    'opencv-python',\n"
    "    'opencv-python-headless',\n"
    "}\n"
    "roots = {Path(sysconfig.get_path(k)).resolve() for k in ('purelib', 'platlib')}\n"
    "dists = [\n"
    "    d for d in distributions()\n"
    "    if (p := getattr(d, '_path', None)) is not None\n"
    "    and Path(p).resolve().parent in roots\n"
    "]\n"
    "names = {d.metadata['Name'].lower() for d in dists}\n"
    "if 'opencv-contrib-python' in names:\n"
    "    keep = {\n"
    "        Path(d.locate_file(f)).resolve()\n"
    "        for d in dists\n"
    "        if d.metadata['Name'].lower() not in extra\n"
    "        for f in (d.files or ())\n"
    "    }\n"
    "    for d in dists:\n"
    "        if d.metadata['Name'].lower() not in extra:\n"
    "            continue\n"
    "        root = Path(d._path).resolve().parent\n"
    "        parents = set()\n"
    "        for f in d.files or ():\n"
    "            target = Path(d.locate_file(f)).resolve()\n"
    "            if target in keep or root not in target.parents:\n"
    "                continue\n"
    "            if target.is_file() or target.is_symlink():\n"
    "                target.unlink(missing_ok=True)\n"
    "                parents.update(target.parents)\n"
    "        for folder in sorted(\n"
    "            (p for p in parents if root in p.parents),\n"
    "            key=lambda p: -len(p.parts),\n"
    "        ):\n"
    "            try:\n"
    "                folder.rmdir()\n"
    "            except OSError:\n"
    "                pass\n"
    "        shutil.rmtree(d._path, ignore_errors=True)\n"
)


def _fake_uv(launcher):
    uv = launcher / "uv"
    uv.write_text(FAKE_UV)
    uv.chmod(0o755)


def _commands(tmp_path):
    return (tmp_path / "commands").read_text().splitlines()


_INSTALL_SYNC_GUI = (
    "sync --inexact --extra gui --no-install-package opencv-contrib-python-headless"
)

_INSTALL_SYNC_HAND = _INSTALL_SYNC_GUI + " --extra hand"

_REPAIR_SYNC = (
    "sync --locked --offline --inexact --extra gui "
    "--reinstall-package opencv-contrib-python "
    "--no-install-package opencv-contrib-python-headless"
)

_REPAIR_SYNC_FALLBACK = (
    "sync --inexact --extra gui --reinstall-package opencv-contrib-python "
    "--no-install-package opencv-contrib-python-headless"
)


def _add_provider(venv, name, files=()):
    norm = name.replace("-", "_")
    dist_info = venv / "site" / f"{norm}-5.0.0.93.dist-info"
    dist_info.mkdir(parents=True, exist_ok=True)
    (dist_info / "METADATA").write_text(f"Name: {name}\n")
    rows = [f"{dist_info.name}/METADATA,,"] + [f"{f},," for f in files]
    (dist_info / "RECORD").write_text("\n".join(rows) + "\n")
    return dist_info


def test_armed_env_collapses_providers_offline(tmp_path, launcher):
    """Par de providers registrado + GUI funcionando + uv quebrado:
    o launcher remove so o dist-info extra e lanca — sem sync, sem
    uninstall e sem wheel algum. Regressao do review: o repair antigo
    desinstalava os providers antes de saber se havia wheel."""
    contrib = _add_provider(tmp_path / ".venv", "opencv-contrib-python")
    headless = _add_provider(tmp_path / ".venv", "opencv-contrib-python-headless")
    # Dist-info fora do purelib do env ("herdado" de outra instalacao):
    # o collapse nunca pode tocar nele.
    foreign = _add_provider(tmp_path / "foreign-venv", "opencv-contrib-python-headless")
    (tmp_path / ".venv" / "hand-installed").touch()
    uv = launcher / "uv"
    uv.write_text("#!/bin/sh\nprintf '%s\\n' \"$*\" >> commands\nexit 7\n")
    uv.chmod(0o755)

    subprocess.run(
        [BASH, "run.sh", "--check"],
        cwd=tmp_path,
        env={"PATH": str(launcher)},
        check=True,
        timeout=10,
    )
    assert not headless.exists()
    assert contrib.exists()
    assert foreign.exists()
    assert not (tmp_path / "commands").exists()
    launched = (tmp_path / "launches").read_text().strip()
    assert tmp_path / launched == tmp_path / ".venv/bin/draw"


def test_collapse_removes_only_exclusive_files(tmp_path, launcher):
    """O dist-info extra sai junto com os arquivos que so ele possui
    (.libs privados); arquivos cv2 compartilhados ficam intactos."""
    site = tmp_path / ".venv" / "site"
    contrib = _add_provider(
        tmp_path / ".venv",
        "opencv-contrib-python",
        files=["cv2/shared.py", "cv2/gui.so"],
    )
    headless = _add_provider(
        tmp_path / ".venv",
        "opencv-contrib-python-headless",
        files=[
            "cv2/shared.py",
            "opencv_contrib_python_headless.libs/libfoo.so",
        ],
    )
    for rel in (
        "cv2/shared.py",
        "cv2/gui.so",
        "opencv_contrib_python_headless.libs/libfoo.so",
    ):
        target = site / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x")
    (tmp_path / ".venv" / "hand-installed").touch()
    uv = launcher / "uv"
    uv.write_text("#!/bin/sh\nexit 7\n")
    uv.chmod(0o755)

    subprocess.run(
        [BASH, "run.sh", "--check"],
        cwd=tmp_path,
        env={"PATH": str(launcher)},
        check=True,
        timeout=10,
    )
    assert not headless.exists()
    assert contrib.exists()
    assert not (site / "opencv_contrib_python_headless.libs").exists()
    assert (site / "cv2" / "shared.py").is_file()
    assert (site / "cv2" / "gui.so").is_file()
    launched = (tmp_path / "launches").read_text().strip()
    assert tmp_path / launched == tmp_path / ".venv/bin/draw"


def test_ximgproc_missing_forces_reinstall(tmp_path, launcher):
    """opencv-python por cima do contrib: a janela abriria, mas sem
    cv2.ximgproc a sonda falha e o repair reinstala o contrib por cima
    antes de derrubar o dist-info do opencv-python."""
    contrib = _add_provider(tmp_path / ".venv", "opencv-contrib-python")
    opencv = _add_provider(tmp_path / ".venv", "opencv-python")
    (tmp_path / ".venv" / "ximgproc-missing").touch()
    (tmp_path / "wheel-cached").touch()
    (tmp_path / "uv.lock").touch()
    _fake_uv(launcher)

    subprocess.run(
        [BASH, "run.sh", "--check"],
        cwd=tmp_path,
        env={"PATH": str(launcher)},
        check=True,
        timeout=10,
    )
    syncs = [c for c in _commands(tmp_path) if c.startswith("sync")]
    assert syncs == [_INSTALL_SYNC_HAND, _REPAIR_SYNC]
    assert not (tmp_path / ".venv" / "ximgproc-missing").exists()
    assert not opencv.exists()
    assert contrib.exists()
    launched = (tmp_path / "launches").read_text().strip()
    assert tmp_path / launched == tmp_path / ".venv/bin/draw"


def test_uv_repair_prefers_locked_offline_sync(tmp_path, launcher):
    """Headless por cima + cache quente: o repair reinstala o contrib
    travado por cima via `uv sync --locked --offline` e derruba o
    dist-info extra — resolve pelo uv.lock, sem listagem, sem rede e
    sem uninstall."""
    headless = _add_provider(tmp_path / ".venv", "opencv-contrib-python-headless")
    (tmp_path / ".venv" / "gui-broken").touch()
    (tmp_path / "wheel-cached").touch()
    (tmp_path / "uv.lock").touch()
    _fake_uv(launcher)

    subprocess.run(
        [BASH, "run.sh", "--check"],
        cwd=tmp_path,
        env={"PATH": str(launcher)},
        check=True,
        timeout=10,
    )
    commands = _commands(tmp_path)
    assert [c for c in commands if c.startswith("sync")] == [
        _INSTALL_SYNC_HAND,
        _REPAIR_SYNC,
    ]
    assert all("uninstall" not in c for c in commands)
    assert not headless.exists()
    launched = (tmp_path / "launches").read_text().strip()
    assert tmp_path / launched == tmp_path / ".venv/bin/draw"


def test_uv_repair_falls_back_to_network_without_cache(tmp_path, launcher):
    """Sem o wheel no cache: `sync --offline` falha e so entao vai a rede."""
    _add_provider(tmp_path / ".venv", "opencv-contrib-python-headless")
    (tmp_path / ".venv" / "gui-broken").touch()
    (tmp_path / "uv.lock").touch()
    _fake_uv(launcher)

    subprocess.run(
        [BASH, "run.sh", "--check"],
        cwd=tmp_path,
        env={"PATH": str(launcher)},
        check=True,
        timeout=10,
    )
    syncs = [c for c in _commands(tmp_path) if c.startswith("sync")]
    assert syncs == [_INSTALL_SYNC_HAND, _REPAIR_SYNC, _REPAIR_SYNC_FALLBACK]
    launched = (tmp_path / "launches").read_text().strip()
    assert tmp_path / launched == tmp_path / ".venv/bin/draw"


def test_uv_hand_retry_syncs_inexact_and_repairs_hand_env(tmp_path, launcher):
    """Todo `uv sync` do launcher leva --inexact (pytest/ruff/drawlink
    editavel sobrevivem) e o repair roda no env -hand, nao no base."""
    marker = tmp_path / ".venv" / ".no_hand"
    marker.touch()
    os.utime(marker, (1, 1))
    (tmp_path / ".venv-hand" / "bin").mkdir(parents=True)
    headless = _add_provider(tmp_path / ".venv-hand", "opencv-contrib-python-headless")
    (tmp_path / ".venv-hand" / "gui-broken").touch()
    (tmp_path / "wheel-cached").touch()
    (tmp_path / "uv.lock").touch()
    _fake_uv(launcher)

    subprocess.run(
        [BASH, "run.sh", "--check"],
        cwd=tmp_path,
        env={"PATH": str(launcher)},
        check=True,
        timeout=10,
    )
    commands = _commands(tmp_path)
    syncs = [c for c in commands if c.startswith("sync")]
    assert syncs == [_INSTALL_SYNC_HAND, _REPAIR_SYNC]
    assert all("uninstall" not in c for c in commands)
    assert not headless.exists()
    assert (tmp_path / ".venv-hand" / "hand-installed").is_file()
    launched = (tmp_path / "launches").read_text().strip()
    assert tmp_path / launched == tmp_path / ".venv-hand/bin/draw"


def test_pip_repair_keeps_single_cv2_provider(tmp_path, launcher):
    """Fallback pip: forca-reinstala o contrib por cima e derruba o
    dist-info extra — um unico dono de cv2/ registrado, sem uninstall."""
    (tmp_path / "gui-available").touch()
    (tmp_path / "hand-available").touch()
    (tmp_path / "gui-stubborn").touch()
    (tmp_path / ".venv" / "gui-broken").touch()

    subprocess.run(
        [BASH, "run.sh", "--check"],
        cwd=tmp_path,
        env={"PATH": str(launcher)},
        check=True,
        timeout=10,
    )
    commands = _commands(tmp_path)
    assert all("uninstall" not in c for c in commands)
    assert commands[-1] == (
        "-m pip install --no-deps --force-reinstall opencv-contrib-python>=4.8,<5.1"
    )
    venv = tmp_path / ".venv"
    providers = [p.name for p in (venv / "site").glob("*.dist-info")]
    assert providers == ["opencv_contrib_python-5.0.0.93.dist-info"]
    launched = (tmp_path / "launches").read_text().strip()
    assert tmp_path / launched == tmp_path / ".venv/bin/draw"


def _setup_pip_fallback(tmp_path, launcher, *names):
    """Cenario de primeiro setup via pip: base com GUI quebrada,
    pacotes disponiveis e so os pythons de sistema listados."""
    (launcher / "python3").unlink()
    for name in names:
        stub = launcher / name
        stub.write_text(_PY_STUB)
        stub.chmod(0o755)
    (tmp_path / "gui-available").touch()
    (tmp_path / "hand-available").touch()
    (tmp_path / ".venv" / "gui-broken").touch()


@pytest.mark.parametrize("stub", ["python", "py"])
@pytest.mark.parametrize("python3", ["absent", "failing"])
def test_falls_back_when_no_usable_python3(tmp_path, launcher, stub, python3):
    """Sem python3 utilizavel no PATH — ausente (Git Bash/MSYS sem
    MSYS2, ou Windows so com o launcher `py`) ou presente mas que
    aborta ao executar (stub da MS Store responde != 0 a qualquer
    `-c`): o candidato seguinte passa na sonda, cria o venv e o
    lancamento segue normal — o word-split de "py -3" e intencional."""
    _setup_pip_fallback(tmp_path, launcher, stub)
    if python3 == "failing":
        broken = launcher / "python3"
        broken.write_text("#!/bin/sh\nexit 1\n")
        broken.chmod(0o755)

    subprocess.run(
        [BASH, "run.sh", "--check"],
        cwd=tmp_path,
        env={"PATH": str(launcher)},
        check=True,
        timeout=10,
    )
    launched = (tmp_path / "launches").read_text().strip()
    assert tmp_path / launched == tmp_path / ".venv/bin/draw"


def _assert_launch_aborted(tmp_path, launcher):
    result = subprocess.run(
        [BASH, "run.sh", "--check"],
        cwd=tmp_path,
        env={"PATH": str(launcher)},
        check=False,
        timeout=10,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "erro" in result.stderr
    assert not (tmp_path / "launches").exists()


def test_no_python_aborts_with_error(tmp_path, launcher):
    """Nenhum Python >= 3.11 no PATH: erro claro com rc 1 em vez do
    127 seco do `python3` ausente — e nada lanca."""
    _setup_pip_fallback(tmp_path, launcher)
    _assert_launch_aborted(tmp_path, launcher)


def test_missing_draw_executable_aborts_with_error(tmp_path, launcher):
    """Venv sem o executavel do Draw (instalacao truncada): erro claro
    com rc 1 em vez do 127 do exec num arquivo inexistente."""
    (tmp_path / "gui-available").touch()
    (tmp_path / "hand-available").touch()
    (tmp_path / ".venv" / "bin" / "draw").unlink()
    (tmp_path / ".venv" / "gui-broken").touch()
    _assert_launch_aborted(tmp_path, launcher)


@pytest.mark.slow
def test_real_uv_plain_sync_keeps_cv2_importable(tmp_path):
    """Cache real populado so por `uv sync` travado (sem listagem de
    indice): o par contrib+headless armado e desarmado so por
    metadados, o repair offline reinstala o contrib pelo uv.lock por
    cima de um headless ativo, e syncs posteriores trocam o provedor
    sem matar `import cv2` nem a janela — tudo sem uninstall."""
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv nao esta no PATH")
    env = {
        key: value
        for key, value in os.environ.items()
        if key != "VIRTUAL_ENV" and not key.startswith("UV_")
    }
    env["UV_CACHE_DIR"] = str(tmp_path / "uvcache")
    (tmp_path / "pyproject.toml").write_text(MINI_PYPROJECT)
    lock = subprocess.run(
        [uv, "lock"],
        cwd=tmp_path,
        env=env,
        check=False,
        timeout=120,
        capture_output=True,
    )
    if lock.returncode != 0:
        pytest.skip("sem rede para resolver o lock do fixture")

    def run(args):
        subprocess.run(args, cwd=tmp_path, env=env, check=True, timeout=300)

    run([uv, "sync", "--extra", "gui"])
    python = tmp_path / ".venv" / "bin" / "python"
    # Cache de um clone que so rodou `uv sync` travado: wheel baixado,
    # sem listagem de indice — o cenario em que `uv pip install` falha.
    shutil.rmtree(tmp_path / "uvcache" / "simple-v21", ignore_errors=True)

    def drop_extras():
        run([str(python), "-c", _CV2_DROP_EXTRAS])

    def repair():
        # A sequencia real do _repair_opencv: reinstala o contrib por
        # cima e derruba os dist-infos extras — nunca desinstala.
        run(
            [
                uv,
                "sync",
                "--locked",
                "--offline",
                "--inexact",
                "--extra",
                "gui",
                "--reinstall-package",
                "opencv-contrib-python",
                "--no-install-package",
                "opencv-contrib-python-headless",
            ]
        )
        drop_extras()

    def providers():
        return subprocess.run(
            [uv, "pip", "list", "--python", str(python)],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    def gui_opens():
        return (
            subprocess.run([str(python), "-c", _CV2_GUI_CHECK], check=False).returncode
            == 0
        )

    def only_contrib():
        dists = providers()
        return "opencv-contrib-python" in dists and not any(
            name in dists
            for name in (
                "opencv-contrib-python-headless",
                "opencv-python",
                "opencv-python-headless",
            )
        )

    # Apos `uv sync --extra gui` os dois dists ficam registrados; o
    # headless pode escrever por cima (GUI morta) — o repair offline
    # reinstala o contrib por cima e desarma, sem uninstall.
    assert not only_contrib()
    repair()
    assert gui_opens()
    assert only_contrib()

    # `uv sync` exato manual: troca o provedor para o headless travado,
    # mas o cv2 segue importavel — provedor unico sai limpo.
    run([uv, "sync"])
    run([str(python), "-c", "import cv2"])
    assert not gui_opens()
    assert "opencv-contrib-python-headless" in providers()
    assert "opencv-contrib-python " not in providers()

    # O sync do proprio launcher (--inexact --extra gui
    # --no-install-package) reinstala o contrib por cima sem
    # desregistrar o headless — a janela abre, o par fica armado de
    # novo e o launcher o desarma no lancamento seguinte.
    run(
        [
            uv,
            "sync",
            "--inexact",
            "--extra",
            "gui",
            "--no-install-package",
            "opencv-contrib-python-headless",
        ]
    )
    assert gui_opens()
    assert not only_contrib()
    drop_extras()
    assert gui_opens()
    assert only_contrib()

    # `uv sync --inexact` manual (sem extra gui): mantem o contrib e
    # instala o headless por cima — a janela nao abre, o cv2 importa, e
    # o repair offline restaura o provedor unico com janela.
    run([uv, "sync", "--inexact"])
    assert not gui_opens()
    run([str(python), "-c", "import cv2"])
    repair()
    assert gui_opens()
    assert only_contrib()


@pytest.mark.slow
def test_real_launcher_lifecycle(tmp_path):
    """run.sh real contra uv real: env armado lanca e desarma so por
    metadados (um provider "herdado" via PYTHONPATH sobrevive intacto);
    depois de um `uv sync` manual que rearma com headless por cima, o
    launcher repara offline e lanca — sem uninstall em nenhum ponto."""
    uv = shutil.which("uv")
    bash = shutil.which("bash")
    if uv is None or bash is None:
        pytest.skip("uv/bash nao esta no PATH")
    env = {
        key: value
        for key, value in os.environ.items()
        if key != "VIRTUAL_ENV" and not key.startswith("UV_") and key != "PYTHONPATH"
    }
    env["UV_CACHE_DIR"] = str(tmp_path / "uvcache")
    (tmp_path / "pyproject.toml").write_text(MINI_PYPROJECT)
    lock = subprocess.run(
        [uv, "lock"],
        cwd=tmp_path,
        env=env,
        check=False,
        timeout=120,
        capture_output=True,
    )
    if lock.returncode != 0:
        pytest.skip("sem rede para resolver o lock do fixture")

    def run(args, **kw):
        subprocess.run(args, cwd=tmp_path, env=env, check=True, timeout=300, **kw)

    run([uv, "sync", "--extra", "gui"])
    python = tmp_path / ".venv" / "bin" / "python"
    site = Path(
        subprocess.run(
            [
                str(python),
                "-c",
                "import sysconfig; print(sysconfig.get_path('purelib'))",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    # Minimo para a sonda do launcher: draw.app + onnxruntime.
    (site / "onnxruntime.py").write_text("")
    (site / "draw").mkdir()
    (site / "draw" / "__init__.py").write_text("")
    (site / "draw" / "app.py").write_text(
        "import re, cv2\n"
        "def _gui_backend_compiled():\n"
        "    m = re.search(r'^[ \\t]*GUI:[ \\t]*(\\S*)', "
        "cv2.getBuildInformation(), re.M)\n"
        "    return bool(m and m.group(1).upper() "
        "not in ('NONE','NO','OFF','FALSE','0'))\n"
    )
    draw_bin = tmp_path / ".venv" / "bin" / "draw"
    draw_bin.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$0" >> launches\n[ "$1" = --check ]\n'
    )
    draw_bin.chmod(0o755)
    (tmp_path / ".venv" / ".no_hand").touch()  # pula o retry do hand
    shutil.copyfile(Path(__file__).resolve().parents[1] / "run.sh", tmp_path / "run.sh")
    foreign = tmp_path / "foreign-site"
    foreign_dist = foreign / "opencv_contrib_python_headless-9.9.9.dist-info"
    foreign_dist.mkdir(parents=True)
    (foreign_dist / "METADATA").write_text("Name: opencv-contrib-python-headless\n")
    env["PYTHONPATH"] = str(foreign)
    shutil.rmtree(tmp_path / "uvcache" / "simple-v21", ignore_errors=True)

    def providers():
        return subprocess.run(
            [uv, "pip", "list", "--python", str(python)],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    # 1. Estado armado deterministico (par registrado, contrib por
    # cima): o launcher lanca e desarma so por metadados — inclusive as
    # .libs privadas do headless sao removidas pelo RECORD, e o provider
    # herdado sobrevive.
    run(
        [
            uv,
            "sync",
            "--locked",
            "--offline",
            "--inexact",
            "--extra",
            "gui",
            "--reinstall-package",
            "opencv-contrib-python",
            "--no-install-package",
            "opencv-contrib-python-headless",
        ]
    )
    gui = subprocess.run([str(python), "-c", _CV2_GUI_CHECK], check=False)
    assert gui.returncode == 0
    assert "opencv-contrib-python-headless" in providers()
    subprocess.run(
        [bash, "run.sh", "--check"], cwd=tmp_path, env=env, check=True, timeout=60
    )
    dists = providers()
    assert "opencv-contrib-python-headless" not in dists
    assert "opencv-contrib-python " in dists
    assert not (site / "opencv_contrib_python_headless.libs").exists()
    assert foreign_dist.exists()
    assert (tmp_path / ".venv" / "bin" / "draw").name in (
        tmp_path / "launches"
    ).read_text()

    # 2. `uv sync` manual rearma com headless por cima: o launcher
    # repara offline (cache quente pelo proprio setup) e lanca.
    run([uv, "sync", "--inexact"])
    gui = subprocess.run([str(python), "-c", _CV2_GUI_CHECK], check=False)
    assert gui.returncode != 0
    subprocess.run(
        [bash, "run.sh", "--check"], cwd=tmp_path, env=env, check=True, timeout=60
    )
    dists = providers()
    assert "opencv-contrib-python-headless" not in dists
    assert "opencv-contrib-python " in dists
    gui = subprocess.run([str(python), "-c", _CV2_GUI_CHECK], check=False)
    assert gui.returncode == 0
    assert (tmp_path / "launches").read_text().count("draw") == 2
