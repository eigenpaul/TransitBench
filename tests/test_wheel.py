"""The wheel test: the built wheel holds only allowlisted files, every packaged data file the env reads, and no test
star id, reserve KIC or secret (scripts/leak_needles.py). image/context/ holds none either.

The wheel is built with `uv build --wheel`. The needle scans need data/private and report SKIPPED, never PASS,
without it.
"""

import hashlib
import importlib.util
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CONTEXT = ROOT / "image" / "context"
ALLOWED = [re.compile(p) for p in (
    r"transitbench/[a-z_]+\.py",
    r"transitbench_env/[a-z_]+\.py",
    r"transitbench_env/templates/TASK\.md",
    r"transitbench_env/data/(images|mini_pool|dev_truth|dev_files|release_r1)\.json",
    r"transitbench_env/data/starter/solution\.py",
    r"transitbench_(survey|mini)/__init__\.py",
    r"transitbench-[^/]+\.dist-info/(METADATA|WHEEL|RECORD|licenses/LICENSE)",
)]
PACKAGED = {  # wheel path -> source in the repo
    "transitbench_env/data/starter/solution.py": "workspace/solution/solution.py",
    "transitbench_env/data/dev_truth.json": "dev/truth.json",
    "transitbench_env/data/dev_files.json": "dev/files.json",
    "transitbench_env/data/release_r1.json": "manifests/release_r1.json",
    "transitbench_env/data/images.json": "transitbench_env/data/images.json",
    "transitbench_env/data/mini_pool.json": "transitbench_env/data/mini_pool.json",
    "transitbench_env/templates/TASK.md": "transitbench_env/templates/TASK.md",
}
MODULES = {"transitbench": ("__init__", "archive", "data", "inject", "match", "run_dev", "runner", "score"),
           "transitbench_env": ("__init__", "box", "grading", "harness", "hidden", "mini_draw", "scan", "task"),
           "transitbench_survey": ("__init__",), "transitbench_mini": ("__init__",)}


@pytest.fixture(scope="module")
def wheel(tmp_path_factory) -> zipfile.ZipFile:
    if shutil.which("uv") is None:
        pytest.skip("needs uv")
    out = tmp_path_factory.mktemp("wheel")
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(out), str(ROOT)], check=True, capture_output=True,
                   timeout=600)
    (path,) = out.glob("transitbench-*.whl")
    with zipfile.ZipFile(path) as zf:
        yield zf


@pytest.fixture(scope="module")
def needles():
    if not (DATA / "private").is_dir():
        pytest.skip("needs data/private: the needle scan is not run, so the wheel test is not passed")
    spec = importlib.util.spec_from_file_location("leak_needles", ROOT / "scripts" / "leak_needles.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.needles(DATA)


def test_wheel_holds_only_allowlisted_files(wheel):
    names = [n for n in wheel.namelist() if not n.endswith("/")]
    assert [n for n in names if not any(p.fullmatch(n) for p in ALLOWED)] == []
    for package, modules in MODULES.items():
        assert {n for n in names if n.startswith(f"{package}/") and n.endswith(".py")
                and n.count("/") == 1} == {f"{package}/{m}.py" for m in modules}


def test_wheel_packages_the_data_the_env_reads(wheel):
    for name, source in PACKAGED.items():
        assert wheel.read(name) == (ROOT / source).read_bytes(), name


def test_wheel_requires_numpy_and_a_verifiers_floor(wheel):
    (meta,) = [n for n in wheel.namelist() if n.endswith(".dist-info/METADATA")]
    requires = [line for line in wheel.read(meta).decode().splitlines() if line.startswith("Requires-Dist:")]
    assert not [r for r in requires if "git+" in r or "@" in r]
    assert requires == ["Requires-Dist: numpy>=2.3", "Requires-Dist: verifiers>=0.4.1.dev84"]


def test_wheel_holds_no_needle(wheel, needles):
    found = {}
    for name in wheel.namelist():
        hits = needles.scan(wheel.read(name))
        if any(hits.values()):
            found[name] = hits
    assert found == {}


def test_image_context_holds_no_needle(needles):
    files = sorted(p for p in CONTEXT.rglob("*") if p.is_file()) if CONTEXT.is_dir() else []
    if not files:
        pytest.skip("no image/context (scripts/image_context.py)")
    found = {str(p.relative_to(ROOT)): h for p in files if any((h := needles.scan(p.read_bytes())).values())}
    assert found == {}


def test_needles_find_what_they_must(needles):
    """The scan is live: it finds a test id, a reserve KIC and each secret, and skips digests, float fractions and
    longer numbers. Needles never show in a report."""
    assert re.fullmatch(r"Needles\(\d+ KICs, \d+ secrets\)", repr(needles))
    kic = sorted(needles.kics)[0]
    planted = b"r1-" + b"A-0042 and r1-" + b"B-0999"  # built at run time: no test-id literal in this file
    assert needles.scan(planted)["test_id"] == 2
    assert needles.scan(b'{"kic": ' + kic + b"}")["kic"] == 1
    assert needles.scan(b"kplr" + kic.rjust(9, b"0") + b"-2011")["kic"] == 1
    assert needles.scan(b"0." + kic + b", 99" + kic + b", " + kic + b"99")["kic"] == 0
    assert needles.scan(hashlib.sha256(b"x").hexdigest().encode()[:20] + kic + b"abcdef0123")["kic"] == 0
    for secret in needles.hex_secrets:
        assert needles.scan(b"key=" + secret.upper())["secret"] == 1
    for secret in needles.raw_secrets:
        assert needles.scan(b"\x00" + secret + b"\x00")["secret"] == 1
