from pathlib import Path

import pytest

from transitbench import archive
from transitbench_env import scan as scan_mod
from transitbench_env.scan import scan

STARTER = Path(__file__).resolve().parents[1] / "workspace" / "solution" / "solution.py"


def report(tmp_path, files: dict[str, str | bytes]) -> dict:
    sol = tmp_path / "solution"
    for rel, body in files.items():
        (sol / rel).parent.mkdir(parents=True, exist_ok=True)
        (sol / rel).write_bytes(body if isinstance(body, bytes) else body.encode())
    c = archive.from_dir(sol)
    assert c.status == "ok"
    return scan(c.tar)


def test_report_schema_and_clean_starter(tmp_path):
    rep = report(tmp_path, {"solution.py": STARTER.read_text()})
    assert set(rep) == {"version", "flags", "n_flags", "max_numeric_container", "max_blob_bytes"}
    assert rep["version"] == 1 and set(rep["flags"]) == set(scan_mod.FLAGS)
    assert rep["n_flags"] == 0 and not any(rep["flags"].values())


def test_numeric_table(tmp_path):
    flat = "T = [" + ", ".join(f"{i}.5" for i in range(201)) + "]\n"
    nested = "x = 1\nROWS = (\n" + "".join(f"    ({i}, -{i}.25, {i}e3),\n" for i in range(70)) + ")\n"
    dict_table = "D = {" + ", ".join(f"{i}: {i}.0" for i in range(101)) + "}\n"
    rep = report(tmp_path, {"a.py": flat, "b.py": nested, "c.py": dict_table,
                            "ok.py": "T = [" + ", ".join(map(str, range(200))) + "]\n"})
    assert rep["flags"]["numeric_table"] == ["a.py:1", "b.py:2", "c.py:1"]
    assert rep["max_numeric_container"] == 210
    assert rep["n_flags"] == 3


def test_numeric_table_ignores_bools_and_strings(tmp_path):
    rep = report(tmp_path, {"a.py": "T = [" + ", ".join(["True", "'1'"] * 150) + "]\n"})
    assert rep["flags"]["numeric_table"] == [] and rep["max_numeric_container"] == 0


def test_blob(tmp_path):
    big = "x" * (8 << 10)
    rep = report(tmp_path, {"a.py": f"S = '{big}'\nB = b'{big}y'\n", "b.py": f"S = '{big[1:]}é'\n",
                            "c.py": f"S = '{big}'\n"})
    assert rep["flags"]["blob"] == ["a.py:2", "b.py:1"]  # 8193 bytes; 'é' is two UTF-8 bytes; 8192 is allowed
    assert rep["max_blob_bytes"] == (8 << 10) + 1


@pytest.mark.parametrize("source", [
    "import base64\nbase64.b64decode(x)\n",
    "import base64 as b\nb.urlsafe_b64decode(x)\n",
    "from base64 import *\nb85decode(x)\n",
    "import binascii\nbinascii.unhexlify(x)\n",
    "from binascii import a2b_base64 as f\nf(x)\n",
    "bytes.fromhex('00ff')\n",
    "import codecs\ncodecs.decode(x, 'hex')\n",
    "import zlib\nzlib.decompress(x)\n",
    "from zlib import decompress as z\nz(x)\n",
    "import gzip\ngzip.decompress(x)\n",
    "import bz2\nbz2.decompress(x)\n",
    "import lzma\nlzma.decompress(x)\n",
    "import pickle\npickle.loads(x)\n",
    "import marshal\nmarshal.loads(x)\n",
    "import numpy as np\nnp.frombuffer(x, dtype=float)\n",
    "import numpy\nnumpy.fromstring(x, sep=',')\n",
    "import struct\nstruct.unpack('<d', x)\n",
    "from struct import unpack_from\nunpack_from('<d', x)\n",
    "__import__('binascii').a2b_hex(x)\n",
])
def test_decoder(tmp_path, source):
    rep = report(tmp_path, {"solution.py": "x = b''\n" + source})
    assert rep["flags"]["decoder"] == ["solution.py:3" if source.count("\n") == 2 else "solution.py:2"]


def test_not_a_decoder(tmp_path):
    source = ("import io, json, sys\nimport numpy as np\nd = np.load(io.BytesIO(sys.stdin.buffer.read()))\n"
              "json.loads('{}')\n'x'.encode()\nb'x'.decode()\nnp.array([1, 2])\n")
    assert report(tmp_path, {"solution.py": source})["n_flags"] == 0


def test_non_py_and_unparseable(tmp_path):
    rep = report(tmp_path, {"solution.py": "print(1)\n", "model.pkl": b"\x80\x04", "data/notes.txt": "hi\n",
                            "bad.py": "x = 1\ndef (:\n", "latin.py": b"# -*- coding: latin-1 -*-\nS = '\xe9'\n",
                            "nul.py": b"x = 1\x00\n", "deep.py": "x = " + "(" * 300 + ")" * 300 + "\n"})
    assert rep["flags"]["non_py"] == ["data/notes.txt", "model.pkl"]
    assert rep["flags"]["unparseable"][0] == "bad.py:2"
    assert sorted(h.split(":")[0] for h in rep["flags"]["unparseable"]) == ["bad.py", "deep.py", "nul.py"]


def test_hits_are_capped_but_counted(tmp_path):
    source = "import zlib\n" + "zlib.decompress(b'')\n" * 120
    rep = report(tmp_path, {"solution.py": source})
    assert len(rep["flags"]["decoder"]) == scan_mod.MAX_HITS and rep["n_flags"] == 120


def test_empty_and_bad_archives():
    assert scan(archive.empty("missing").tar)["n_flags"] == 0
    assert scan(b"not a tar")["flags"]["unparseable"] == ["."]
