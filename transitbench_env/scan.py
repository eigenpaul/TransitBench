"""Static scan of a canonical solution archive.

It runs on the evaluator, never executes anything and only reports; nothing is penalised. Flags:

- numeric_table: a container literal (list, tuple, set or dict display) with more than NUMERIC_LIMIT numbers,
  counted through nested literals; one flag for the outermost literal;
- blob: a str or bytes constant over BLOB_BYTES (UTF-8 bytes for str);
- decoder: a call to a base64, binascii, bytes.fromhex, codecs.decode, zlib, gzip, bz2 or lzma decoder,
  pickle or marshal loads, numpy frombuffer or fromstring, or struct unpack, with import aliases resolved;
- non_py: any file that is not .py;
- unparseable: a .py file that does not parse.

Hits are "path:line" with the path relative to app/solution ("path" alone for non_py, and for unparseable when no
line is known). Each flag keeps its first MAX_HITS hits; n_flags counts them all.

Evaluator only; pure Python. Parsing up to 1 MiB of source takes a moment, so call scan from a thread in async code.
"""

import ast
import io
import tarfile

VERSION = 1
NUMERIC_LIMIT = 200
BLOB_BYTES = 8 << 10
MAX_HITS = 50
ROOT = "app/solution/"
FLAGS = ("numeric_table", "blob", "decoder", "non_py", "unparseable")
CONTAINERS = (ast.List, ast.Tuple, ast.Set, ast.Dict)
NUMBERS = (int, float, complex)
DECODERS = {
    "base64": {"b64decode", "b32decode", "b32hexdecode", "b16decode", "a85decode", "b85decode", "z85decode",
               "standard_b64decode", "urlsafe_b64decode", "decodebytes", "decode"},
    "binascii": {"a2b_base64", "a2b_hex", "a2b_uu", "a2b_qp", "unhexlify"},
    "bytes": {"fromhex"},
    "bytearray": {"fromhex"},
    "codecs": {"decode"},
    "zlib": {"decompress", "decompressobj"},
    "gzip": {"decompress", "open", "GzipFile"},
    "bz2": {"decompress", "open", "BZ2File", "BZ2Decompressor"},
    "lzma": {"decompress", "open", "LZMAFile", "LZMADecompressor"},
    "pickle": {"loads", "load", "Unpickler"},
    "marshal": {"loads", "load"},
    "numpy": {"frombuffer", "fromstring"},
    "struct": {"unpack", "unpack_from", "iter_unpack", "Struct"},
}
# Names that mean decoding whatever object they are called on (x.fromhex, a star-imported b64decode, ...).
UNAMBIGUOUS = {"b64decode", "b32decode", "b16decode", "a85decode", "b85decode", "urlsafe_b64decode", "unhexlify",
               "a2b_base64", "a2b_hex", "frombuffer", "fromstring", "fromhex"}


def scan(tar: bytes) -> dict:
    """The scan report of a canonical solution tar:
    {"version": 1, "flags": {name: ["path:line", ...]}, "n_flags", "max_numeric_container", "max_blob_bytes"}."""
    hits = {name: [] for name in FLAGS}
    counts = dict.fromkeys(FLAGS, 0)
    largest = {"numeric": 0, "blob": 0}

    def flag(name: str, where: str) -> None:
        counts[name] += 1
        if len(hits[name]) < MAX_HITS:
            hits[name].append(where)

    try:
        with tarfile.open(fileobj=io.BytesIO(tar), mode="r:") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                path = member.name.removeprefix(ROOT)
                if not path.endswith(".py"):
                    flag("non_py", path)
                    continue
                _scan_source(path, archive.extractfile(member).read(), flag, largest)
    except tarfile.TarError:
        flag("unparseable", ".")
    return {"version": VERSION, "flags": hits, "n_flags": sum(counts.values()),
            "max_numeric_container": largest["numeric"], "max_blob_bytes": largest["blob"]}


def _scan_source(path: str, source: bytes, flag, largest: dict) -> None:
    try:
        tree = ast.parse(source, filename=path)  # bytes: honours a BOM or coding cookie like the interpreter
    except SyntaxError as e:
        flag("unparseable", f"{path}:{e.lineno}" if e.lineno else path)
        return
    except (ValueError, RecursionError, MemoryError, OverflowError):
        flag("unparseable", path)
        return
    aliases = _aliases(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
            size = len(node.value) if isinstance(node.value, bytes) else len(node.value.encode("utf-8", "surrogatepass"))
            largest["blob"] = max(largest["blob"], size)
            if size > BLOB_BYTES:
                flag("blob", f"{path}:{node.lineno}")
        elif isinstance(node, ast.Call) and _is_decoder(node.func, aliases):
            flag("decoder", f"{path}:{node.lineno}")
    for node in _outer_containers(tree):
        n = sum(isinstance(c, ast.Constant) and type(c.value) in NUMBERS for c in ast.walk(node))
        largest["numeric"] = max(largest["numeric"], n)
        if n > NUMERIC_LIMIT:
            flag("numeric_table", f"{path}:{node.lineno}")


def _outer_containers(tree: ast.AST) -> list[ast.AST]:
    """Container literals not inside another container literal, without recursion."""
    found, stack = [], [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, CONTAINERS):
            found.append(node)
            continue
        stack.extend(ast.iter_child_nodes(node))
    return found


def _aliases(tree: ast.AST) -> dict[str, str]:
    """Local name -> dotted module path for every import in the file, plus numpy's usual np."""
    names = {"np": "numpy"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                names[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for a in node.names:
                if a.name != "*":
                    names[a.asname or a.name] = f"{node.module}.{a.name}"
    return names


def _is_decoder(func: ast.AST, aliases: dict[str, str]) -> bool:
    parts = []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        parts.append(func.id)
    elif not parts:
        return False
    else:
        return parts[0] in UNAMBIGUOUS  # a call on an expression, e.g. __import__("zlib").decompress(...)
    parts.reverse()
    dotted = ".".join([aliases.get(parts[0], parts[0]), *parts[1:]]).split(".")
    return dotted[-1] in UNAMBIGUOUS or (len(dotted) > 1 and dotted[-1] in DECODERS.get(dotted[0], ()))
