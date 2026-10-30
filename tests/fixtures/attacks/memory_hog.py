"""Attack: allocate and touch memory past the per-star cap. Sentinel if 1 GiB gets allocated (the jail tests run
with a 512 MiB cap). Bounded at 1 GiB."""

import sys

SENTINEL = "424242,42,42"
CHUNK = 64 << 20


def main():
    sys.stdin.buffer.read()
    held = []
    for _ in range(16):
        held.append(b"\x01" * CHUNK)
    print(SENTINEL)


if __name__ == "__main__":
    main()
