"""Attack: print the sentinel, then flood stdout with 18 MiB of valid rows. Contained if the star gets status
stdout_overflow and no rows."""

import sys

SENTINEL = "424242,42,42"


def main():
    sys.stdin.buffer.read()
    print(SENTINEL, flush=True)
    sys.stdout.buffer.write(b"9,9,9\n" * (3 << 20))


if __name__ == "__main__":
    main()
