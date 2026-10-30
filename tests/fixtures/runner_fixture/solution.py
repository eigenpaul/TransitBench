"""Runner fixture: a two-file solution that uses multiprocessing.

The sha256 of the star's bytes picks the behaviour (helper.mode): sleep past any cap, raise, or print two rows that
encode what the sandbox allows:
  period = 1 + helper.probe() flags, t0 = uid,                        score = 3
  period = CPUs in the affinity mask, t0 = a multiprocessing.Pool(2) sum, score = 2
"""

import multiprocessing
import sys
import time

import helper


def main():
    mode = helper.mode(sys.stdin.buffer.read())
    if mode == "sleep":
        time.sleep(3600)
    if mode == "raise":
        raise RuntimeError("runner fixture: raise")
    flags = helper.probe()
    with multiprocessing.Pool(2) as pool:
        total = sum(pool.map(helper.square, range(10)))
    print(f"{1 + flags},{helper.uid()},3")
    print(f"{helper.cpus()},{total},2")


if __name__ == "__main__":
    main()
