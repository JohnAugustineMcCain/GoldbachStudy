#!/usr/bin/env python3
"""
goldbach.py

Parallel Goldbach decomposition sampler using two CPU cores.

Only per-digit statistics are printed.

For each even n >= 6, we look for primes p, q such that n = p + q.
Two workers search in parallel:
    Core 1: subtractor primes from index 0 upward (p = 3, 5, 7, …)
    Core 2: subtractor primes starting at current median_subs index.

Primes are generated on-demand using gmpy2.next_prime().

After each digit length, prints:
    Digit length: D | Median Sub Count: X | Max Sub Count: Y |
    Avg. ms per n: Z | Digit total ms: W
"""

import argparse
import random
import sys
import time
from statistics import median
import multiprocessing as mp

try:
    import gmpy2
except ImportError:
    sys.stderr.write("Install gmpy2 with: pip install gmpy2\n")
    sys.exit(1)


# ---------------------------------------------------------------
# Sweep & sampling
# ---------------------------------------------------------------

def parse_sweep(sweep_str):
    try:
        s, e, st = sweep_str.split(":")
        return list(range(int(s), int(e) + 1, int(st)))
    except Exception:
        raise ValueError("Sweep must be start:end:step  (example: 4:10:2)")


def sample_even_numbers_of_digit_length(d, count, rng):
    start = 10**(d - 1)
    end = 10**d - 1
    start = max(start, 6)

    if start % 2 != 0:
        start += 1
    if end % 2 != 0:
        end -= 1

    if start > end:
        return []

    num_evens = (end - start) // 2 + 1
    return [start + 2 * rng.randint(0, num_evens - 1) for _ in range(count)]


# ---------------------------------------------------------------
# Global dynamic stats
# ---------------------------------------------------------------

class DynamicStats:
    def __init__(self):
        self._subs = []

    def update(self, c):
        self._subs.append(c)

    @property
    def has_data(self):
        return len(self._subs) > 0

    @property
    def max_subs(self):
        return max(self._subs) if self._subs else 0

    @property
    def median_subs(self):
        return median(self._subs) if self._subs else 0.0


# ---------------------------------------------------------------
# Worker: parallel subtractor search
# ---------------------------------------------------------------

def worker_search(n, start_idx, found_event, result_dict, lock):
    """
    Worker searches primes on-demand:
        p0 = 3
        pk = next_prime(pk-1)

    start_idx = starting subtractor index (0-based).
    """
    n_mpz = gmpy2.mpz(n)

    # Build first prime for this worker
    p = gmpy2.mpz(3)
    for _ in range(start_idx):
        p = gmpy2.next_prime(p)

    tests = 0

    while not found_event.is_set():
        q = n_mpz - p

        if q < 2:
            break

        tests += 1

        if gmpy2.is_prime(q):
            # Try to claim result
            with lock:
                if not found_event.is_set():
                    result_dict["subs_count"] = tests
                    found_event.set()
            break

        p = gmpy2.next_prime(p)


def goldbach_parallel_for_n(n, stats, manager):
    """
    Launch 2 real processes racing to find a decomposition for n.
    Returns subs_count (or 0 if none).
    """
    result = manager.dict()
    found_event = manager.Event()
    lock = manager.Lock()

    # Dynamic starting index for Core 2
    median_idx = max(0, int(round(stats.median_subs)) - 1) if stats.has_data else 0

    # Two cores
    p1 = mp.Process(target=worker_search, args=(n, 0, found_event, result, lock))
    p2 = mp.Process(target=worker_search, args=(n, median_idx, found_event, result, lock))

    p1.start()
    p2.start()
    p1.join()
    p2.join()

    return result.get("subs_count", 0)


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Goldbach parallel sweeper (quiet mode)")
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    digit_lengths = parse_sweep(args.sweep)
    stats = DynamicStats()

    manager = mp.Manager()

    print(f"Sweep digit lengths: {digit_lengths}")
    print(f"Samples per digit length: {args.count}")
    print(f"Random seed: {args.seed}")
    print("")

    for d in digit_lengths:
        ns = sample_even_numbers_of_digit_length(d, args.count, rng)
        ns.sort()

        digit_start = time.perf_counter()
        n_times = []

        for n in ns:
            t0 = time.perf_counter()
            subs = goldbach_parallel_for_n(n, stats, manager)
            t1 = time.perf_counter()

            n_times.append(t1 - t0)

            if subs > 0:
                stats.update(subs)

        digit_end = time.perf_counter()

        avg_ms = (sum(n_times) / len(n_times) * 1000.0) if n_times else 0.0
        total_ms = (digit_end - digit_start) * 1000.0

        print(
            f"Digit length: {d} | "
            f"Median Sub Count: {stats.median_subs:.3f} | "
            f"Max Sub Count: {stats.max_subs} | "
            f"Avg. ms per n: {avg_ms:.3f} | "
            f"Digit total ms: {total_ms:.3f}"
        )
        print("")


if __name__ == "__main__":
    main()
