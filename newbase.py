#!/usr/bin/env python3
"""
goldbach.py

Goldbach decomposition sampler using two persistent worker processes.

For each even n >= 6, we search for primes p, q such that n = p + q
with n = p + q and p, q prime.

We use subtractor primes p indexed as:
    index 0 -> 3
    index 1 -> 5
    index 2 -> 7
    index 3 -> 11
    ...

Two cores, two phases per n:

Phase 1:
    - Let max_idx = max_subs_so_far - 1 (or 0 if no history).
    - Let mid = max_idx // 2.
    - Core 1 tests indices 0 .. mid (ascending).
    - Core 2 tests indices max_idx .. mid+1 (descending).
    They "work towards each other" over [0, max_idx].

Phase 2 (if Phase 1 finds no decomposition):
    - Core 1 tests indices max_idx+1, max_idx+3, max_idx+5, ...
    - Core 2 tests indices max_idx+2, max_idx+4, max_idx+6, ...
    They "expand max" together until a decomposition is found.

Only per-digit stats are printed:

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
# Sweep & sampling helpers
# ---------------------------------------------------------------

def parse_sweep(sweep_str):
    try:
        start_str, end_str, step_str = sweep_str.split(":")
        start = int(start_str)
        end = int(end_str)
        step = int(step_str)
    except Exception:
        raise ValueError("Sweep must be start:end:step  (example: 100:120:1)")

    if start <= 0 or end <= 0 or step <= 0:
        raise ValueError("start, end, and step must be positive integers")
    if start > end:
        raise ValueError("sweep start must be <= end")

    return list(range(start, end + 1, step))


def sample_even_numbers_of_digit_length(d, count, rng):
    """
    Sample 'count' even integers n with exactly d decimal digits, n >= 6.
    Uniform among evens in the range.
    """
    start = 10 ** (d - 1)
    end = 10 ** d - 1
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
# Global dynamic stats (master process only)
# ---------------------------------------------------------------

class DynamicStats:
    def __init__(self):
        self._subs = []

    def update(self, subs_count: int):
        self._subs.append(subs_count)

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
# Worker process logic
# ---------------------------------------------------------------

def worker_loop(worker_id, task_queue, result_queue, winner_event, winner_subs_count, winner_lock):
    """
    Persistent worker loop.

    Receives tasks of the form:

        ("phase1", n, start_idx, end_idx, step)
        ("phase2", n, start_idx, step)

    For each n, the worker maintains a running count of how many subtractor
    primes it has tested for that n. When a worker finds a valid decomposition,
    it tries to claim victory by setting winner_subs_count and winner_event.

    Prime generation is done by index, with a small cached list of subtractor
    primes: [3, 5, 7, 11, ...]. This is tiny (thousands of primes at most).
    """
    # Local prime cache by index: index 0 -> 3, 1 -> 5, 2 -> 7, ...
    sub_primes = [gmpy2.mpz(3)]
    tests_per_n = {}  # n (int) -> number of tests this worker has done for that n

    def ensure_prime_index(idx):
        """Ensure sub_primes[idx] exists."""
        while len(sub_primes) <= idx:
            sub_primes.append(gmpy2.next_prime(sub_primes[-1]))

    while True:
        task = task_queue.get()
        if task is None:
            # Sentinel: shut down worker
            break

        mode = task[0]

        if mode == "phase1":
            _, n, start_idx, end_idx, step = task
            n_int = int(n)
            n_mpz = gmpy2.mpz(n)
            tests = tests_per_n.get(n_int, 0)

            if step > 0:
                cond = lambda i: i <= end_idx
            else:
                cond = lambda i: i >= end_idx

            i = start_idx
            while cond(i) and not winner_event.is_set():
                ensure_prime_index(i)
                p = sub_primes[i]
                q = n_mpz - p
                if q < 2:
                    break

                tests += 1
                if gmpy2.is_prime(q):
                    with winner_lock:
                        if not winner_event.is_set():
                            winner_subs_count.value = tests
                            winner_event.set()
                    break

                i += step

            tests_per_n[n_int] = tests
            result_queue.put(1)

        elif mode == "phase2":
            _, n, start_idx, step = task
            n_int = int(n)
            n_mpz = gmpy2.mpz(n)
            tests = tests_per_n.get(n_int, 0)

            i = start_idx
            while not winner_event.is_set():
                ensure_prime_index(i)
                p = sub_primes[i]
                q = n_mpz - p
                if q < 2:
                    break

                tests += 1
                if gmpy2.is_prime(q):
                    with winner_lock:
                        if not winner_event.is_set():
                            winner_subs_count.value = tests
                            winner_event.set()
                    break

                i += step

            tests_per_n[n_int] = tests
            result_queue.put(1)

        else:
            # Unknown mode; ignore
            result_queue.put(1)


# ---------------------------------------------------------------
# Per-n coordination using the persistent workers
# ---------------------------------------------------------------

def goldbach_parallel_for_n(n, stats, ctx):
    """
    Coordinate the two persistent workers to process a single n, using:

      Phase 1:
        Core 1: indices [0 .. mid] ascending
        Core 2: indices [max_idx .. mid+1] descending

      Phase 2 (if needed):
        Core 1: max_idx+1, max_idx+3, max_idx+5, ...
        Core 2: max_idx+2, max_idx+4, max_idx+6, ...

    Returns:
        subs_count (int): number of subtractor primes tested by the winning core,
                          or 0 if, unexpectedly, no decomposition is found.
    """
    task_queue = ctx["task_queue"]
    result_queue = ctx["result_queue"]
    winner_event = ctx["winner_event"]
    winner_subs_count = ctx["winner_subs_count"]

    # Reset winner state for this n
    winner_event.clear()
    winner_subs_count.value = 0

    # Determine current frontier
    if stats.has_data:
        max_idx = max(0, int(stats.max_subs) - 1)
    else:
        max_idx = 0

    # -------------------- Phase 1 --------------------
    mid = max_idx // 2

    # Core 1: indices 0 .. mid (ascending)
    phase1_task_core1 = ("phase1", n, 0, mid, +1)

    # Core 2: indices max_idx .. mid+1 (descending), but only if max_idx >= 1
    if max_idx >= 1:
        phase1_task_core2 = ("phase1", n, max_idx, mid + 1, -1)
    else:
        # Degenerate: nothing to do in Phase 1 for core2
        phase1_task_core2 = ("phase1", n, 0, -1, -1)  # empty range

    task_queue.put(phase1_task_core1)
    task_queue.put(phase1_task_core2)

    # Wait for both workers to finish Phase 1
    result_queue.get()
    result_queue.get()

    if winner_event.is_set():
        return int(winner_subs_count.value)

    # -------------------- Phase 2 (expand beyond max) --------------------
    # Core 1: odd offsets above max_idx
    # Core 2: even offsets above max_idx
    start1 = max_idx + 1
    start2 = max_idx + 2

    phase2_task_core1 = ("phase2", n, start1, +2)
    phase2_task_core2 = ("phase2", n, start2, +2)

    task_queue.put(phase2_task_core1)
    task_queue.put(phase2_task_core2)

    result_queue.get()
    result_queue.get()

    return int(winner_subs_count.value)


# ---------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Goldbach decomposition sweeper with two persistent workers and two-phase search."
    )
    parser.add_argument(
        "--sweep",
        required=True,
        help="Digit-length sweep in format start:end:step (e.g., 100:120:1).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="Number of sampled n values per digit length (default: 50).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None).",
    )

    args = parser.parse_args()

    try:
        digit_lengths = parse_sweep(args.sweep)
    except ValueError as e:
        sys.stderr.write(f"Error parsing --sweep: {e}\n")
        sys.exit(1)

    rng = random.Random(args.seed)
    stats = DynamicStats()

    # Shared IPC primitives & worker context
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    winner_event = mp.Event()
    winner_subs_count = mp.Value("i", 0)
    winner_lock = mp.Lock()

    ctx = {
        "task_queue": task_queue,
        "result_queue": result_queue,
        "winner_event": winner_event,
        "winner_subs_count": winner_subs_count,
        "winner_lock": winner_lock,
    }

    # Start two persistent worker processes
    workers = []
    for wid in range(2):
        p = mp.Process(
            target=worker_loop,
            args=(wid, task_queue, result_queue, winner_event, winner_subs_count, winner_lock),
        )
        p.start()
        workers.append(p)

    print(f"Sweep digit lengths: {digit_lengths}")
    print(f"Samples per digit length: {args.count}")
    print(f"Random seed: {args.seed}")
    print("")

    try:
        for d in digit_lengths:
            ns = sample_even_numbers_of_digit_length(d, args.count, rng)
            if not ns:
                print(f"Digit length: {d} | No valid n values.\n")
                continue

            ns.sort()

            digit_start = time.perf_counter()
            n_times = []

            for n in ns:
                t0 = time.perf_counter()
                subs = goldbach_parallel_for_n(n, stats, ctx)
                t1 = time.perf_counter()

                n_times.append(t1 - t0)

                if subs > 0:
                    stats.update(subs)
                # If subs == 0, no decomposition found (should not happen).

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

    finally:
        # Cleanly shut down workers
        for _ in workers:
            task_queue.put(None)   # Sentinel for each worker
        for p in workers:
            p.join()


if __name__ == "__main__":
    main()
