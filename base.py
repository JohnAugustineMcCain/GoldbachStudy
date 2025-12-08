#!/usr/bin/env python3
"""
goldbach.py

Parallel Goldbach decomposition sampler using two CPU cores.

For each even n >= 6, we look for primes p, q such that n = p + q, using
subtractor primes p generated on the fly with gmpy2.next_prime.

We run two workers in parallel:

  Core 1: starts from p = 3 and increases (3, 5, 7, 11, ...)
  Core 2: starts from the prime at index ~median_subs_so_far and increases.

We record:
  - subs_count: number of subtractor primes tested by the *winning* core.
  - Dynamic global stats: median_subs, max_subs across all n so far.
  - Timing per digit length (total ms and avg ms per n).

Example:
    python3 goldbach.py --sweep 4:10:2 --count 50 --seed 1
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
    sys.stderr.write(
        "Error: gmpy2 is required for this script.\n"
        "Install with: pip install gmpy2\n"
    )
    sys.exit(1)


# ----------------------------------------------------------------------
# Utility: sweep, sampling, stats
# ----------------------------------------------------------------------

def parse_sweep(sweep_str):
    """
    Parse a sweep spec of the form 'start:end:step' into a list of digit lengths.
    Example: '4:10:2' -> [4, 6, 8, 10]
    """
    try:
        start_str, end_str, step_str = sweep_str.split(":")
        start = int(start_str)
        end = int(end_str)
        step = int(step_str)
    except ValueError:
        raise ValueError("Sweep must be in the form start:end:step, e.g. 4:10:2")

    if start <= 0 or end <= 0 or step <= 0:
        raise ValueError("start, end, and step in sweep must be positive integers")
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


class DynamicStats:
    """
    Tracks global subs counts across all n.
    """

    def __init__(self):
        self._subs_counts = []

    def update(self, subs_count):
        self._subs_counts.append(subs_count)

    @property
    def has_data(self):
        return len(self._subs_counts) > 0

    @property
    def max_subs(self):
        return max(self._subs_counts) if self._subs_counts else 0

    @property
    def median_subs(self):
        return median(self._subs_counts) if self._subs_counts else 0.0


# ----------------------------------------------------------------------
# Worker: parallel search for a single n
# ----------------------------------------------------------------------

def worker_search(n, start_index, core_name, found_event, result_dict, lock):
    """
    Search for a Goldbach decomposition n = p + q with p, q both prime.

    - start_index: 0-based index of the subtractor prime to start from.
                  index 0 corresponds to p = 3.
    - core_name: label ("core1", "core2") for debugging.

    Primes are generated *on the fly*:
      p_0 = 3
      p_{k+1} = next_prime(p_k)

    We count how many subtractor primes this worker tests until it either:
      - finds a valid decomposition and wins the race, or
      - sees found_event set, or
      - runs out of candidates (q < 2).
    """
    n_mpz = gmpy2.mpz(n)

    # Build up to the starting prime
    p = gmpy2.mpz(3)
    for _ in range(start_index):
        p = gmpy2.next_prime(p)

    tests = 0

    while not found_event.is_set():
        q = n_mpz - p

        # If q < 2, any larger p will only make q smaller -> nothing left to do
        if q < 2:
            break

        tests += 1

        if gmpy2.is_prime(q):
            # Try to claim victory
            with lock:
                if not found_event.is_set():
                    result_dict["p"] = int(p)
                    result_dict["q"] = int(q)
                    result_dict["subs_count"] = tests
                    result_dict["core"] = core_name
                    found_event.set()
            break

        # Move to next subtractor prime
        p = gmpy2.next_prime(p)


def goldbach_parallel_for_n(n, stats, manager):
    """
    Run a two-core search for a single n using the current dynamic stats.

    Returns:
        (p, q, subs_count, winner_core)
    or
        (None, None, subs_count, None) if no decomposition was found
        (subs_count in that case is 0 or partial).
    """
    result = manager.dict()
    found_event = manager.Event()
    lock = manager.Lock()

    # Convert dynamic stats to starting indices for the cores.
    # subs_count is "number of primes tested"; index is 0-based.
    if stats.has_data:
        median_idx = max(0, int(round(stats.median_subs)) - 1)
    else:
        median_idx = 0

    # Two workers:
    #   core1: start at index 0 (p = 3)
    #   core2: start at index median_idx
    start_idx_core1 = 0
    start_idx_core2 = median_idx

    processes = [
        mp.Process(
            target=worker_search,
            args=(n, start_idx_core1, "core1", found_event, result, lock),
        ),
        mp.Process(
            target=worker_search,
            args=(n, start_idx_core2, "core2", found_event, result, lock),
        ),
    ]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    if "p" in result and "q" in result:
        return (
            result["p"],
            result["q"],
            result["subs_count"],
            result.get("core", None),
        )
    else:
        # No decomposition found (should not happen for large even n >= 6).
        return None, None, 0, None


# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Goldbach decomposition sweeper with two-core parallel search."
    )
    parser.add_argument(
        "--sweep",
        required=True,
        help="Digit-length sweep in format start:end:step (e.g., 4:10:2 for 4,6,8,10).",
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

    print(f"Sweep digit lengths: {digit_lengths}")
    print(f"Samples per digit length: {args.count}")
    print(f"Random seed: {args.seed}")
    print("")

    # Create a single Manager process to reuse across all n
    manager = mp.Manager()

    for d in digit_lengths:
        print(f"=== Digit length {d} ===")
        ns = sample_even_numbers_of_digit_length(d, args.count, rng)
        if not ns:
            print(f"No valid n values for digit length {d}, skipping.\n")
            continue

        ns.sort()

        digit_start_time = time.perf_counter()
        n_times = []

        for n in ns:
            t0 = time.perf_counter()
            p, q, subs_count, core_name = goldbach_parallel_for_n(n, stats, manager)
            t1 = time.perf_counter()
            elapsed = t1 - t0
            n_times.append(elapsed)

            if p is None or q is None:
                print(f"n={n}: NO decomposition found, subs_tried={subs_count}")
                continue

            # Update dynamic stats with this n's subs_count
            stats.update(subs_count)

            print(
                f"n={n}: p={p}, q={q}, subs_count={subs_count}, "
                f"winner={core_name}, "
                f"global_median_subs={stats.median_subs:.3f}, "
                f"global_max_subs={stats.max_subs}"
            )

        digit_end_time = time.perf_counter()
        digit_total = digit_end_time - digit_start_time
        if n_times:
            avg_per_n = sum(n_times) / len(n_times)
        else:
            avg_per_n = 0.0

        # Convert times to milliseconds for display
        avg_ms_per_n = avg_per_n * 1000.0
        digit_total_ms = digit_total * 1000.0

        # Per-digit summary snapshot of the dynamic stats
        if stats.has_data:
            print(
                f"Digit length: {d} | "
                f"Median Sub Count: {stats.median_subs:.3f} | "
                f"Max Sub Count: {stats.max_subs} | "
                f"Avg. ms per n: {avg_ms_per_n:.3f} | "
                f"Digit total ms: {digit_total_ms:.3f}"
            )
        else:
            print(
                f"Digit length: {d} | "
                f"No successful decompositions yet | "
                f"Avg. ms per n: {avg_ms_per_n:.3f} | "
                f"Digit total ms: {digit_total_ms:.3f}"
            )

        print("")

    # Final global summary
    if stats.has_data:
        print("=== Overall dynamic stats ===")
        print(f"Global Median Sub Count: {stats.median_subs:.3f}")
        print(f"Global Max Sub Count:    {stats.max_subs}")
    else:
        print("No decompositions recorded (this should not happen for even n >= 6).")


if __name__ == "__main__":
    # Multiprocessing guard (important on Windows)
    main()
