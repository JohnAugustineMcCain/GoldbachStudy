#!/usr/bin/env python3
"""
goldbach.py

Experimentally probes Goldbach decompositions by sampling even n in digit-length
ranges and searching for prime decompositions n = p + q using "subtractor primes"
p tested in increasing order (3, 5, 7, 11, ...).

Example:
    python3 goldbach.py --sweep 4:10:2 --count 50 --seed 1
"""

import argparse
import random
import math
import sys
from collections import defaultdict
from statistics import median

try:
    import gmpy2
except ImportError:
    sys.stderr.write(
        "Error: gmpy2 is required for this script.\n"
        "Install with: pip install gmpy2\n"
    )
    sys.exit(1)


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

    digit_lengths = list(range(start, end + 1, step))
    return digit_lengths


class SubtractorPrimeGenerator:
    """
    Maintains a growing list of subtractor primes p (3, 5, 7, 11, ...)
    using gmpy2.next_prime(). The list is extended as needed up to some
    maximum p requested.
    """

    def __init__(self):
        self.sub_primes = []
        self._init_first_prime()

    def _init_first_prime(self):
        # First subtractor prime is 3
        self.sub_primes.append(gmpy2.mpz(3))

    def ensure_up_to(self, max_p):
        """
        Ensure that all subtractor primes up to max_p are generated.
        """
        max_p = gmpy2.mpz(max_p)
        while self.sub_primes[-1] < max_p:
            next_p = gmpy2.next_prime(self.sub_primes[-1])
            self.sub_primes.append(next_p)

    def iter_subtractors(self, upper_bound):
        """
        Yield subtractor primes p in increasing order, up to upper_bound (inclusive).
        """
        if upper_bound < 3:
            return
        self.ensure_up_to(upper_bound)
        for p in self.sub_primes:
            if p > upper_bound:
                break
            yield p


def sample_even_numbers_of_digit_length(d, count, rng):
    """
    Sample 'count' distinct (or not necessarily distinct) even integers n
    with exactly d decimal digits, n >= 6.

    We sample uniformly among the even numbers in the digit-length range.
    """
    # Range of numbers with d digits
    start = 10 ** (d - 1)
    end = 10 ** d - 1

    # Ensure start is at least 6
    start = max(start, 6)

    # Make start even
    if start % 2 != 0:
        start += 1

    # Ensure end is even (largest even <= end)
    if end % 2 != 0:
        end -= 1

    if start > end:
        # No valid n for this digit length
        return []

    # Number of even values in [start, end]
    num_evens = (end - start) // 2 + 1

    samples = []
    for _ in range(count):
        # Choose an index uniformly in [0, num_evens - 1]
        k = rng.randint(0, num_evens - 1)
        n = start + 2 * k
        samples.append(n)

    return samples


def goldbach_decomposition(n, subgen):
    """
    For an even integer n >= 6, find primes p and q such that n = p + q,
    by testing subtractor primes p sequentially: 3, 5, 7, 11, ...

    Returns:
        (p, q, tests) where:
            p, q are gmpy2.mpz primes with p + q = n
            tests is number of subtractor primes tested (including the successful one)

    If no decomposition is found up to p <= n - 3 (q >= 2), returns (None, None, tests).
    """
    n_mpz = gmpy2.mpz(n)
    tests = 0

    # Maximum subtractor p we ever need to consider is n - 2
    # (since q = n - p >= 2).
    max_p = n_mpz - 2

    for p in subgen.iter_subtractors(max_p):
        tests += 1
        q = n_mpz - p
        if q < 2:
            # No valid q from here on
            break
        if gmpy2.is_prime(q):
            return p, q, tests

    return None, None, tests


def main():
    parser = argparse.ArgumentParser(
        description="Goldbach decomposition sweeper with dynamic subtractor primes."
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

    subgen = SubtractorPrimeGenerator()

    # Stats: for each digit length, store list of test counts
    tests_per_digit = defaultdict(list)
    # Also keep global counts
    global_test_counts = []

    print(f"Sweep digit lengths: {digit_lengths}")
    print(f"Samples per digit length: {args.count}")
    print(f"Random seed: {args.seed}")
    print("")

    for d in digit_lengths:
        print(f"=== Digit length {d} ===")
        ns = sample_even_numbers_of_digit_length(d, args.count, rng)

        if not ns:
            print(f"No valid n values for digit length {d}, skipping.")
            print("")
            continue

        # For stable, "sort-of evenly distributed" feel, we can sort the sampled ns
        # (they are still randomly drawn).
        ns.sort()

        for n in ns:
            p, q, tests = goldbach_decomposition(n, subgen)

            if p is None:
                print(f"n={n}: NO decomposition found after {tests} tests")
            else:
                print(f"n={n}: p={p}, q={q}, tests={tests}")
                tests_per_digit[d].append(tests)
                global_test_counts.append(tests)

        # Per-digit summary
        if tests_per_digit[d]:
            med = median(tests_per_digit[d])
            avg = sum(tests_per_digit[d]) / len(tests_per_digit[d])
            print(
                f"Digit length {d} summary: "
                f"samples={len(tests_per_digit[d])}, "
                f"median_tests={med}, "
                f"mean_tests={avg:.3f}"
            )
        else:
            print(f"Digit length {d} summary: no successful decompositions recorded.")

        print("")

    # Global summary
    print("=== Overall summary ===")
    if global_test_counts:
        global_median = median(global_test_counts)
        global_mean = sum(global_test_counts) / len(global_test_counts)
        print(
            f"Total successful decompositions: {len(global_test_counts)}\n"
            f"Global median tests: {global_median}\n"
            f"Global mean tests: {global_mean:.3f}"
        )
    else:
        print("No successful decompositions recorded overall (this should not happen for even n >= 6).")


if __name__ == "__main__":
    main()
