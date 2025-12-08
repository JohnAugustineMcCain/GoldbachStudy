#!/usr/bin/env python3
"""
Minimal Goldbach sampler using only Strategy 1 (small subtractor primes).

Options:
  --sweep A:B   digit range (e.g. 4:10)
  --count N     number of random even n per digit (default 1000)
  --seed S      random seed (default: None)

Output per digit:
  D: _ | Ms: _ | Ms per n: _ | Max Sub: _ | Avg. Sub: _ | Skip: _
"""

import argparse
import math
import random
import time

import gmpy2
from gmpy2 import mpz, is_prime, next_prime


PRIME_COUNT = 100000      # total subtractor primes to precompute
WINDOW_SIZE = 20          # use the last 20 digits' means to seed the next
SMALL_PRIMES = []         # filled by init_primes()


def init_primes():
    """Precompute PRIME_COUNT subtractor primes starting from 3."""
    global SMALL_PRIMES
    if SMALL_PRIMES:
        return
    p = mpz(3)
    SMALL_PRIMES = [p]
    for _ in range(PRIME_COUNT - 1):
        p = next_prime(p)
        SMALL_PRIMES.append(p)


def get_prime(idx):
    """Return the idx-th subtractor prime, or None if out of range."""
    if idx < 0 or idx >= PRIME_COUNT:
        return None
    return SMALL_PRIMES[idx]


class Strat1State:
    """
    Strategy 1: small subtractor primes with mean-based zig-zag.

    We index primes p[i] = 3,5,7,11,...
    For a given n and mean_sub, we prepare an index pattern:

        0, mean_index, 1, mean_index-1, 2, mean_index-2, ...

    then tail: mean_index+1, mean_index+2, ...

    and stop when p[i] >= n or we run out of precomputed primes.
    """

    def __init__(self, mean_sub, n):
        if mean_sub is None or mean_sub <= 1:
            mi = 0
        else:
            mi = int(math.ceil(mean_sub)) - 1
            if mi < 0:
                mi = 0
        if mi >= PRIME_COUNT:
            mi = PRIME_COUNT - 1

        self.mean_index = mi
        self.n = n
        self.low = 0
        self.high = self.mean_index
        self.phase = "zigzag"  # or "tail"
        self.tail_idx = self.mean_index + 1
        self.step = 0
        self.done = False

    def next_candidate(self):
        """Return next subtractor prime p, or None if exhausted."""
        if self.done:
            return None

        n = self.n

        while True:
            if self.phase == "zigzag":
                if self.low > self.high:
                    self.phase = "tail"
                    continue

                if self.step % 2 == 0:
                    idx = self.low
                    self.low += 1
                else:
                    idx = self.high
                    self.high -= 1

                self.step += 1
                p = get_prime(idx)
                if p is None:
                    self.phase = "tail"
                    continue

                if p >= n:
                    if self.low > self.high:
                        self.phase = "tail"
                    continue
                return p

            else:  # tail
                idx = self.tail_idx
                self.tail_idx += 1
                p = get_prime(idx)
                if p is None or p >= n:
                    self.done = True
                    return None
                return p


def goldbach_s1(n, mean_sub):
    """
    Strategy 1 only.

    For even n >= 4, search for a Goldbach decomposition n = p + q
    using small subtractor primes p.

    Returns: (found, total_checks) where total_checks is number of q primality tests.
    """
    n = mpz(n)
    if n < 4 or n % 2 != 0:
        return False, 0

    state = Strat1State(mean_sub, n)
    total_checks = 0

    while True:
        p = state.next_candidate()
        if p is None:
            break
        q = n - p
        total_checks += 1
        if is_prime(q):
            return True, total_checks

    return False, total_checks


def random_even_with_digits(rng, digits):
    """Draw a random even integer with exactly `digits` decimal digits."""
    if digits <= 0:
        raise ValueError("Digit length must be positive.")

    low = 10 ** (digits - 1)
    high = 10 ** digits - 1

    first_even = low if low % 2 == 0 else low + 1
    last_even = high if high % 2 == 0 else high - 1
    if last_even < first_even:
        raise ValueError(f"No even numbers with {digits} digits.")

    count_evens = ((last_even - first_even) // 2) + 1
    k = rng.randrange(count_evens)
    n = first_even + 2 * k
    if n < 4:
        n = 4
    return n


def run_sweep(start_digits, end_digits, count_per_digit, seed):
    if start_digits > end_digits:
        start_digits, end_digits = end_digits, start_digits

    rng = random.Random(seed)
    last_digit_means = []  # per-digit means (window of last WINDOW_SIZE digits)

    for D in range(start_digits, end_digits + 1):
        # Seed mean_sub from the last up-to-20 digit means
        if last_digit_means:
            seed_mean = sum(last_digit_means) / len(last_digit_means)
        else:
            seed_mean = None

        mean_sub = seed_mean
        sub_sum = 0
        sub_count = 0

        digit_sub_sum = 0
        digit_sub_count = 0
        digit_max_sub = 0
        digit_skip = 0

        start_time = time.perf_counter()

        for _ in range(count_per_digit):
            try:
                n = random_even_with_digits(rng, D)
            except ValueError:
                digit_skip = count_per_digit
                break

            found, total_checks = goldbach_s1(n, mean_sub)
            if found:
                digit_sub_sum += total_checks
                digit_sub_count += 1
                if total_checks > digit_max_sub:
                    digit_max_sub = total_checks

                sub_sum += total_checks
                sub_count += 1
                mean_sub = sub_sum / sub_count
            else:
                digit_skip += 1

        # Finish this digit: compute its per-digit mean and feed into window
        if sub_count > 0:
            digit_mean = sub_sum / sub_count
            last_digit_means.append(digit_mean)
            if len(last_digit_means) > WINDOW_SIZE:
                last_digit_means.pop(0)

        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000.0
        ms_per_n = elapsed_ms / count_per_digit if count_per_digit > 0 else 0.0
        avg_sub = math.ceil(digit_sub_sum / digit_sub_count) if digit_sub_count > 0 else 0

        print(
            f"D: {D} | Ms: {elapsed_ms:.3f} | Ms per n: {ms_per_n:.6f} | "
            f"Max Sub: {digit_max_sub} | Avg. Sub: {avg_sub} | Skip: {digit_skip}"
        )


def parse_args():
    p = argparse.ArgumentParser(description="Minimal Goldbach sampler (Strategy 1 only).")
    p.add_argument(
        "--sweep",
        type=str,
        required=True,
        help="Digit range A:B (e.g. 4:10).",
    )
    p.add_argument(
        "--count",
        type=int,
        default=1000,
        help="Number of random even n per digit (default 1000).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default None).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    try:
        start_str, end_str = args.sweep.split(":")
        start_digits = int(start_str)
        end_digits = int(end_str)
    except Exception as e:
        raise SystemExit(f"Invalid --sweep format. Expected A:B, got '{args.sweep}'.") from e

    init_primes()
    run_sweep(start_digits, end_digits, args.count, args.seed)


if __name__ == "__main__":
    main()
