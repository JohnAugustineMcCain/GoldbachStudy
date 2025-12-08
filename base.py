#!/usr/bin/env python3
"""
Goldbach sampler using gmpy2.

Requirements:
    pip install gmpy2

Example usage:
    python goldbach_sampler.py --sweep 4:10 --count 1000 --seed 2

Behavior:
    - For each digit length D in the sweep, sample `count` random even n
      uniformly from [10^(D-1), 10^D - 1], restricted to even numbers >= 4.
    - For each n, search for a Goldbach decomposition n = p + q with primes p, q.
    - Subtractor primes p are tested in this pattern:
        * Maintain a global mean number of subtractor checks (mean_sub).
        * Let P = [3, 5, 7, 11, 13, 17, ...] be the odd primes list.
        * Let mean_index = ceil(mean_sub) - 1 (index into P, clamped >= 0).
        * Interleave indices "from the bottom" and "around the mean":
              low, high, low+1, high-1, ...
          where low starts at 0, high starts at mean_index.
        * After those meet, continue with indices > mean_index upward.
    - Track:
        * Max number of subtractor primes tested for any successful n
        * Average subtractor count (rounded up) for successful n
        * Number of skipped n (no decomposition found, theoretically shouldn't
          happen in the tested ranges, but handled safely)
        * Timing in milliseconds per digit length and per n.
    - No decompositions themselves are printed.
    - Output per digit length:
        D: __ | Ms: __ | Ms per n: __ | Max Sub: __ | Avg. Sub: __ | Skip: __
"""

import argparse
import math
import random
import time

import gmpy2
from gmpy2 import mpz, is_prime, next_prime


# ---------- Prime management: global list of subtractor primes ----------

# Start with the first odd prime 3
SMALL_PRIMES = [mpz(3)]


def ensure_prime_index(idx: int) -> mpz:
    """
    Ensure that SMALL_PRIMES has a prime at position `idx` and return it.
    Primes are: 3, 5, 7, 11, 13, ...
    """
    global SMALL_PRIMES
    while len(SMALL_PRIMES) <= idx:
        last = SMALL_PRIMES[-1]
        SMALL_PRIMES.append(next_prime(last))
    return SMALL_PRIMES[idx]


# ---------- Goldbach decomposition search ----------

def goldbach_subtractor_count(n: int, mean_sub: float | None) -> tuple[bool, int]:
    """
    For even n >= 4, attempt to find a Goldbach decomposition n = p + q
    where p and q are primes, and p is taken from a list of "small subtractor
    primes" ordered in a special pattern.

    The search pattern:
        - Let P[i] be the i-th subtractor prime (P[0] = 3, P[1] = 5, ...).
        - mean_sub is the global mean subtractor count so far.
        - mean_index = ceil(mean_sub) - 1 (clamped to >= 0).
        - Try primes in the order:
              0, mean_index, 1, mean_index - 1, 2, mean_index - 2, ...
          until 0..mean_index are exhausted (interleaving bottom and around mean).
        - Then continue with indices > mean_index in increasing order.

    Returns:
        (found: bool, count: int)

        * found = True if a decomposition was found, False otherwise.
        * count = number of subtractor primes tested.
    """
    n = mpz(n)

    if n < 4 or n % 2 != 0:
        # Not a valid Goldbach candidate in the standard sense
        return False, 0

    # Map mean_sub to an index in SMALL_PRIMES
    if mean_sub is None or mean_sub <= 1:
        mean_index = 0
    else:
        mean_index = int(math.ceil(mean_sub)) - 1
        if mean_index < 0:
            mean_index = 0

    # Ensure at least mean_index prime exists
    ensure_prime_index(mean_index)

    tested = 0
    visited = set()

    def try_index(idx: int) -> bool:
        """Test subtractor prime at index idx; return True if decomposition found."""
        nonlocal tested
        p = ensure_prime_index(idx)
        if p >= n:  # Once p >= n, q <= 0, no valid decomposition
            return False
        q = n - p
        tested += 1
        if is_prime(q):
            return True
        return False

    # Interleave around the mean:
    low = 0
    high = mean_index

    while low <= high:
        # From the bottom (very small subtractor primes)
        if low not in visited:
            visited.add(low)
            if try_index(low):
                return True, tested

        # From around the mean (jump to larger subtractor primes)
        if high not in visited and high >= 0:
            visited.add(high)
            if try_index(high):
                return True, tested

        low += 1
        high -= 1

    # After we've exhausted 0..mean_index in the zig-zag pattern,
    # continue from mean_index+1 upward.
    idx = mean_index + 1
    while True:
        p = ensure_prime_index(idx)
        if p >= n:
            # No more possible subtractor primes
            break
        if try_index(idx):
            return True, tested
        idx += 1

    # If we get here, no decomposition was found (highly unlikely in practical ranges)
    return False, tested


# ---------- Sampling logic per digit length ----------

def random_even_with_digits(rng: random.Random, digits: int) -> int:
    """
    Draw a random even integer with exactly `digits` decimal digits.
    Ensures n >= 4 (so Goldbach is meaningful).
    """
    if digits <= 0:
        raise ValueError("Digit length must be positive.")

    low = 10 ** (digits - 1)
    high = 10 ** digits - 1

    # First even in [low, high]
    first_even = low if low % 2 == 0 else low + 1
    # Last even in [low, high]
    last_even = high if high % 2 == 0 else high - 1

    if last_even < first_even:
        # No evens (can happen if digits=1 and range is 1..9 but we don't care about
        # digits=1 for Goldbach; 2 is even but <4, so not useful for our purposes)
        raise ValueError(f"No usable even numbers in digit range {digits}.")

    # Number of even values in this range:
    count_evens = ((last_even - first_even) // 2) + 1

    # Pick k from [0, count_evens-1]
    k = rng.randrange(count_evens)
    n = first_even + 2 * k

    # Ensure n >= 4
    if n < 4:
        n += 2 * ((4 - n + 1) // 2)  # bump up to at least 4, keeping even
        if n > last_even:
            # If we somehow stepped out of range, wrap back (should be rare)
            n = first_even

    return n


# ---------- Main sweep logic ----------

def run_sweep(start_digits: int, end_digits: int, count_per_digit: int, seed: int | None) -> None:
    """
    Run the Goldbach sampler across digit lengths [start_digits, end_digits].

    For each digit length D:
        - Sample `count_per_digit` random even n with D digits.
        - Find Goldbach decompositions using the special subtractor order.
        - Collect statistics and print summary.
    """
    if start_digits > end_digits:
        start_digits, end_digits = end_digits, start_digits

    rng = random.Random(seed)

    # Global stats for mean subtractor count that carries across digits
    global_mean_sub: float | None = None
    global_sub_sum = 0
    global_sub_count = 0

    for D in range(start_digits, end_digits + 1):
        # Per-digit stats
        digit_sub_sum = 0
        digit_sub_count = 0
        digit_max_sub = 0
        digit_skip = 0

        start_time = time.perf_counter()

        for _ in range(count_per_digit):
            try:
                n = random_even_with_digits(rng, D)
            except ValueError:
                # If no valid evens for this digit length, skip the whole digit
                digit_skip = count_per_digit
                break

            found, sub_count = goldbach_subtractor_count(n, global_mean_sub)

            if found:
                digit_sub_sum += sub_count
                digit_sub_count += 1
                if sub_count > digit_max_sub:
                    digit_max_sub = sub_count

                # Update global mean stats
                global_sub_sum += sub_count
                global_sub_count += 1
                global_mean_sub = global_sub_sum / global_sub_count
            else:
                # In theory, this should not happen often for reasonable ranges.
                digit_skip += 1

        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000.0
        ms_per_n = elapsed_ms / count_per_digit if count_per_digit > 0 else 0.0

        if digit_sub_count > 0:
            avg_sub = math.ceil(digit_sub_sum / digit_sub_count)
        else:
            avg_sub = 0

        # Print the summary line for this digit length
        print(
            f"D: {D} | "
            f"Ms: {elapsed_ms:.3f} | "
            f"Ms per n: {ms_per_n:.6f} | "
            f"Max Sub: {digit_max_sub} | "
            f"Avg. Sub: {avg_sub} | "
            f"Skip: {digit_skip}"
        )


# ---------- Argument parsing ----------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Goldbach sampler with small subtractor primes and mean-based search."
    )
    parser.add_argument(
        "--sweep",
        type=str,
        required=True,
        help="Digit range in the form START:END (e.g., 4:10).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1000,
        help="Number of random even n per digit length (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Parse the sweep argument "A:B"
    try:
        start_str, end_str = args.sweep.split(":")
        start_digits = int(start_str)
        end_digits = int(end_str)
    except Exception as e:
        raise SystemExit(f"Invalid --sweep format. Expected A:B, got '{args.sweep}'.") from e

    run_sweep(
        start_digits=start_digits,
        end_digits=end_digits,
        count_per_digit=args.count,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
