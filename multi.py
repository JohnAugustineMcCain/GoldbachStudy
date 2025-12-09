#!/usr/bin/env python3

import argparse
import math
import random
import time

import gmpy2
from gmpy2 import mpz, is_prime, next_prime

SMALL_PRIMES = [mpz(3)]  # 3, 5, 7, 11, ...

def ensure_prime_index(idx: int) -> mpz:
    while len(SMALL_PRIMES) <= idx:
        SMALL_PRIMES.append(next_prime(SMALL_PRIMES[-1]))
    return SMALL_PRIMES[idx]


def goldbach_s1_first_phase(n: int, mean_sub: float | None) -> tuple[bool, int, int, int]:
    """
    Strategy 1, but we ONLY run the first zigzag phase:

      indices 0..mean_index in the pattern:
        0, mean_index, 1, mean_index-1, 2, mean_index-2, ...

    We do *not* enter the tail phase.

    We:
      - Count how many q-tests we do (total_checks),
      - Count how many decompositions we see in this phase (decomp_count),
      - Record the step where the *first* decomposition appears (first_step).

    Returns:
      (found, first_step, total_checks, decomp_count)

      found       : True iff at least one decomposition found
      first_step  : step index where the first decomp was found
                    (1-based, counting only q-tests) if found else 0
      total_checks: total number of q primality checks in this phase
      decomp_count: how many decompositions found in this phase
    """
    n_mpz = mpz(n)
    if n_mpz < 4 or n_mpz % 2 != 0:
        return False, 0, 0, 0

    # Derive mean_index from mean_sub (same as your Strat1State.__init__)
    if mean_sub is None or mean_sub <= 1:
        mean_index = 0
    else:
        mi = int(math.ceil(mean_sub)) - 1
        mean_index = mi if mi >= 0 else 0

    # Ensure primes up to mean_index exist
    ensure_prime_index(mean_index)

    low = 0
    high = mean_index
    step_idx = 0        # zigzag steps over indices
    total_checks = 0    # number of q primality tests
    decomp_count = 0
    first_step = 0

    # Zigzag over [0, mean_index] only
    while low <= high:
        if step_idx % 2 == 0:
            idx = low
            low += 1
        else:
            idx = high
            high -= 1
        step_idx += 1

        if idx < 0:
            continue

        p = ensure_prime_index(idx)
        if p >= n_mpz:
            continue

        q = n_mpz - p
        total_checks += 1
        if is_prime(q):
            decomp_count += 1
            if first_step == 0:
                first_step = total_checks

    found = (decomp_count > 0)
    return found, first_step, total_checks, decomp_count


def random_even_with_digits(rng: random.Random, digits: int) -> int:
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


def run_sweep(start_digits: int, end_digits: int, count_per_digit: int, seed: int | None) -> None:
    if start_digits > end_digits:
        start_digits, end_digits = end_digits, start_digits

    rng = random.Random(seed)

    for D in range(start_digits, end_digits + 1):
        # mean_sub is per digit length, reset for each D
        mean_sub: float | None = None
        sub_sum = 0
        sub_count = 0

        digit_sub_sum = 0
        digit_sub_count = 0
        digit_max_sub = 0

        digit_decomp_sum = 0
        digit_max_decomp = 0
        tested_n = 0       # how many n we actually sampled
        digit_skip = 0     # only for true generation failures

        start_time = time.perf_counter()

        for _ in range(count_per_digit):
            try:
                n = random_even_with_digits(rng, D)
            except ValueError:
                digit_skip = count_per_digit
                break

            tested_n += 1

            found, first_step, total_checks, decomp_count =
                goldbach_s1_first_phase(n, mean_sub)

            # Decomposition stats: we care about the count *per n*,
            # even if it's 0 or 1.
            digit_decomp_sum += decomp_count
            if decomp_count > digit_max_decomp:
                digit_max_decomp = decomp_count

            # Adapt mean_sub + subtractor stats only if a first decomp appeared
            if found and first_step > 0:
                digit_sub_sum += first_step
                digit_sub_count += 1
                if first_step > digit_max_sub:
                    digit_max_sub = first_step

                sub_sum += first_step
                sub_count += 1
                mean_sub = sub_sum / sub_count
            # If not found, we just don't update the mean; n is still counted in decompositions.

        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000.0
        ms_per_n = elapsed_ms / count_per_digit if count_per_digit > 0 else 0.0

        avg_sub = math.ceil(digit_sub_sum / digit_sub_count) if digit_sub_count > 0 else 0

        # Average decompositions per n in the first phase, including zeros:
        effective_n = tested_n if tested_n > 0 else 1
        avg_decomp_per_n = digit_decomp_sum / effective_n

        print(
            f"D: {D} | Ms: {elapsed_ms:.3f} | Ms per n: {ms_per_n:.6f} | "
            f"Max Sub: {digit_max_sub} | Avg. Sub: {avg_sub} | "
            f"Avg Dec: {avg_decomp_per_n:.3f} | Max Dec: {digit_max_decomp} | "
            f"Skip: {digit_skip}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Goldbach sampler (Strategy 1, first phase only, multi-decomp).")
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


def main() -> None:
    args = parse_args()
    try:
        start_str, end_str = args.sweep.split(":")
        start_digits = int(start_str)
        end_digits = int(end_str)
    except Exception as e:
        raise SystemExit(f"Invalid --sweep format. Expected A:B, got '{args.sweep}'.") from e

        run_sweep(start_digits, end_digits, args.count, args.seed)

if __name__ == "__main__":
    main()
