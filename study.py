#!/usr/bin/env python3
"""
Goldbach sampler using only Strategy 1 (small subtractor primes),
collecting as many decompositions as possible per n, with a 10x
average-per-n cutoff for total q-tests.

Options:
  --sweep A:B   digit range (e.g. 10:100)
  --count N     number of random even n per digit (default 1000)
  --seed S      random seed (default: None)

Output per digit:
  D: _ | Ms: _ | Ms per n: _ | Max Sub: _ | Avg. Sub: _ |
      AvgDec: _ | MaxDec: _ | Two+: X/Y | TwoStepAvg: _ | Cut: _ | Skip: _

Where:
  Ms         = wall-clock ms spent on this digit (all work)
  Ms per n   = average ms to find the FIRST decomposition (over n with a hit)
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
        p = next_prime(SMALL_PRIMES[-1])
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


def goldbach_s1(n, mean_sub, avg_checks_per_n):
    """
    Strategy 1 only.

    For even n >= 4, search for Goldbach decompositions n = p + q
    using small subtractor primes p.

    Behavior:
      - Search the subtractor sequence and collect as many decompositions
        as possible (up to the precomputed prime limit).
      - We track:
          * first_step      : q-test index of the first decomp
          * second_step     : q-test index of the second decomp (if any)
          * decomp_count    : total decompositions found
          * time_to_first_ms: wall-clock ms until the FIRST decomp is found
                              (or full search time if none)
      - We stop early if total q-tests for this n exceed
        10 * avg_checks_per_n (if available).

    Returns:
      (found_first, found_second,
       first_step, second_step,
       decomp_count, total_checks_for_n,
       cutoff_triggered,
       time_to_first_ms)
    """
    n = mpz(n)
    if n < 4 or n % 2 != 0:
        return False, False, 0, 0, 0, 0, False, 0.0

    state = Strat1State(mean_sub, n)
    total_checks = 0

    decomp_count = 0
    first_step = 0
    second_step = 0
    cutoff_triggered = False

    # Timing for "time to first decomp"
    start_t = time.perf_counter()
    time_to_first_ms = 0.0
    first_time_recorded = False

    # Threshold for total checks for this n
    cutoff = None
    if avg_checks_per_n is not None and avg_checks_per_n > 0:
        cutoff = 10.0 * avg_checks_per_n  # 10x instead of 3x

    while True:
        p = state.next_candidate()
        if p is None:
            break

        q = n - p
        total_checks += 1

        if is_prime(q):
            decomp_count += 1
            if decomp_count == 1:
                first_step = total_checks
                if not first_time_recorded:
                    time_to_first_ms = (time.perf_counter() - start_t) * 1000.0
                    first_time_recorded = True
            elif decomp_count == 2:
                second_step = total_checks
            # For decomp_count >= 3 we just keep counting; no extra steps tracked.

        # Check cutoff after each test
        if cutoff is not None and total_checks >= cutoff:
            cutoff_triggered = True
            break

    # If we never saw a decomp, record the full search time
    if not first_time_recorded:
        time_to_first_ms = (time.perf_counter() - start_t) * 1000.0

    found_first = (decomp_count >= 1)
    found_second = (decomp_count >= 2)

    return (
        found_first,
        found_second,
        first_step,
        second_step,
        decomp_count,
        total_checks,
        cutoff_triggered,
        time_to_first_ms,
    )


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

        # For adapting mean_sub (checks to *first* decomposition)
        sub_sum = 0
        sub_count = 0

        digit_sub_sum = 0
        digit_sub_count = 0
        digit_max_sub = 0
        digit_skip = 0

        # For "average time per n" used as cutoff (total checks per n)
        time_sum_checks = 0.0
        time_count_checks = 0

        # Decomposition statistics per digit
        digit_total_decomp_sum = 0
        digit_max_decomp = 0
        digit_two_plus = 0
        second_step_sum = 0
        second_step_count = 0

        # Cutoff statistics
        digit_cutoff_hits = 0

        tested_n = 0

        # For Ms per n based on time to first decomp
        first_time_sum_ms = 0.0
        first_time_count = 0

        digit_start_time = time.perf_counter()

        for _ in range(count_per_digit):
            try:
                n = random_even_with_digits(rng, D)
            except ValueError:
                digit_skip = count_per_digit
                break

            tested_n += 1

            # Current average total checks per n (for cutoff)
            avg_checks_per_n = (
                time_sum_checks / time_count_checks if time_count_checks > 0 else None
            )

            (
                found_first,
                found_second,
                first_step,
                second_step,
                decomp_count,
                total_checks_for_n,
                cutoff_triggered,
                time_to_first_ms,
            ) = goldbach_s1(n, mean_sub, avg_checks_per_n)

            # Update per-digit "checks" stats (used only for cutoff for *later* n)
            time_sum_checks += total_checks_for_n
            time_count_checks += 1

            # First-hit stats (adaptation + printed Avg. Sub / Max Sub)
            if found_first and first_step > 0:
                digit_sub_sum += first_step
                digit_sub_count += 1
                if first_step > digit_max_sub:
                    digit_max_sub = first_step

                sub_sum += first_step
                sub_count += 1
                mean_sub = sub_sum / sub_count

                # Timing to first decomp
                first_time_sum_ms += time_to_first_ms
                first_time_count += 1
            else:
                digit_skip += 1

            # Decomposition counts
            digit_total_decomp_sum += decomp_count
            if decomp_count > digit_max_decomp:
                digit_max_decomp = decomp_count

            if found_second:
                digit_two_plus += 1
                if second_step > 0:
                    second_step_sum += second_step
                    second_step_count += 1

            # Cutoff tracking
            if cutoff_triggered:
                digit_cutoff_hits += 1

        digit_end_time = time.perf_counter()
        elapsed_ms = (digit_end_time - digit_start_time) * 1000.0

        # Finish this digit: compute its per-digit mean and feed into window
        if sub_count > 0:
            digit_mean = sub_sum / sub_count
            last_digit_means.append(digit_mean)
            if len(last_digit_means) > WINDOW_SIZE:
                last_digit_means.pop(0)

        # Ms per n: average time to FIRST decomp (over those with a hit)
        ms_per_n = (
            first_time_sum_ms / first_time_count if first_time_count > 0 else 0.0
        )

        avg_sub = math.ceil(digit_sub_sum / digit_sub_count) if digit_sub_count > 0 else 0

        # Average decompositions per n (including zeros)
        denom_n = tested_n if tested_n > 0 else 1
        avg_dec = digit_total_decomp_sum / denom_n

        # Average step of second decomposition where it exists
        avg_second_step = (
            second_step_sum / second_step_count if second_step_count > 0 else 0.0
        )

        print(
            f"D: {D} | Ms: {elapsed_ms:.3f} | Ms per n: {ms_per_n:.6f} | "
            f"Max Sub: {digit_max_sub} | Avg. Sub: {avg_sub} | "
            f"AvgDec: {avg_dec:.3f} | MaxDec: {digit_max_decomp} | "
            f"Two+: {digit_two_plus}/{denom_n} | TwoStepAvg: {avg_second_step:.3f} | "
            f"Cut: {digit_cutoff_hits} | Skip: {digit_skip}"
        )


def parse_args():
    p = argparse.ArgumentParser(
        description="Goldbach sampler (Strategy 1, many decomps, 10x cutoff)."
    )
    p.add_argument(
        "--sweep",
        type=str,
        required=True,
        help="Digit range A:B (e.g. 10:100).",
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
        raise SystemExit(
            f"Invalid --sweep format. Expected A:B, got '{args.sweep}'."
        ) from e

    init_primes()
    run_sweep(start_digits, end_digits, args.count, args.seed)


if __name__ == "__main__":
    main()
