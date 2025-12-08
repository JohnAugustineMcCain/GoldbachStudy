#!/usr/bin/env python3
"""
Goldbach sampler with racing strategies, dynamic histograms, and per-strategy stats.

Requirements:
    pip install gmpy2

Example usage:
    python3 gold3.py --sweep 14:20 --count 10000 --seed 1

Behavior:
    - For each digit length D in the sweep, sample `count` random even n
      uniformly from [10^(D-1), 10^D - 1], restricted to even numbers >= 4.

    - For each n, run a "race" between two strategies:

      Strategy 1 (Strat1): small subtractor primes with mean-based zig-zag
        * Maintain a global mean subtractor count mean_sub.
        * Let P[i] be the i-th odd prime (P[0] = 3, P[1] = 5, ...).
        * mean_index = ceil(mean_sub) - 1 (clamped to >= 0).
        * Try indices in order:
              0, mean_index, 1, mean_index - 1, 2, mean_index - 2, ...
          (interleaving very small primes and around the mean index).
        * After exhausting [0 .. mean_index], continue with indices
          mean_index+1, mean_index+2, ... while P[i] < n.

      Strategy 2 (Strat2): subtractor primes near n/2 with mean-distance zig-zag
        * Let mid = n/2.
        * Start from the first odd >= mid.
        * Maintain a global mean "step distance" mean_mid_steps.
          A "step" is +2 from the starting odd:
              step k --> p = start + 2*k
        * mean_k = ceil(mean_mid_steps).
        * Try k in order:
              0, mean_k, 1, mean_k - 1, 2, mean_k - 2, ...
          (interleaving center and around average distance).
        * After exhausting [0 .. mean_k], continue with k = mean_k+1, mean_k+2, ...
          while p < n.

      Race logic:
        * Alternate between Strat1 and Strat2:
            - Take one candidate from Strat1, test it.
            - Take one candidate from Strat2, test it.
            - Repeat until one strategy finds a decomposition n = p + q
              with both p and q prime, or both strategies run out.
        * Maintain a set of tested p so we never test the same subtractor twice
          even if both strategies propose it.
        * The "subtractor count" for this n is the TOTAL number of tested p
          across both strategies before success.

    - Statistics (per digit D):
        * Max Sub: maximum total subtractor count over successful n.
        * Avg. Sub: ceil(average total subtractor count over successful n).
        * Skip: number of n that ended with no decomposition found.
        * Strat1 wins / Strat2 wins.
        * Mean s1_count: average # of candidates Strat1 tested per successful n.
        * Mean s2_count: average # of candidates Strat2 tested per successful n.
        * Histograms of total subtractor counts:
            - All successful n.
            - Only cases where Strat1 wins.
            - Only cases where Strat2 wins.

    - Global stats:
        * Total Strat1 wins, total Strat2 wins.
        * Global mean mid-distance steps for Strat2 when it wins:
              step distance = |p - n/2| / 2

    - Output per digit length (main line, unchanged format):
        D: __ | Ms: __ | Ms per n: __ | Max Sub: __ | Avg. Sub: __ | Skip: __
"""

import argparse
import math
import random
import time

import gmpy2
from gmpy2 import mpz, is_prime, next_prime


# ---------- Global list of small subtractor primes for Strategy 1 ----------

SMALL_PRIMES = [mpz(3)]  # P[0] = 3, then 5, 7, 11, ...


def ensure_prime_index(idx: int) -> mpz:
    """
    Ensure SMALL_PRIMES has a prime at position idx and return it.
    """
    global SMALL_PRIMES
    while len(SMALL_PRIMES) <= idx:
        last = SMALL_PRIMES[-1]
        SMALL_PRIMES.append(next_prime(last))
    return SMALL_PRIMES[idx]


# ---------- Strategy 1: small subtractor primes with mean-based zig-zag ----------

class Strat1State:
    """
    Strategy 1 state:

    Uses a global list of odd primes P[i] starting from 3.
    For a given n and mean_sub, constructs an index pattern:
        0, mean_index, 1, mean_index-1, 2, mean_index-2, ...
    then tail: mean_index+1, mean_index+2, ...
    stopping when P[i] >= n.
    """

    def __init__(self, mean_sub: float | None, n: mpz):
        if mean_sub is None or mean_sub <= 1:
            self.mean_index = 0
        else:
            mi = int(math.ceil(mean_sub)) - 1
            self.mean_index = mi if mi >= 0 else 0

        # Extend the prime list up to mean_index
        ensure_prime_index(self.mean_index)

        self.low = 0
        self.high = self.mean_index
        self.phase = "zigzag"   # "zigzag" or "tail"
        self.tail_idx = self.mean_index + 1
        self.n = n
        self.done = False
        self.step = 0

    def next_candidate(self) -> mpz | None:
        """
        Return the next subtractor prime p for this strategy, or None if done.
        """
        if self.done:
            return None

        n = self.n

        while True:
            if self.phase == "zigzag":
                if self.low > self.high:
                    # Finished interleaving [0 .. mean_index], move to tail
                    self.phase = "tail"
                    continue

                if self.step % 2 == 0:
                    idx = self.low
                    self.low += 1
                else:
                    idx = self.high
                    self.high -= 1

                self.step += 1

                if idx < 0:
                    continue

                p = ensure_prime_index(idx)
                if p >= n:
                    # Primes at or above n are useless for n - p = q
                    if self.low > self.high:
                        self.phase = "tail"
                    continue

                return p

            else:  # tail phase
                idx = self.tail_idx
                self.tail_idx += 1

                p = ensure_prime_index(idx)
                if p >= n:
                    # No more possible primes for this strategy
                    self.done = True
                    return None

                return p


# ---------- Strategy 2: around n/2 with mean-distance zig-zag ----------

class Strat2State:
    """
    Strategy 2 state:

    Let mid = n // 2. Start from the first odd >= mid ("center" p0).
    Define steps k >= 0:
        step k → p = start + 2*k

    With a global mean_mid_steps, we define mean_k = ceil(mean_mid_steps).
    Then we use the pattern:
        k = 0, mean_k, 1, mean_k-1, 2, mean_k-2, ...
    and then tail: mean_k+1, mean_k+2, ...
    Any candidate p >= n is invalid (q <= 0).
    """

    def __init__(self, mean_steps: float | None, n: mpz):
        self.n = n
        mid = n // 2
        self.mid = mid

        # First odd >= mid
        self.start = mid if mid % 2 == 1 else mid + 1
        if self.start >= n:
            self.done = True
        else:
            self.done = False

        if mean_steps is None or mean_steps <= 0:
            self.mean_k = 0
        else:
            mk = int(math.ceil(mean_steps))
            self.mean_k = mk if mk >= 0 else 0

        self.low = 0
        self.high = self.mean_k
        self.phase = "zigzag"  # "zigzag" or "tail"
        self.tail_k = self.mean_k + 1
        self.step = 0

    def next_candidate(self) -> mpz | None:
        """
        Return the next subtractor prime candidate p (odd integer),
        or None if done.
        """
        if self.done:
            return None

        n = self.n

        while True:
            if self.phase == "zigzag":
                if self.low > self.high:
                    # Finished interleaving [0 .. mean_k], move to tail
                    self.phase = "tail"
                    continue

                if self.step % 2 == 0:
                    k = self.low
                    self.low += 1
                else:
                    k = self.high
                    self.high -= 1

                self.step += 1

                if k < 0:
                    continue

                p = self.start + 2 * k
                if p >= n:
                    # Reached/passed n; future steps will only be larger
                    if self.low > self.high:
                        self.phase = "tail"
                    else:
                        # Even if there are k left, they only increase p further
                        self.done = True
                        return None
                    continue

                return p

            else:  # tail phase
                k = self.tail_k
                self.tail_k += 1

                p = self.start + 2 * k
                if p >= n:
                    self.done = True
                    return None

                return p


# ---------- Combined race between the two strategies ----------

def goldbach_race(
    n: int,
    mean_sub: float | None,
    mean_mid_steps: float | None,
) -> tuple[bool, int, int, int, int, int | None]:
    """
    Run a race between Strat1 and Strat2 for even n >= 4.

    Returns:
        (found, total_checks, winner, s1_count, s2_count, mid_steps_result)

        * found: True if a decomposition was found.
        * total_checks: total number of subtractor candidates tested
          across BOTH strategies before success (the "subtractor count").
        * winner: 0 if none; 1 if Strat1 won; 2 if Strat2 won.
        * s1_count: how many candidates Strat1 tested.
        * s2_count: how many candidates Strat2 tested.
        * mid_steps_result: if winner == 2, the distance in "steps" from n/2
          to the winning p (i.e., |p - n/2| / 2); otherwise None.
    """
    n = mpz(n)

    if n < 4 or n % 2 != 0:
        return False, 0, 0, 0, 0, None

    strat1 = Strat1State(mean_sub, n)
    strat2 = Strat2State(mean_mid_steps, n)

    visited: set[int] = set()
    total = 0
    s1_count = 0
    s2_count = 0

    winner = 0
    mid_steps_result: int | None = None

    # Alternate between strategies until one finds a decomposition
    while True:
        progressed = False

        # Step from Strategy 1
        if not strat1.done:
            p = strat1.next_candidate()
            if p is not None:
                progressed = True
                ip = int(p)
                if ip not in visited:
                    visited.add(ip)
                    q = n - p
                    total += 1
                    s1_count += 1
                    if is_prime(q):
                        winner = 1
                        break

        # Step from Strategy 2
        if not strat2.done and winner == 0:
            p2 = strat2.next_candidate()
            if p2 is not None:
                progressed = True
                ip2 = int(p2)
                if ip2 not in visited:
                    visited.add(ip2)
                    q2 = n - p2
                    total += 1
                    s2_count += 1
                    if is_prime(q2):
                        winner = 2
                        # Distance from n/2 in "steps" of 2
                        mid = n // 2
                        dist = abs(p2 - mid)
                        mid_steps_result = int(dist // 2)
                        break

        if winner != 0:
            break

        if not progressed:
            # Both strategies are out of candidates
            break

    if winner == 0:
        return False, total, 0, s1_count, s2_count, None

    return True, total, winner, s1_count, s2_count, mid_steps_result


# ---------- Random even generator for a given digit length ----------

def random_even_with_digits(rng: random.Random, digits: int) -> int:
    """
    Draw a random even integer with exactly `digits` decimal digits,
    ensuring n >= 4.
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
        raise ValueError(f"No usable even numbers in digit range {digits}.")

    count_evens = ((last_even - first_even) // 2) + 1
    k = rng.randrange(count_evens)
    n = first_even + 2 * k

    if n < 4:
        n += 2 * ((4 - n + 1) // 2)
        if n > last_even:
            n = first_even

    return n


# ---------- Dynamic histogram of subtractor counts ----------

def build_histogram(counts: list[int]) -> list[tuple[int, int, int]]:
    """
    Build a dynamic histogram of subtractor counts.

    Returns:
        List of (start, end, frequency) bins.
    """
    if not counts:
        return []

    max_c = max(counts)
    if max_c <= 0:
        return []

    bins = 10  # number of bins
    width = max(1, math.ceil(max_c / bins))

    hist: list[tuple[int, int, int]] = []
    for i in range(bins):
        start = i * width + 1
        end = (i + 1) * width
        if start > max_c:
            break
        freq = sum(1 for c in counts if start <= c <= end)
        hist.append((start, end, freq))

    return hist


# ---------- Main sweep logic ----------

def run_sweep(start_digits: int, end_digits: int, count_per_digit: int, seed: int | None) -> None:
    """
    Run the Goldbach sampler across digit lengths [start_digits, end_digits].
    """
    if start_digits > end_digits:
        start_digits, end_digits = end_digits, start_digits

    rng = random.Random(seed)

    # Global mean subtractor count (for Strat1)
    global_mean_sub: float | None = None
    global_sub_sum = 0
    global_sub_count = 0

    # Global mean mid-distance steps (for Strat2)
    global_mean_mid_steps: float | None = None
    global_mid_steps_sum = 0
    global_mid_steps_count = 0

    # Global strategy win counts
    global_s1_wins = 0
    global_s2_wins = 0

    for D in range(start_digits, end_digits + 1):
        digit_sub_sum = 0
        digit_sub_count = 0
        digit_max_sub = 0
        digit_skip = 0

        digit_counts_all: list[int] = []
        digit_counts_s1wins: list[int] = []
        digit_counts_s2wins: list[int] = []

        digit_s1_wins = 0
        digit_s2_wins = 0

        # Per-digit sums of s1_count and s2_count (over successful n)
        digit_s1_count_sum = 0
        digit_s2_count_sum = 0

        start_time = time.perf_counter()

        for _ in range(count_per_digit):
            try:
                n = random_even_with_digits(rng, D)
            except ValueError:
                # No valid evens for this digit length
                digit_skip = count_per_digit
                break

            found, total_checks, winner, s1_count, s2_count, mid_steps = goldbach_race(
                n, global_mean_sub, global_mean_mid_steps
            )

            if found:
                # Use total_checks (across both strategies) as the subtractor count
                digit_sub_sum += total_checks
                digit_sub_count += 1
                digit_counts_all.append(total_checks)
                if total_checks > digit_max_sub:
                    digit_max_sub = total_checks

                # Per-strategy candidate counts
                digit_s1_count_sum += s1_count
                digit_s2_count_sum += s2_count

                # Per-winner distributions
                if winner == 1:
                    digit_s1_wins += 1
                    global_s1_wins += 1
                    digit_counts_s1wins.append(total_checks)
                elif winner == 2:
                    digit_s2_wins += 1
                    global_s2_wins += 1
                    digit_counts_s2wins.append(total_checks)

                # Update global mean subtractor count (for Strat1)
                global_sub_sum += total_checks
                global_sub_count += 1
                global_mean_sub = global_sub_sum / global_sub_count

                # Update global mean mid-distance for Strat2 when it wins
                if winner == 2 and mid_steps is not None:
                    global_mid_steps_sum += mid_steps
                    global_mid_steps_count += 1
                    global_mean_mid_steps = global_mid_steps_sum / global_mid_steps_count

            else:
                # No decomposition found (should be rare)
                digit_skip += 1

        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000.0
        ms_per_n = elapsed_ms / count_per_digit if count_per_digit > 0 else 0.0

        if digit_sub_count > 0:
            avg_sub = math.ceil(digit_sub_sum / digit_sub_count)
            mean_s1 = digit_s1_count_sum / digit_sub_count
            mean_s2 = digit_s2_count_sum / digit_sub_count
        else:
            avg_sub = 0
            mean_s1 = 0.0
            mean_s2 = 0.0

        # Main summary line (kept in your requested format)
        print(
            f"D: {D} | "
            f"Ms: {elapsed_ms:.3f} | "
            f"Ms per n: {ms_per_n:.6f} | "
            f"Max Sub: {digit_max_sub} | "
            f"Avg. Sub: {avg_sub} | "
            f"Skip: {digit_skip}"
        )

        # Per-digit strategy wins
        print(f"  Strat1 wins: {digit_s1_wins} | Strat2 wins: {digit_s2_wins}")

        # Per-digit mean candidate counts for each strategy
        print(f"  Mean s1 count: {mean_s1:.3f} | Mean s2 count: {mean_s2:.3f}")

        # Histograms
        hist_all = build_histogram(digit_counts_all)
        hist_s1 = build_histogram(digit_counts_s1wins)
        hist_s2 = build_histogram(digit_counts_s2wins)

        if hist_all:
            print("  Histogram (All totals):")
            for start, end, freq in hist_all:
                if freq > 0:
                    print(f"    [{start:3d}-{end:3d}]: {freq}")
        else:
            print("  Histogram (All totals): (no data)")

        print("  Histogram by winner:")

        if hist_s1:
            print("    Strat1 wins:")
            for start, end, freq in hist_s1:
                if freq > 0:
                    print(f"      [{start:3d}-{end:3d}]: {freq}")
        else:
            print("    Strat1 wins: (no data)")

        if hist_s2:
            print("    Strat2 wins:")
            for start, end, freq in hist_s2:
                if freq > 0:
                    print(f"      [{start:3d}-{end:3d}]: {freq}")
        else:
            print("    Strat2 wins: (no data)")

    # Global summary
    print(f"Total Strat1 wins: {global_s1_wins} | Total Strat2 wins: {global_s2_wins}")
    if global_mid_steps_count > 0:
        print(f"Global mean mid-distance steps (Strat2): {global_mean_mid_steps:.3f}")


# ---------- Argument parsing / entry point ----------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Goldbach sampler with racing strategies and detailed histograms."
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
