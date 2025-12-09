#!/usr/bin/env python3
"""
Goldbach sampler with 3 strategies and 2-core parallelism.

Core 1 (worker_core1): Strategy 1 + Strategy 2
  - S1: small subtractor primes (precomputed table of 100000 primes).
        For each n:
          * search subtractor primes p
          * count q-tests (is_prime(n - p))
          * record step of 1st and 2nd decompositions
          * stop after a 3rd decomposition (just record that it happened)
  - S2: around n/2 (only runs, conceptually, for stats; S1 is independent).
        For each n:
          * search odd p >= n/2
          * require p and q = n - p both prime
          * record step of 1st decomposition if any

Core 2 (worker_core2): Strategy 3
  - S3: around sqrt(n).
        For each n:
          * search odd p near floor(sqrt(n))
          * require p and q = n - p both prime
          * record step of 1st decomposition if any

Heuristic adaptation:
  - For S1, S2, S3 we track per-digit mean *total steps* (q-tests) per strategy.
  - For digit D, each strategy’s initial mean is seeded from a moving window
    of the last 20 digits’ per-digit means for that strategy.
  - Within digit D, a strategy’s mean is updated only on ns where it finds a
    first decomposition.

We only treat an n as “skipped” if *S1* fails to find a first decomposition
(i.e., no Goldbach decomposition found by subtractor primes within our prime table).
S2/S3 failures are allowed and simply not used in their averages.

CLI:
  --sweep A:B   digit range inclusive (e.g. 20:50)
  --count N     number of random even n per digit (default 1000)
  --seed S      RNG seed (default None)
"""

import argparse
import math
import random
import time
from concurrent.futures import ProcessPoolExecutor

import gmpy2
from gmpy2 import mpz, is_prime, next_prime, isqrt

# ----- Global configuration -----

PRIME_COUNT = 100000      # number of subtractor primes for S1
WINDOW_SIZE = 20          # moving-window length for per-strategy means
SMALL_PRIMES = []         # filled by init_primes()


# ----- Prime table for S1 -----

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


def get_prime(idx: int):
    """Return idx-th subtractor prime, or None if out of range."""
    if 0 <= idx < PRIME_COUNT:
        return SMALL_PRIMES[idx]
    return None


# ----- Strategy 1: small subtractor primes, mean-based zig-zag -----

class Strat1State:
    """
    Strategy 1: subtractor primes p[i] = 3,5,7,11,... using a mean-based zig-zag.

    For a given n and mean_sub, define:
      mean_index = ceil(mean_sub) - 1 (clamped into [0, PRIME_COUNT-1])

    Index pattern:
      0, mean_index, 1, mean_index-1, 2, mean_index-2, ...
    then:
      mean_index+1, mean_index+2, ...

    Stop when p >= n or we run out of prime table.
    """

    def __init__(self, mean_sub, n: mpz):
        if mean_sub is None or mean_sub <= 1:
            mi = 0
        else:
            mi = int(math.ceil(mean_sub)) - 1
            mi = max(mi, 0)
        if mi >= PRIME_COUNT:
            mi = PRIME_COUNT - 1

        self.mean_index = mi
        self.n = n

        self.low = 0
        self.high = self.mean_index
        self.phase = "zigzag"
        self.tail_idx = self.mean_index + 1
        self.step = 0
        self.done = False

    def next_candidate(self):
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
            else:
                idx = self.tail_idx
                self.tail_idx += 1
                p = get_prime(idx)
                if p is None or p >= n:
                    self.done = True
                    return None
                return p


# ----- Strategy 2: around n/2, mean-distance zig-zag -----

class Strat2State:
    """
    Strategy 2: around n/2 in step space.

    mid = n//2, start = first odd >= mid.
    Steps k >= 0, p = start + 2*k.

    mean_k = ceil(mean_mid_steps)
    Pattern over k:
      0, mean_k, 1, mean_k-1, 2, mean_k-2, ...
    then:
      mean_k+1, mean_k+2, ...
    """

    def __init__(self, mean_steps, n: mpz):
        self.n = n
        mid = n // 2
        self.mid = mid
        self.start = mid if mid % 2 == 1 else mid + 1
        if self.start >= n:
            self.done = True
            return
        self.done = False

        if mean_steps is None or mean_steps <= 0:
            mk = 0
        else:
            mk = int(math.ceil(mean_steps))
            mk = max(mk, 0)
        self.mean_k = mk

        self.low = 0
        self.high = self.mean_k
        self.phase = "zigzag"
        self.tail_k = self.mean_k + 1
        self.step = 0

    def next_candidate(self):
        if self.done:
            return None
        n = self.n
        while True:
            if self.phase == "zigzag":
                if self.low > self.high:
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
                    if self.low > self.high:
                        self.phase = "tail"
                    else:
                        self.done = True
                        return None
                    continue
                return p
            else:
                k = self.tail_k
                self.tail_k += 1
                p = self.start + 2 * k
                if p >= n:
                    self.done = True
                    return None
                return p


# ----- Strategy 3: around sqrt(n), mean-distance zig-zag -----

class Strat3State:
    """
    Strategy 3: around sqrt(n) in step space.

    root = floor(sqrt(n)), start = first odd >= root.
    Steps k >= 0, p = start + 2*k.

    mean_k = ceil(mean_sqrt_steps)
    Pattern over k:
      0, mean_k, 1, mean_k-1, 2, mean_k-2, ...
    then:
      mean_k+1, mean_k+2, ...
    """

    def __init__(self, mean_steps, n: mpz):
        self.n = n
        root = isqrt(n)
        self.root = root
        self.start = root if root % 2 == 1 else root + 1
        if self.start >= n:
            self.done = True
            return
        self.done = False

        if mean_steps is None or mean_steps <= 0:
            mk = 0
        else:
            mk = int(math.ceil(mean_steps))
            mk = max(mk, 0)
        self.mean_k = mk

        self.low = 0
        self.high = self.mean_k
        self.phase = "zigzag"
        self.tail_k = self.mean_k + 1
        self.step = 0

    def next_candidate(self):
        if self.done:
            return None
        n = self.n
        while True:
            if self.phase == "zigzag":
                if self.low > self.high:
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
                    if self.low > self.high:
                        self.phase = "tail"
                    else:
                        self.done = True
                        return None
                    continue
                return p
            else:
                k = self.tail_k
                self.tail_k += 1
                p = self.start + 2 * k
                if p >= n:
                    self.done = True
                    return None
                return p


# ----- Core1: S1 up to 3 decomps, S2 one decomp -----

def s1_find_up_to_three(n: int, mean_sub):
    """
    Strategy 1 on n, using mean_sub.

    Returns:
      (step_first, step_second, step_third, total_steps)

    step_first  = step index where first decomposition found (or None)
    step_second = step index for second decomposition (or None)
    step_third  = step index for third decomposition (or None)
    total_steps = total q-tests performed
    """
    init_primes()
    state = Strat1State(mean_sub, mpz(n))
    total_steps = 0
    first = None
    second = None
    third = None
    n_mpz = mpz(n)

    while True:
        p = state.next_candidate()
        if p is None:
            break
        q = n_mpz - p
        total_steps += 1
        if is_prime(q):
            if first is None:
                first = total_steps
            elif second is None:
                second = total_steps
            elif third is None:
                third = total_steps
                break

    return first, second, third, total_steps


def s2_find_one(n: int, mean_mid_steps):
    """
    Strategy 2 on n, using mean_mid_steps.

    Returns:
      (step_first, total_steps)

    step_first  = step index where first decomposition found (or None)
    total_steps = total q-tests performed
    """
    state = Strat2State(mean_mid_steps, mpz(n))
    total_steps = 0
    first = None
    n_mpz = mpz(n)

    while True:
        p = state.next_candidate()
        if p is None:
            break
        if not is_prime(p):
            continue
        q = n_mpz - p
        total_steps += 1
        if is_prime(q):
            first = total_steps
            break

    return first, total_steps


def worker_core1(args):
    """
    Worker for core 1 (process 1): S1 and S2.

    Input: (n, mean_sub, mean_mid_steps)
    Output:
      (s1_first, s1_second, s1_third, s1_total,
       s2_first, s2_total)
    """
    n, mean_sub, mean_mid_steps = args
    init_primes()

    s1_first, s1_second, s1_third, s1_total = s1_find_up_to_three(n, mean_sub)
    s2_first, s2_total = s2_find_one(n, mean_mid_steps)

    return (s1_first, s1_second, s1_third, s1_total,
            s2_first, s2_total)


# ----- Core2: S3 one decomp -----

def s3_find_one(n: int, mean_sqrt_steps):
    """
    Strategy 3 on n, using mean_sqrt_steps.

    Returns:
      (step_first, total_steps)
    """
    state = Strat3State(mean_sqrt_steps, mpz(n))
    total_steps = 0
    first = None
    n_mpz = mpz(n)

    while True:
        p = state.next_candidate()
        if p is None:
            break
        if not is_prime(p):
            continue
        q = n_mpz - p
        total_steps += 1
        if is_prime(q):
            first = total_steps
            break

    return first, total_steps


def worker_core2(args):
    """
    Worker for core 2 (process 2): S3 only.

    Input: (n, mean_sqrt_steps)
    Output:
      (s3_first, s3_total)
    """
    n, mean_sqrt_steps = args
    init_primes()  # harmless if already inited
    s3_first, s3_total = s3_find_one(n, mean_sqrt_steps)
    return (s3_first, s3_total)


# ----- Random even generator -----

def random_even_with_digits(rng: random.Random, digits: int) -> int:
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


# ----- Main sweep with 2-core parallelism -----

def run_sweep(start_digits: int, end_digits: int, count_per_digit: int, seed: int | None) -> None:
    if start_digits > end_digits:
        start_digits, end_digits = end_digits, start_digits

    rng = random.Random(seed)

    # Moving windows of per-digit mean total steps for each strategy
    last_means_s1: list[float] = []
    last_means_s2: list[float] = []
    last_means_s3: list[float] = []

    with ProcessPoolExecutor(max_workers=2) as pool:
        for D in range(start_digits, end_digits + 1):
            # Seed means from moving windows
            mean_sub = (sum(last_means_s1) / len(last_means_s1)) if last_means_s1 else None
            mean_mid = (sum(last_means_s2) / len(last_means_s2)) if last_means_s2 else None
            mean_sqrt = (sum(last_means_s3) / len(last_means_s3)) if last_means_s3 else None

            # Per-digit accumulators for adaptation (total steps per n)
            s1_total_sum = 0
            s1_total_count = 0

            s2_total_sum = 0
            s2_total_count = 0

            s3_total_sum = 0
            s3_total_count = 0

            # Per-digit stats for step indices
            s1_first_sum = 0
            s1_first_count = 0
            s1_second_sum = 0
            s1_second_count = 0
            s1_third_hits = 0  # count of n where S1 found a 3rd decomposition

            s2_first_sum = 0
            s2_first_count = 0

            s3_first_sum = 0
            s3_first_count = 0

            # n where S1 found no decompositions at all (Goldbach failure in this sampler)
            skip = 0

            start_time = time.perf_counter()

            for _ in range(count_per_digit):
                try:
                    n = random_even_with_digits(rng, D)
                except ValueError:
                    skip = count_per_digit
                    break

                args1 = (n, mean_sub, mean_mid)
                args2 = (n, mean_sqrt)

                fut1 = pool.submit(worker_core1, args1)
                fut2 = pool.submit(worker_core2, args2)

                (s1_first, s1_second, s1_third, s1_total,
                 s2_first, s2_total) = fut1.result()
                s3_first, s3_total = fut2.result()

                # If S1 never found a decomposition, count this n as skipped
                if s1_first is None:
                    skip += 1
                else:
                    # S1 adaptation & stats
                    s1_total_sum += s1_total
                    s1_total_count += 1
                    mean_sub = s1_total_sum / s1_total_count

                    s1_first_sum += s1_first
                    s1_first_count += 1
                    if s1_second is not None:
                        s1_second_sum += s1_second
                        s1_second_count += 1
                    if s1_third is not None:
                        s1_third_hits += 1

                # S2 adaptation & stats (only if S2 found a decomp)
                if s2_first is not None:
                    s2_total_sum += s2_total
                    s2_total_count += 1
                    mean_mid = s2_total_sum / s2_total_count

                    s2_first_sum += s2_first
                    s2_first_count += 1

                # S3 adaptation & stats (only if S3 found a decomp)
                if s3_first is not None:
                    s3_total_sum += s3_total
                    s3_total_count += 1
                    mean_sqrt = s3_total_sum / s3_total_count

                    s3_first_sum += s3_first
                    s3_first_count += 1

            end_time = time.perf_counter()
            elapsed_ms = (end_time - start_time) * 1000.0
            ms_per_n = elapsed_ms / count_per_digit if count_per_digit > 0 else 0.0

            # Update moving windows with per-digit mean total steps
            if s1_total_count > 0:
                digit_mean_s1 = s1_total_sum / s1_total_count
                last_means_s1.append(digit_mean_s1)
                if len(last_means_s1) > WINDOW_SIZE:
                    last_means_s1.pop(0)

            if s2_total_count > 0:
                digit_mean_s2 = s2_total_sum / s2_total_count
                last_means_s2.append(digit_mean_s2)
                if len(last_means_s2) > WINDOW_SIZE:
                    last_means_s2.pop(0)

            if s3_total_count > 0:
                digit_mean_s3 = s3_total_sum / s3_total_count
                last_means_s3.append(digit_mean_s3)
                if len(last_means_s3) > WINDOW_SIZE:
                    last_means_s3.pop(0)

            def avg_or_zero(s, c):
                return (s / c) if c > 0 else 0.0

            s1_first_mean = avg_or_zero(s1_first_sum, s1_first_count)
            s1_second_mean = avg_or_zero(s1_second_sum, s1_second_count)
            s2_first_mean = avg_or_zero(s2_first_sum, s2_first_count)
            s3_first_mean = avg_or_zero(s3_first_sum, s3_first_count)

            print(f"D: {D} | Ms: {elapsed_ms:.3f} | Ms per n: {ms_per_n:.6f} |")
            print(
                f"  S1 steps (1st/2nd): {s1_first_mean:.3f} ({s1_first_count}) / "
                f"{s1_second_mean:.3f} ({s1_second_count})"
            )
            print(f"  S1 third hits: {s1_third_hits}")
            print(
                f"  S2 steps (1st): {s2_first_mean:.3f} ({s2_first_count})"
            )
            print(
                f"  S3 steps (1st): {s3_first_mean:.3f} ({s3_first_count})"
            )
            print(
                f"  Skip (S1 found no decomp): {skip}"
            )


# ----- CLI -----

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Goldbach sampler with 3 strategies and 2-core parallelism."
    )
    p.add_argument(
        "--sweep",
        type=str,
        required=True,
        help="Digit range A:B (e.g. 20:50).",
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
        raise SystemExit(
            f"Invalid --sweep format. Expected A:B, got '{args.sweep}'."
        ) from e

    # Precompute primes in parent so children can inherit
    init_primes()
    run_sweep(start_digits, end_digits, args.count, args.seed)


if __name__ == "__main__":
    main()
