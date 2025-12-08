#!/usr/bin/env python3
"""
Goldbach sampler with three racing strategies, optional quiet mode, a --race flag,
hard-case logging, and an --unlucky mode to bias n toward "mod-bad" / unlucky cases.

Normal mode (default):
    - Three strategies (S1, S2, S3) race on each n.
    - Outputs detailed stats per digit, including histograms.
    - --quiet collapses output to one line per digit.

Race mode (--race):
    - Runs the same sweep three times with the same seed:
        1. Strategy 1 only (S1)
        2. Strategy 2 only (S2)
        3. Strategy 3 only (S3)
    - Each run sees the same sequence of n values.
    - Per-digit output:
        D: _ | Seconds: _ | Ms per n: _ | Max Sub: _ | Avg. Sub: _ | Wins: _ | Skip: _

Hard-case logging (--log-hard K):
    - In normal mode, for each digit length, track all successful n and their total
      subtractor checks.
    - At the end of that digit's run, print the top K "hardest" n
      (largest total_checks), along with winner and per-strategy counts.

Unlucky mode (--unlucky):
    - Instead of uniform random evens, choose n that line up with multiple "bad"
      modular characteristics:
      for many of the first small subtractor primes p, q = n - p is divisible
      by a small prime.
    - This biases sampling toward "hard" / unlucky n, but the Goldbach search
      itself remains unchanged.
"""

import argparse
import math
import random
import time

import gmpy2
from gmpy2 import mpz, is_prime, next_prime, isqrt


# ---------- Global small primes for Strategy 1 ----------

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
        Return the next subtractor candidate p (odd integer),
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


# ---------- Strategy 3: around sqrt(n) with mean-distance zig-zag ----------

class Strat3State:
    """
    Strategy 3 state:

    Let root = floor(sqrt(n)). Start from the first odd >= root ("center" r0).
    Define steps k >= 0:
        step k → p = start + 2*k

    With a global mean_sqrt_steps, we define mean_k = ceil(mean_sqrt_steps).
    Then we use the pattern:
        k = 0, mean_k, 1, mean_k-1, 2, mean_k-2, ...
    and then tail: mean_k+1, mean_k+2, ...
    Any candidate p >= n is invalid (q <= 0).
    """

    def __init__(self, mean_steps: float | None, n: mpz):
        self.n = n
        root = isqrt(n)
        self.root = root

        # First odd >= root
        self.start = root if root % 2 == 1 else root + 1
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
        Return the next subtractor candidate p (odd integer),
        or None if done.
        """
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

            else:  # tail
                k = self.tail_k
                self.tail_k += 1

                p = self.start + 2 * k
                if p >= n:
                    self.done = True
                    return None

                return p


# ---------- Three-strategy race (normal mode) with 3:1:1 scheduling ----------

def goldbach_race(
    n: int,
    mean_sub: float | None,
    mean_mid_steps: float | None,
    mean_sqrt_steps: float | None,
) -> tuple[bool, int, int, int, int, int, int | None, int | None]:
    """
    Run a race between Strat1, Strat2, and Strat3 for even n >= 4.

    Scheduling per round:
        - Up to 3 candidates from Strategy 1 (S1),
        - Then 1 candidate from Strategy 2 (S2),
        - Then 1 candidate from Strategy 3 (S3).

    Returns:
        (found,
         total_checks,
         winner,         # 0=none, 1=S1, 2=S2, 3=S3
         s1_count,
         s2_count,
         s3_count,
         mid_steps_result,   # if winner==2, distance in steps from n/2
         sqrt_steps_result)  # if winner==3, distance in steps from sqrt(n)
    """
    n = mpz(n)

    if n < 4 or n % 2 != 0:
        return False, 0, 0, 0, 0, 0, None, None

    s1 = Strat1State(mean_sub, n)
    s2 = Strat2State(mean_mid_steps, n)
    s3 = Strat3State(mean_sqrt_steps, n)

    visited: set[int] = set()
    total = 0
    s1_count = 0
    s2_count = 0
    s3_count = 0

    winner = 0
    mid_steps_result: int | None = None
    sqrt_steps_result: int | None = None

    while True:
        progressed = False

        # --- Strategy 1: up to 3 candidates per round ---
        for _ in range(3):
            if winner != 0 or s1.done:
                break
            p1 = s1.next_candidate()
            if p1 is None:
                continue
            progressed = True
            ip1 = int(p1)
            if ip1 in visited:
                continue
            visited.add(ip1)
            q1 = n - p1
            total += 1
            s1_count += 1
            if is_prime(q1):
                winner = 1
                break

        if winner != 0:
            break

        # --- Strategy 2: one candidate per round ---
        if winner == 0 and not s2.done:
            p2 = s2.next_candidate()
            if p2 is not None:
                progressed = True
                ip2 = int(p2)
                if ip2 not in visited:
                    visited.add(ip2)
                    if is_prime(p2):  # p2 must be prime
                        q2 = n - p2
                        total += 1
                        s2_count += 1
                        if is_prime(q2):
                            winner = 2
                            mid = n // 2
                            dist = abs(p2 - mid)
                            mid_steps_result = int(dist // 2)
                            break

        if winner != 0:
            break

        # --- Strategy 3: one candidate per round ---
        if winner == 0 and not s3.done:
            p3 = s3.next_candidate()
            if p3 is not None:
                progressed = True
                ip3 = int(p3)
                if ip3 not in visited:
                    visited.add(ip3)
                    if is_prime(p3):  # p3 must be prime
                        q3 = n - p3
                        total += 1
                        s3_count += 1
                        if is_prime(q3):
                            winner = 3
                            root = s3.root
                            dist3 = abs(p3 - root)
                            sqrt_steps_result = int(dist3 // 2)
                            break

        if winner != 0:
            break

        if not progressed:
            # All strategies are out of candidates
            break

    if winner == 0:
        return False, total, 0, s1_count, s2_count, s3_count, None, None

    return True, total, winner, s1_count, s2_count, s3_count, mid_steps_result, sqrt_steps_result


# ---------- Single-strategy solvers (used for --race mode) ----------

def goldbach_single_s1(n: int, mean_sub: float | None) -> tuple[bool, int]:
    """
    Strategy 1 only on n. Returns (found, total_checks).
    """
    n = mpz(n)
    if n < 4 or n % 2 != 0:
        return False, 0

    s1 = Strat1State(mean_sub, n)
    total = 0

    while True:
        p = s1.next_candidate()
        if p is None:
            break
        q = n - p
        total += 1
        if is_prime(q):
            return True, total

    return False, total


def goldbach_single_s2(n: int, mean_mid_steps: float | None) -> tuple[bool, int, int | None]:
    """
    Strategy 2 only on n. Returns (found, total_checks, mid_steps).
    mid_steps is distance in steps from n/2 when found; None if not found.
    """
    n = mpz(n)
    if n < 4 or n % 2 != 0:
        return False, 0, None

    s2 = Strat2State(mean_mid_steps, n)
    total = 0
    mid_steps_result: int | None = None

    while True:
        p = s2.next_candidate()
        if p is None:
            break
        if not is_prime(p):
            continue
        q = n - p
        total += 1
        if is_prime(q):
            mid = n // 2
            dist = abs(p - mid)
            mid_steps_result = int(dist // 2)
            return True, total, mid_steps_result

    return False, total, mid_steps_result


def goldbach_single_s3(n: int, mean_sqrt_steps: float | None) -> tuple[bool, int, int | None]:
    """
    Strategy 3 only on n. Returns (found, total_checks, sqrt_steps).
    sqrt_steps is distance in steps from sqrt(n) when found; None if not found.
    """
    n = mpz(n)
    if n < 4 or n % 2 != 0:
        return False, 0, None

    s3 = Strat3State(mean_sqrt_steps, n)
    total = 0
    sqrt_steps_result: int | None = None

    while True:
        p = s3.next_candidate()
        if p is None:
            break
        if not is_prime(p):
            continue
        q = n - p
        total += 1
        if is_prime(q):
            root = s3.root
            dist = abs(p - root)
            sqrt_steps_result = int(dist // 2)
            return True, total, sqrt_steps_result

    return False, total, sqrt_steps_result


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


# ---------- "Unlucky" even generator based on bad mod characteristics ----------

# Small primes used to measure how often q = n - p is divisible by something small
SIEVE_PRIMES = [3, 5, 7, 11, 13, 17, 19]

# How many early small subtractor primes p to look at when scoring a candidate n
UNLUCKY_MAX_P_INDEX = 16  # first 16 subtractor primes (3,5,7,...)

# Threshold for "badness"
UNLUCKY_SCORE_THRESHOLD = 6

# Max attempts to find a high-score n before giving up
UNLUCKY_MAX_ATTEMPTS = 32


def unlucky_score(n: int) -> int:
    """
    Compute a "badness" score for n based on early small subtractor primes:

    For the first UNLUCKY_MAX_P_INDEX primes p (3,5,7,...),
    we look at q = n - p and see if q is divisible by some small prime in
    SIEVE_PRIMES. If yes (and q > that prime), we count one "blocked" p.

    The score is how many of these early subtractors are "blocked".
    """
    n_mpz = mpz(n)
    score = 0

    for i in range(UNLUCKY_MAX_P_INDEX):
        p = ensure_prime_index(i)
        if p >= n_mpz:
            break
        q = n_mpz - p
        # Check divisibility by small primes
        for r in SIEVE_PRIMES:
            r_mpz = mpz(r)
            if r_mpz >= q:
                break
            if q % r_mpz == 0:
                score += 1
                break

    return score


def unlucky_even_with_digits(rng: random.Random, digits: int) -> int:
    """
    Draw an "unlucky" even integer with exactly `digits` decimal digits.

    We repeatedly sample random evens in the digit range, compute an
    unlucky_score(n), and try to pick n with a high score (many early subtractor
    primes p for which q = n - p is divisible by some small prime).

    After UNLUCKY_MAX_ATTEMPTS attempts, we return the best (highest-score) n
    we have seen. If we ever find one with score >= UNLUCKY_SCORE_THRESHOLD,
    we return it immediately.
    """
    if digits <= 0:
        raise ValueError("Digit length must be positive.")

    best_n = None
    best_score = -1

    for _ in range(UNLUCKY_MAX_ATTEMPTS):
        n = random_even_with_digits(rng, digits)
        s = unlucky_score(n)

        if s > best_score:
            best_score = s
            best_n = n

        if s >= UNLUCKY_SCORE_THRESHOLD:
            return n

    # If we didn't hit the threshold, return the best scoring n so far (or a fresh random if somehow None)
    if best_n is not None:
        return best_n
    return random_even_with_digits(rng, digits)


# ---------- Dynamic histogram of subtractor counts (for verbose mode) ----------

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


# ---------- Main sweep logic (three-way race, normal mode) ----------

def run_sweep(
    start_digits: int,
    end_digits: int,
    count_per_digit: int,
    seed: int | None,
    quiet: bool,
    log_hard: int,
    unlucky_mode: bool,
) -> None:
    """
    Run the Goldbach sampler across digit lengths [start_digits, end_digits]
    with all three strategies racing.

    If log_hard > 0, for each digit length we track all successful n
    and at the end print the top log_hard hardest cases.

    If unlucky_mode is True, we bias n selection toward "unlucky" cases using
    unlucky_even_with_digits. Otherwise we use uniform random_even_with_digits.
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

    # Global mean sqrt-distance steps (for Strat3)
    global_mean_sqrt_steps: float | None = None
    global_sqrt_steps_sum = 0
    global_sqrt_steps_count = 0

    # Global strategy win counts
    global_s1_wins = 0
    global_s2_wins = 0
    global_s3_wins = 0

    for D in range(start_digits, end_digits + 1):
        digit_sub_sum = 0
        digit_sub_count = 0
        digit_max_sub = 0
        digit_skip = 0

        digit_counts_all: list[int] = []
        digit_counts_s1wins: list[int] = []
        digit_counts_s2wins: list[int] = []
        digit_counts_s3wins: list[int] = []

        digit_s1_wins = 0
        digit_s2_wins = 0
        digit_s3_wins = 0

        # Per-digit sums of s1_count, s2_count, s3_count (over successful n)
        digit_s1_count_sum = 0
        digit_s2_count_sum = 0
        digit_s3_count_sum = 0

        # Hard-case storage: (total_checks, n, winner, s1_count, s2_count, s3_count)
        hard_cases: list[tuple[int, int, int, int, int, int]] = []

        start_time = time.perf_counter()

        for _ in range(count_per_digit):
            try:
                if unlucky_mode:
                    n = unlucky_even_with_digits(rng, D)
                else:
                    n = random_even_with_digits(rng, D)
            except ValueError:
                # No valid evens for this digit length
                digit_skip = count_per_digit
                break

            found, total_checks, winner, s1_count, s2_count, s3_count, mid_steps, sqrt_steps = goldbach_race(
                n, global_mean_sub, global_mean_mid_steps, global_mean_sqrt_steps
            )

            if found:
                digit_sub_sum += total_checks
                digit_sub_count += 1
                digit_counts_all.append(total_checks)
                if total_checks > digit_max_sub:
                    digit_max_sub = total_checks

                # Per-strategy candidate counts
                digit_s1_count_sum += s1_count
                digit_s2_count_sum += s2_count
                digit_s3_count_sum += s3_count

                # Per-winner distributions
                if winner == 1:
                    digit_s1_wins += 1
                    global_s1_wins += 1
                    digit_counts_s1wins.append(total_checks)
                elif winner == 2:
                    digit_s2_wins += 1
                    global_s2_wins += 1
                    digit_counts_s2wins.append(total_checks)
                elif winner == 3:
                    digit_s3_wins += 1
                    global_s3_wins += 1
                    digit_counts_s3wins.append(total_checks)

                # Update global mean subtractor count (for Strat1)
                global_sub_sum += total_checks
                global_sub_count += 1
                global_mean_sub = global_sub_sum / global_sub_count

                # Update global means for distance steps
                if winner == 2 and mid_steps is not None:
                    global_mid_steps_sum += mid_steps
                    global_mid_steps_count += 1
                    global_mean_mid_steps = global_mid_steps_sum / global_mid_steps_count

                if winner == 3 and sqrt_steps is not None:
                    global_sqrt_steps_sum += sqrt_steps
                    global_sqrt_steps_count += 1
                    global_mean_sqrt_steps = global_sqrt_steps_sum / global_sqrt_steps_count

                # Record hard-case info
                if log_hard > 0:
                    hard_cases.append(
                        (
                            total_checks,
                            int(n),
                            winner,
                            s1_count,
                            s2_count,
                            s3_count,
                        )
                    )

            else:
                digit_skip += 1

        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000.0
        seconds = elapsed_ms / 1000.0
        ms_per_n = elapsed_ms / count_per_digit if count_per_digit > 0 else 0.0

        if digit_sub_count > 0:
            avg_sub = math.ceil(digit_sub_sum / digit_sub_count)
            mean_s1 = digit_s1_count_sum / digit_sub_count
            mean_s2 = digit_s2_count_sum / digit_sub_count
            mean_s3 = digit_s3_count_sum / digit_sub_count
        else:
            avg_sub = 0
            mean_s1 = mean_s2 = mean_s3 = 0.0

        if quiet:
            # Quiet mode: minimal per-digit line only (3-strategy race)
            print(
                f"D: {D} | Seconds: {seconds:.3f} | Ms per n: {ms_per_n:.6f} | "
                f"S1/S2/S3 Wins: {digit_s1_wins} vs {digit_s2_wins} vs {digit_s3_wins} | "
                f"Skip: {digit_skip}"
            )
        else:
            # Verbose mode: full output
            print(
                f"D: {D} | "
                f"Ms: {elapsed_ms:.3f} | "
                f"Ms per n: {ms_per_n:.6f} | "
                f"Max Sub: {digit_max_sub} | "
                f"Avg. Sub: {avg_sub} | "
                f"Skip: {digit_skip}"
            )

            print(
                f"  Strat wins: S1={digit_s1_wins} | S2={digit_s2_wins} | S3={digit_s3_wins}"
            )
            print(
                f"  Mean counts: s1={mean_s1:.3f} | s2={mean_s2:.3f} | s3={mean_s3:.3f}"
            )

            # Histograms
            hist_all = build_histogram(digit_counts_all)
            hist_s1 = build_histogram(digit_counts_s1wins)
            hist_s2 = build_histogram(digit_counts_s2wins)
            hist_s3 = build_histogram(digit_counts_s3wins)

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

            if hist_s3:
                print("    Strat3 wins:")
                for start, end, freq in hist_s3:
                    if freq > 0:
                        print(f"      [{start:3d}-{end:3d}]: {freq}")
            else:
                print("    Strat3 wins: (no data)")

        # Hard-case reporting (both quiet and verbose modes)
        if log_hard > 0 and hard_cases:
            # Sort by total_checks descending
            hard_cases.sort(key=lambda t: t[0], reverse=True)
            top_k = hard_cases[:log_hard]
            print(f"  Hardest {len(top_k)} cases:")
            for idx, (checks, n_val, winner, c1, c2, c3) in enumerate(top_k, start=1):
                if winner == 1:
                    wstr = "S1"
                elif winner == 2:
                    wstr = "S2"
                else:
                    wstr = "S3"
                print(
                    f"    #{idx} checks={checks}, winner={wstr}, "
                    f"s1={c1}, s2={c2}, s3={c3}, n={n_val}"
                )

    # Global summary in verbose mode
    if not quiet:
        print(
            f"Total wins: S1={global_s1_wins} | S2={global_s2_wins} | S3={global_s3_wins}"
        )
        if global_mid_steps_count > 0:
            print(f"Global mean mid-distance steps (S2): {global_mean_mid_steps:.3f}")
        if global_sqrt_steps_count > 0:
            print(f"Global mean sqrt-distance steps (S3): {global_mean_sqrt_steps:.3f}")


# ---------- Single-strategy sweep (for --race mode) ----------

def run_sweep_single_strategy(
    start_digits: int,
    end_digits: int,
    count_per_digit: int,
    seed: int | None,
    strategy: int,
) -> None:
    """
    Run the sweep using only one strategy (1, 2, or 3) on each n.
    Used for --race mode to compare strategies separately.

    Output per digit:
        D: _ | Seconds: _ | Ms per n: _ | Max Sub: _ | Avg. Sub: _ | Wins: _ | Skip: _
    """
    if start_digits > end_digits:
        start_digits, end_digits = end_digits, start_digits

    rng = random.Random(seed)

    # Per-strategy global adaptives
    mean_sub: float | None = None       # for S1
    sub_sum = 0
    sub_count = 0

    mean_mid_steps: float | None = None  # for S2
    mid_sum = 0
    mid_count = 0

    mean_sqrt_steps: float | None = None  # for S3
    sqrt_sum = 0
    sqrt_count = 0

    for D in range(start_digits, end_digits + 1):
        digit_sub_sum = 0
        digit_sub_count = 0
        digit_max_sub = 0
        digit_skip = 0
        digit_wins = 0

        start_time = time.perf_counter()

        for _ in range(count_per_digit):
            try:
                n = random_even_with_digits(rng, D)
            except ValueError:
                digit_skip = count_per_digit
                break

            if strategy == 1:
                found, total = goldbach_single_s1(n, mean_sub)
                mid_steps = None
                sqrt_steps = None
            elif strategy == 2:
                found, total, mid_steps = goldbach_single_s2(n, mean_mid_steps)
                sqrt_steps = None
            else:  # strategy == 3
                found, total, sqrt_steps = goldbach_single_s3(n, mean_sqrt_steps)
                mid_steps = None

            if found:
                digit_wins += 1
                digit_sub_sum += total
                digit_sub_count += 1
                if total > digit_max_sub:
                    digit_max_sub = total

                # Update global means for this strategy
                if strategy == 1:
                    sub_sum += total
                    sub_count += 1
                    mean_sub = sub_sum / sub_count
                elif strategy == 2 and mid_steps is not None:
                    mid_sum += mid_steps
                    mid_count += 1
                    mean_mid_steps = mid_sum / mid_count
                elif strategy == 3 and sqrt_steps is not None:
                    sqrt_sum += sqrt_steps
                    sqrt_count += 1
                    mean_sqrt_steps = sqrt_sum / sqrt_count
            else:
                digit_skip += 1

        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000.0
        seconds = elapsed_ms / 1000.0
        ms_per_n = elapsed_ms / count_per_digit if count_per_digit > 0 else 0.0

        if digit_sub_count > 0:
            avg_sub = math.ceil(digit_sub_sum / digit_sub_count)
        else:
            avg_sub = 0

        print(
            f"D: {D} | Seconds: {seconds:.3f} | Ms per n: {ms_per_n:.6f} | "
            f"Max Sub: {digit_max_sub} | Avg. Sub: {avg_sub} | Wins: {digit_wins} | Skip: {digit_skip}"
        )


# ---------- Argument parsing / entry point ----------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Goldbach sampler with three racing strategies, race mode, hard-case logging, and unlucky sampling."
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
        help="Number of even n per digit length (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Quiet mode (for normal 3-strategy race): "
             "only minimal per-digit stats, hide histograms and global summary.",
    )
    parser.add_argument(
        "--race",
        action="store_true",
        help="Race mode: run the sweep three times with the same seed, "
             "once per strategy (S1-only, S2-only, S3-only).",
    )
    parser.add_argument(
        "--log-hard",
        type=int,
        default=0,
        help="If > 0, for each digit length print the top K hardest n "
             "(largest total subtractor checks).",
    )
    parser.add_argument(
        "--unlucky",
        action="store_true",
        help="Bias n sampling toward 'unlucky' evens whose early subtractor primes "
             "tend to give q divisible by small primes.",
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

    if args.race:
        # Run three separate sweeps, one per strategy, all with the same seed.
        print("=== RACE: STRATEGY 1 ONLY (S1) ===")
        run_sweep_single_strategy(
            start_digits=start_digits,
            end_digits=end_digits,
            count_per_digit=args.count,
            seed=args.seed,
            strategy=1,
        )

        print("\n=== RACE: STRATEGY 2 ONLY (S2) ===")
        run_sweep_single_strategy(
            start_digits=start_digits,
            end_digits=end_digits,
            count_per_digit=args.count,
            seed=args.seed,
            strategy=2,
        )

        print("\n=== RACE: STRATEGY 3 ONLY (S3) ===")
        run_sweep_single_strategy(
            start_digits=start_digits,
            end_digits=end_digits,
            count_per_digit=args.count,
            seed=args.seed,
            strategy=3,
        )
    else:
        # Normal mode: three strategies race together.
        run_sweep(
            start_digits=start_digits,
            end_digits=end_digits,
            count_per_digit=args.count,
            seed=args.seed,
            quiet=args.quiet,
            log_hard=args.log_hard,
            unlucky_mode=args.unlucky,
        )


if __name__ == "__main__":
    main()
