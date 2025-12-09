import gmpy2
import random
import time
import statistics as stats
import argparse
import math

# ---------- random N generation ----------

def random_even_with_digits(d, rng):
    """
    Return a random even integer N with exactly d decimal digits,
    using the provided random.Random instance rng.
    """
    low = 10 ** (d - 1)
    high = 10 ** d - 1

    # Restrict to even numbers in [low, high]
    low_even = low if low % 2 == 0 else low + 1
    high_even = high if high % 2 == 0 else high - 1

    if low_even > high_even:
        raise ValueError(f"No even numbers for digit length {d} (should not happen)")

    # Number of even values in that range
    count_evens = (high_even - low_even) // 2 + 1
    offset = rng.randrange(count_evens)
    return low_even + 2 * offset


# ---------- zig-zag generators over subtractor indices ----------

def zigzag_avg_only(avg_idx):
    """
    Yield subtractor indices (1-based) in the pattern:

        1, avg, 2, avg-1, 3, avg+1, 4, avg-2, ...

    avoiding duplicates and indices <= 0.
    """
    seen = set()

    def add(x):
        if x > 0 and x not in seen:
            seen.add(x)
            return x
        return None

    # Initial two: 1, avg
    if (v := add(1)) is not None:
        yield v
    if (v := add(avg_idx)) is not None:
        yield v

    small = 2
    delta = 1
    sign = -1  # first wiggle is avg - 1

    while True:
        # Small increasing: 2,3,4,...
        while small in seen:
            small += 1
        if (v := add(small)) is not None:
            yield v
        small += 1

        # Wiggle around avg: avg-1, avg+1, avg-2, avg+2, ...
        tried = 0
        while tried < 1000:  # safety
            cand = avg_idx + sign * delta
            sign *= -1
            if sign == -1:
                delta += 1
            if cand > 0 and cand not in seen:
                if (v := add(cand)) is not None:
                    yield v
                break
            tried += 1


def zigzag_avg_and_med(avg_idx, med_idx):
    """
    Yield subtractor indices (1-based) in the pattern:

        1, avg, med,
        2, avg-1, med+1,
        3, avg+1, med+2,
        ...

    avoiding duplicates and indices <= 0.
    """
    seen = set()

    def add(x):
        if x > 0 and x not in seen:
            seen.add(x)
            return x
        return None

    # Initial triple: 1, avg, med
    if (v := add(1)) is not None:
        yield v
    if (v := add(avg_idx)) is not None:
        yield v
    if (v := add(med_idx)) is not None:
        yield v

    next_small = 2
    delta_avg = 1
    sign = -1  # first wiggle is avg-1
    next_med = med_idx + 1

    while True:
        # small increasing: 2,3,4,5,...
        while next_small in seen:
            next_small += 1
        if (v := add(next_small)) is not None:
            yield v
        next_small += 1

        # wiggle around avg: avg-1, avg+1, avg-2, avg+2, ...
        tried = 0
        while tried < 1000:
            cand = avg_idx + sign * delta_avg
            sign *= -1
            if sign == -1:
                delta_avg += 1
            if cand > 0 and cand not in seen:
                if (v := add(cand)) is not None:
                    yield v
                break
            tried += 1

        # march upwards from med: med+1, med+2, med+3, ...
        while next_med in seen:
            next_med += 1
        if (v := add(next_med)) is not None:
            yield v
        next_med += 1


# ---------- prime handling ----------

class PrimeCache:
    """
    Precompute a fixed list of subtractor primes: 3,5,7,11,...
    Access by 1-based index: index 1 -> 3, index 2 -> 5, etc.
    """
    def __init__(self, max_count=50000):
        self.primes = []
        p = gmpy2.mpz(3)
        for _ in range(max_count):
            self.primes.append(p)
            p = gmpy2.next_prime(p)

    def get_prime(self, index):
        if index < 1 or index > len(self.primes):
            raise IndexError(
                f"Subtractor prime index {index} out of precomputed range "
                f"(1..{len(self.primes)})"
            )
        return self.primes[index - 1]


# ---------- search ----------

def search_zigzag(N, prime_cache, avg_center, med_center):
    """
    Zig-zag search using fixed centers for this digit length:

      - avg_center: seed for the first decomposition region
      - med_center: seed for the second decomposition region (or None)

    Returns:
        first_sub_count, second_sub_count, first_pair, second_pair

    where:
        first_sub_count, second_sub_count are step numbers (1-based),
        first_pair = (p1, q1) or None,
        second_pair = (p2, q2) or None.
    """
    first_sub_count = None
    second_sub_count = None
    first_pair = None
    second_pair = None
    step = 0

    avg_idx = max(1, int(math.ceil(avg_center)))

    if med_center is None:
        index_gen = zigzag_avg_only(avg_idx)
    else:
        med_idx = max(1, int(round(med_center)))
        index_gen = zigzag_avg_and_med(avg_idx, med_idx)

    for idx in index_gen:
        try:
            p = prime_cache.get_prime(idx)
        except IndexError:
            break
        if p >= N:
            break

        step += 1
        q = N - p
        if gmpy2.is_prime(q):
            if first_sub_count is None:
                first_sub_count = step
                first_pair = (int(p), int(q))
            elif second_sub_count is None:
                second_sub_count = step
                second_pair = (int(p), int(q))
                break

    return first_sub_count, second_sub_count, first_pair, second_pair


# ---------- main sweep logic ----------

def sweep_digit_lengths(start_d, stop_d, step_d, count, seed, show_decomps=False):
    """
    Sweep over digit lengths from start_d to stop_d inclusive,
    stepping by step_d.

    For each digit length D:

      - Use avg/median centers:
          * avg_seed = previous digit's Avg. Sub if available, else D
          * med_seed = previous digit's Med. Sub if available, else None
      - For all Ns of that digit length, use the same zig-zag centers.
      - Measure the actual Avg. Sub and Med. Sub from the results.
      - Optionally, collect and print sample decompositions.
    """
    rng = random.Random(seed)
    prime_cache = PrimeCache(max_count=50000)

    prev_avg_first = None
    prev_med_second = None

    MAX_SAMPLES_PER_DIGIT_FOR_PRINT = 5  # how many Ns to show per digit length

    for d in range(start_d, stop_d + 1, step_d):
        first_counts = []    # step counts for first decomp
        second_counts = []   # step counts for second decomp

        # For sanity output
        sample_decomps = []  # list of (N, first_sub, first_pair, second_sub, second_pair)

        # Choose centers for this digit length
        if prev_avg_first is not None:
            avg_seed = prev_avg_first
        else:
            # Reasonable starting guess: around D
            avg_seed = float(d)

        med_seed = prev_med_second  # may be None

        t0 = time.perf_counter()

        for _ in range(count):
            N = random_even_with_digits(d, rng)

            first_sub, second_sub, first_pair, second_pair = search_zigzag(
                N,
                prime_cache,
                avg_center=avg_seed,
                med_center=med_seed,
            )

            if first_sub is None:
                # No decomposition found (unlikely for these ranges), skip.
                continue

            first_counts.append(first_sub)

            if second_sub is not None:
                second_counts.append(second_sub)

            # Collect some examples for sanity check
            if show_decomps and len(sample_decomps) < MAX_SAMPLES_PER_DIGIT_FOR_PRINT:
                sample_decomps.append(
                    (N, first_sub, first_pair, second_sub, second_pair)
                )

        elapsed = time.perf_counter() - t0

        # Aggregate stats for this digit length
        if first_counts:
            avg_first = sum(first_counts) / len(first_counts)
        else:
            avg_first = float("nan")

        if second_counts:
            med_second = stats.median(second_counts)
            med_str = f"{med_second:.3f}"
        else:
            med_second = None
            med_str = "n/a"

        ms_per_n = (elapsed / count) * 1000.0 if count > 0 else float("nan")

        # Summary line (2nd vs Sample removed)
        print(
            f"D: {d} | "
            f"Seconds: {elapsed:.3f} | "
            f"Avg. Sub: {avg_first:.3f} | "
            f"Med. Sub: {med_str} | "
            f"Ms per N: {ms_per_n:.3f}"
        )

        # Optional decompositions display
        if show_decomps and sample_decomps:
            for (N, first_sub, first_pair, second_sub, second_pair) in sample_decomps:
                if first_pair is None:
                    continue
                p1, q1 = first_pair
                if second_pair is not None:
                    p2, q2 = second_pair
                    print(
                        f"  N={N} | "
                        f"1st (step {first_sub}): {p1} + {q1} | "
                        f"2nd (step {second_sub}): {p2} + {q2}"
                    )
                else:
                    # Only first decomp found; rare, but we handle it.
                    print(
                        f"  N={N} | "
                        f"1st (step {first_sub}): {p1} + {q1} | "
                        f"2nd: none"
                    )

        # Seed next digit length with current digit's stats
        if not math.isnan(avg_first):
            prev_avg_first = avg_first
        if med_second is not None:
            prev_med_second = med_second


# ---------- CLI parsing ----------

def parse_sweep(sweep_str):
    """
    Parse a sweep spec of the form 'A:B:S' into three ints.
    """
    parts = sweep_str.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid --sweep format '{sweep_str}', expected A:B:S")
    start_d, stop_d, step_d = map(int, parts)
    return start_d, stop_d, step_d


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Goldbach subtractor statistics with seeded zig-zag search using gmpy2."
    )
    parser.add_argument(
        "--sweep",
        type=str,
        required=True,
        help="Digit-length sweep in the form A:B:S (start:stop:step), e.g. 6:10:2",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1000,
        help="Number of N samples per digit length (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for reproducibility (default: 1)",
    )
    parser.add_argument(
        "--show-decomps",
        action="store_true",
        help="Print sample decompositions per digit length for sanity checking",
    )

    args = parser.parse_args()
    start_d, stop_d, step_d = parse_sweep(args.sweep)

    sweep_digit_lengths(
        start_d=start_d,
        stop_d=stop_d,
        step_d=step_d,
        count=args.count,
        seed=args.seed,
        show_decomps=args.show_decomps,
    )
