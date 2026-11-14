"""Bloom filter and Count-Min Sketch benchmark over MovieLens 1M.

Only standard-library modules are used. The ratings file is streamed line by line.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from array import array
from collections import Counter
from pathlib import Path
from typing import Iterator


class BloomFilter:
    def __init__(self, bit_count: int, hash_count: int) -> None:
        if bit_count <= 0 or hash_count <= 0:
            raise ValueError("bit_count and hash_count must be positive")
        self.bit_count = bit_count
        self.hash_count = hash_count
        self.bits = bytearray((bit_count + 7) // 8)

    def _indices(self, key: str) -> Iterator[int]:
        digest = hashlib.blake2b(
            key.encode("utf-8"), digest_size=16, person=b"BloomBenchV1"
        ).digest()
        first = int.from_bytes(digest[:8], "little")
        step = int.from_bytes(digest[8:], "little") | 1
        for i in range(self.hash_count):
            yield (first + i * step) % self.bit_count

    def add(self, key: str) -> None:
        for index in self._indices(key):
            self.bits[index >> 3] |= 1 << (index & 7)

    def might_contain(self, key: str) -> bool:
        return all(
            self.bits[index >> 3] & (1 << (index & 7))
            for index in self._indices(key)
        )

    @property
    def memory_bytes(self) -> int:
        return sys.getsizeof(self) + sys.getsizeof(self.__dict__) + sys.getsizeof(self.bits)

    @property
    def payload_bytes(self) -> int:
        return len(self.bits)


class CountMinSketch:
    def __init__(self, width: int, depth: int) -> None:
        if width <= 0 or depth <= 0:
            raise ValueError("width and depth must be positive")
        self.width = width
        self.depth = depth
        self.rows = array("Q", [0]) * (width * depth)
        self.seeds = tuple(f"CMS-row-{i}".encode("ascii") for i in range(depth))

    def _index(self, key: str, row: int) -> int:
        digest = hashlib.blake2b(
            key.encode("utf-8"), digest_size=8, key=self.seeds[row]
        ).digest()
        return int.from_bytes(digest, "little") % self.width

    def add(self, key: str, amount: int = 1) -> None:
        for row in range(self.depth):
            index = row * self.width + self._index(key, row)
            self.rows[index] += amount

    def estimate(self, key: str) -> int:
        return min(
            self.rows[row * self.width + self._index(key, row)]
            for row in range(self.depth)
        )

    @property
    def memory_bytes(self) -> int:
        return (
            sys.getsizeof(self)
            + sys.getsizeof(self.__dict__)
            + sys.getsizeof(self.rows)
            + sys.getsizeof(self.seeds)
            + sum(sys.getsizeof(seed) for seed in self.seeds)
        )

    @property
    def payload_bytes(self) -> int:
        return self.rows.buffer_info()[1] * self.rows.itemsize


def parse_movie_id(line: str) -> int:
    fields = line.rstrip("\n").split("::")
    if len(fields) != 4:
        raise ValueError(f"Expected four ::-separated columns, got {len(fields)}")
    return int(fields[1])


def iter_movie_keys(path: Path) -> Iterator[str]:
    with path.open("r", encoding="latin-1", newline="") as stream:
        for line in stream:
            yield f"movie:{parse_movie_id(line)}"


def exact_ground_truth(path: Path) -> tuple[Counter[str], int]:
    """One-pass exact counts; kept separately as a small evaluation baseline."""
    counts: Counter[str] = Counter()
    records = 0
    for key in iter_movie_keys(path):
        counts[key] += 1
        records += 1
    return counts, records


def bloom_experiment(path: Path, truth: Counter[str], bit_count: int, hash_count: int,
                    repeats: int, negative_queries: int) -> dict:
    build_times: list[float] = []
    final_filter: BloomFilter | None = None
    record_count = 0
    for _ in range(repeats):
        bloom = BloomFilter(bit_count, hash_count)
        started = time.perf_counter()
        record_count = 0
        for key in iter_movie_keys(path):
            bloom.add(key)
            record_count += 1
        build_times.append(time.perf_counter() - started)
        final_filter = bloom
    assert final_filter is not None

    positives = list(truth)
    positive_start = time.perf_counter()
    false_negatives = sum(not final_filter.might_contain(key) for key in positives)
    positive_query_seconds = time.perf_counter() - positive_start

    false_positives = 0
    negative_start = time.perf_counter()
    for i in range(negative_queries):
        if final_filter.might_contain(f"movie:{10_000_000 + i}"):
            false_positives += 1
    negative_query_seconds = time.perf_counter() - negative_start

    unique_count = len(truth)
    theoretical_fpr = (1.0 - math.exp(-hash_count * unique_count / bit_count)) ** hash_count
    median_build = statistics.median(build_times)
    return {
        "algorithm": "Bloom Filter",
        "bit_count": bit_count,
        "hash_count": hash_count,
        "records_processed": record_count,
        "unique_movie_ids": unique_count,
        "false_positives": false_positives,
        "negative_queries": negative_queries,
        "false_positive_rate": false_positives / negative_queries,
        "theoretical_false_positive_rate": theoretical_fpr,
        "false_negatives": false_negatives,
        "positive_queries": len(positives),
        "payload_bytes": final_filter.payload_bytes,
        "structure_bytes_sys_getsizeof": final_filter.memory_bytes,
        "build_seconds_runs": build_times,
        "build_seconds_median": median_build,
        "records_per_second": record_count / median_build,
        "positive_query_seconds": positive_query_seconds,
        "negative_query_seconds": negative_query_seconds,
        "negative_queries_per_second": negative_queries / negative_query_seconds,
        "bit_occupancy": sum(byte.bit_count() for byte in final_filter.bits) / bit_count,
    }


def cms_metrics(sketch: CountMinSketch, truth: Counter[str]) -> dict:
    errors: list[int] = []
    relative_errors: list[float] = []
    estimates: dict[str, int] = {}
    for key, actual in truth.items():
        estimate = sketch.estimate(key)
        estimates[key] = estimate
        errors.append(estimate - actual)
        relative_errors.append((estimate - actual) / actual)

    exact_top = {key for key, _ in truth.most_common(10)}
    estimated_top = {key for key, _ in sorted(estimates.items(), key=lambda item: (-item[1], item[0]))[:10]}
    total_error = sum(errors)
    return {
        "mean_absolute_error": statistics.mean(errors),
        "max_absolute_error": max(errors),
        "mean_relative_error_percent": statistics.mean(relative_errors) * 100.0,
        "p95_relative_error_percent": sorted(relative_errors)[math.ceil(0.95 * len(relative_errors)) - 1] * 100.0,
        "total_overcount_percent_of_all_records": total_error / sum(truth.values()) * 100.0,
        "top10_recall": len(exact_top & estimated_top) / 10.0,
        "all_estimates_at_least_exact": all(error >= 0 for error in errors),
        "exact_top10": [{"movie_id": key.removeprefix("movie:"), "count": value} for key, value in truth.most_common(10)],
        "estimated_top10": [{"movie_id": key.removeprefix("movie:"), "estimate": estimates[key]} for key in sorted(estimates, key=lambda item: (-estimates[item], item))[:10]],
    }


def cms_experiment(path: Path, truth: Counter[str], width: int, depth: int,
                   repeats: int) -> dict:
    build_times: list[float] = []
    final_sketch: CountMinSketch | None = None
    record_count = 0
    for _ in range(repeats):
        sketch = CountMinSketch(width, depth)
        started = time.perf_counter()
        record_count = 0
        for key in iter_movie_keys(path):
            sketch.add(key)
            record_count += 1
        build_times.append(time.perf_counter() - started)
        final_sketch = sketch
    assert final_sketch is not None

    query_start = time.perf_counter()
    metrics = cms_metrics(final_sketch, truth)
    query_seconds = time.perf_counter() - query_start
    median_build = statistics.median(build_times)
    return {
        "algorithm": "Count-Min Sketch",
        "width": width,
        "depth": depth,
        "records_processed": record_count,
        "unique_movie_ids": len(truth),
        "payload_bytes": final_sketch.payload_bytes,
        "structure_bytes_sys_getsizeof": final_sketch.memory_bytes,
        "epsilon_bound": math.e / width,
        "delta_bound": math.exp(-depth),
        "build_seconds_runs": build_times,
        "build_seconds_median": median_build,
        "records_per_second": record_count / median_build,
        "all_keys_query_seconds": query_seconds,
        **metrics,
    }


def run(args: argparse.Namespace) -> dict:
    ratings_path = args.ratings.resolve()
    truth, records = exact_ground_truth(ratings_path)
    if records < 100_000:
        raise ValueError(f"Dataset has only {records} records; at least 100,000 are required")

    bloom_configs = [(16_384, 3), (32_768, 6), (65_536, 12)]
    cms_configs = [(64, 3), (256, 3), (512, 5), (2_048, 5)]
    bloom_results = [
        bloom_experiment(ratings_path, truth, bits, hashes, args.repeats, args.negative_queries)
        for bits, hashes in bloom_configs
    ]
    cms_results = [
        cms_experiment(ratings_path, truth, width, depth, args.repeats)
        for width, depth in cms_configs
    ]

    return {
        "dataset": {
            "name": "MovieLens 1M",
            "ratings_path": str(ratings_path),
            "records": records,
            "unique_movie_ids": len(truth),
            "record_format": "UserID::MovieID::Rating::Timestamp",
            "stream_key": "movie:<MovieID>",
            "ground_truth_distinct_frequency_map_entries": len(truth),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": __import__("os").cpu_count(),
            "timing_method": "time.perf_counter; median of repeated full file streaming passes",
            "repeats": args.repeats,
            "negative_queries_per_bloom_config": args.negative_queries,
        },
        "methodology": {
            "streaming": "Each timed pass opens ratings.dat and parses it one line at a time; records are not loaded as a collection.",
            "ground_truth": "A separate one-pass Counter over MovieID is held only for evaluation. It has O(number of distinct MovieIDs) memory and is excluded from sketch/filter memory.",
            "memory": "Reports compact payload bytes and sys.getsizeof-based structure bytes. Ground-truth Counter and input buffers are excluded.",
            "bloom_negative_queries": "Unseen keys movie:10000000 through the requested query range; key format matches inserted keys.",
            "cms_error": "Count-Min estimates are compared for all distinct observed MovieIDs; report mean absolute error, mean/p95 relative error, total overcount and exact top-10 recall.",
        },
        "bloom_filter": bloom_results,
        "count_min_sketch": cms_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path, default=Path("data/ml-1m/ratings.dat"))
    parser.add_argument("--output", type=Path, default=Path("results/benchmark_results.json"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--negative-queries", type=int, default=100_000)
    args = parser.parse_args()
    if args.repeats < 1 or args.negative_queries < 1:
        parser.error("--repeats and --negative-queries must be at least 1")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"records={result['dataset']['records']:,}")
    print(f"unique_movie_ids={result['dataset']['unique_movie_ids']:,}")
    print(f"results={args.output.resolve()}")
    for row in result["bloom_filter"]:
        print(f"Bloom m={row['bit_count']:,} k={row['hash_count']}: FPR={row['false_positive_rate']:.6%}, "
              f"memory={row['structure_bytes_sys_getsizeof']:,} B, throughput={row['records_per_second']:,.0f} records/s")
    for row in result["count_min_sketch"]:
        print(f"CMS w={row['width']:,} d={row['depth']}: mean_rel_error={row['mean_relative_error_percent']:.4f}%, "
              f"top10_recall={row['top10_recall']:.0%}, memory={row['structure_bytes_sys_getsizeof']:,} B, "
              f"throughput={row['records_per_second']:,.0f} records/s")


if __name__ == "__main__":
    main()

