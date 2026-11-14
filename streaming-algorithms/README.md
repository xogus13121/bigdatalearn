# Streaming Algorithms Benchmark (MovieLens 1M)

Implements and compares a Bloom Filter and Count-Min Sketch without loading the ratings stream into memory. The benchmark data is not included in Git because the GroupLens license prohibits redistribution.

## Dataset

Download MovieLens 1M from the official GroupLens source and place `ratings.dat` at `data/ml-1m/ratings.dat`:

https://files.grouplens.org/datasets/movielens/ml-1m.zip

MovieLens 1M contains 1,000,209 anonymous rating events. This benchmark uses `MovieID` as the stream key: Bloom Filter answers approximate membership, while Count-Min Sketch estimates per-movie event counts.

## Run

Requires Python 3.10+; the implementation itself uses only the standard library.

```powershell
python src/streaming_benchmark.py --repeats 3 --negative-queries 100000
```

The script streams one record per line. It writes experiment measurements to `results/benchmark_results.json`. Exact `Counter` counts are built in a separate one-pass evaluation baseline; baseline memory is reported separately and is not counted as sketch/filter memory.

## Implementations

- Bloom Filter: bytearray bit vector; deterministic BLAKE2b double hashing; configurable bit count and hash count.
- Count-Min Sketch: packed unsigned 64-bit counters in `array('Q')`; one independently keyed BLAKE2b hash per row; configurable width and depth.

## License and data handling

The MovieLens README allows research/education use but prohibits redistributing the dataset. The archive and extracted data are therefore excluded from the public source repository by `.gitignore`. Cite Harper and Konstan (2015) in derivative reports.

