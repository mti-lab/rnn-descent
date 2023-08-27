## Benchmark

Please put your dataset on `benches/datasets` directory.

Example of [siftsmall](http://corpus-texmex.irisa.fr) dataset:
```
benches/datasets
|- siftsmall
    |- siftsmall_base.fvecs
    |- siftsmall_groundtruth.ivecs
    |- siftsmall_learn.fvecs
    |- siftsmall_query.fvecs
```

If you want to use another dataset, please add the settings to `datasets/settings.json`.

<!-- How to build and run the benchmark: -->
Build & run:
```
$ make -C build -j bench_rnndescent
$ ./benches/bench_rnndescent.sh
```
