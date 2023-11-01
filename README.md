# Relative NN-Descent: A Fast Index Construction for Graph-Based Approximate Nearest Neighbor Search

[Paper](https://arxiv.org/abs/2310.20419)

## Prerequisite

We use [Faiss](https://github.com/facebookresearch/faiss) as a submodule. Please add the `--recursive` option when you clone this repository:
```
$ git clone --recursive git@github.com:mti-lab/rnn-descent.git
```

## Build

```
$ cmake \
    -B build \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_PYTHON=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DFAISS_OPT_LEVEL=avx2 \
    .
$ make -C build -j rnndescent
```

## Usage
Our index has the same interface as Faiss::Index. Please see the sample code in `benches/bench_rnndescent.cpp` for details.

## Reference

```
@inproceedings{ono2023rnndescent,
author = {Ono, Naoki and Matsui, Yusuke},
title = {Relative NN-Descent: A Fast Index Construction for Graph-Based Approximate Nearest Neighbor Search},
booktitle = {ACM International Conference on Multimedia (ACMMM)},
year = {2023},
}
```
