set -e

DATASET="siftsmall"
K=64

export OMP_NUM_THREADS=16
FN_RESULT="benches/results/nndescent.json"
./build/benches/bench_nndescent \
    --K ${K} \
    --dataset ${DATASET} \
    --fn_result ${FN_RESULT}
