set -e

DATASET="siftsmall"
R=32

export OMP_NUM_THREADS=16
FN_RESULT="benches/results/nsg.json"
./build/benches/bench_nsg \
    --R ${R} \
    --dataset ${DATASET} \
    --fn_result ${FN_RESULT}
