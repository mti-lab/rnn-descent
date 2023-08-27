set -e

DATASET="siftsmall"
M=32
efConstruction=500

export OMP_NUM_THREADS=16
FN_RESULT="benches/results/hnsw.json"
./build/benches/bench_hnsw \
    --M ${M} \
    --efConstruction ${efConstruction} \
    --dataset ${DATASET} \
    --fn_result ${FN_RESULT}
