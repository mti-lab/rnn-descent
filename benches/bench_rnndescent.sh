set -e

DATASET="siftsmall"
S=20
R=96
T1=4
T2=15

export OMP_NUM_THREADS=16
FN_RESULT="benches/results/rnndescent.json"
./build/benches/bench_rnndescent \
    --S ${S} \
    --R ${R} \
    --T1 ${T1} \
    --T2 ${T2} \
    --dataset ${DATASET} \
    --fn_result ${FN_RESULT}
