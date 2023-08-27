#pragma once

#include <faiss/Index.h>

#include <benches/utils/Timer.hpp>
#include <cassert>
#include <memory>

float recall_at_k(const size_t nq, const size_t k, const size_t stride_labels,
                  const std::unique_ptr<faiss::idx_t[]>& labels,
                  const size_t stride_gt,
                  const std::unique_ptr<faiss::idx_t[]>& gt) {
    assert(k <= stride_labels);

    size_t n_correct = 0;
    for (int i = 0; i < nq; ++i) {
        for (int j = 0; j < k; ++j) {
            if (labels[i * stride_labels + j] == gt[i * stride_gt]) {
                ++n_correct;
                break;
            }
        }
    }

    return static_cast<float>(n_correct) / static_cast<float>(nq);
};

template <class T>
std::pair<double, double> compute_qps_recall(
    const T& index, const int nq,
    const std::unique_ptr<float[]>& xq, const int k,
    const std::unique_ptr<faiss::idx_t[]>& gt) {
    static_assert(std::is_base_of<faiss::Index, T>::value);
    using idx_t = faiss::idx_t;

    std::unique_ptr<idx_t[]> I(new idx_t[nq]);
    std::unique_ptr<float[]> D(new float[nq]);

    Timer timer;
    index.search(nq, xq.get(), 1, D.get(), I.get());
    auto elapsed = timer.elapsed_ns() * 1e-9;

    float qps = nq / elapsed;
    float r_at_1 = recall_at_k(nq, 1, 1, I, k, gt);

    return {qps, r_at_1};
}