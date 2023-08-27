#include <faiss/Index.h>

#include <rnn-descent/RNNDescent.h>

namespace rnndescent {

using idx_t = faiss::idx_t;

struct IndexRNNDescent : faiss::Index {
    bool own_fields;
    faiss::Index* storage;
    bool verbose;

    RNNDescent rnndescent;

    explicit IndexRNNDescent(int d = 0, int K = 32,
                             faiss::MetricType metric = faiss::METRIC_L2);
    explicit IndexRNNDescent(Index* storage, int K = 32);

    ~IndexRNNDescent() override;

    void add(idx_t n, const float* x) override;

    void train(idx_t n, const float* x) override;

    void search(idx_t n, const float* x, idx_t k, float* distances,
                idx_t* labels,
                const faiss::SearchParameters* params = nullptr) const override;

    void reconstruct(idx_t key, float* recons) const override;

    void reset() override;
};

}  // namespace rnndescent