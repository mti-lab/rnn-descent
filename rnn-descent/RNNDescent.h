#include <faiss/impl/NNDescent.h>

#include <vector>

namespace rnndescent {

struct RNNDescent {
    using storage_idx_t = int;

    using KNNGraph = std::vector<faiss::nndescent::Nhood>;

    explicit RNNDescent(const int d);

    ~RNNDescent();

    void build(faiss::DistanceComputer& qdis, const int n, bool verbose);

    void search(faiss::DistanceComputer& qdis, const int topk,
                faiss::idx_t* indices, float* dists,
                faiss::VisitedTable& vt) const;

    void reset();

    /// Initialize the KNN graph randomly
    void init_graph(faiss::DistanceComputer& qdis);

    void update_neighbors(faiss::DistanceComputer& qdis);
    void add_reverse_edges();

    void insert_nn(int id, int nn_id, float distance, bool flag);

    bool has_built = false;

    int T1 = 4;
    int T2 = 15;
    int S = 16;
    int R = 96;
    int K0 = 32; // maximum out-degree (mentioned as K in the original paper)

    int search_L = 0;        // size of candidate pool in searching
    int random_seed = 2021;  // random seed for generators

    int d;  // dimensions
    int L = 8;  // initial size of memory allocation

    int ntotal = 0;

    KNNGraph graph;
    std::vector<int> final_graph;
    std::vector<int> offsets;
};

}  // namespace rnndescent