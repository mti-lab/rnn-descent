/**
 * This implementation is heavily based on faiss::IndexNNDescent.cpp
 * (https://github.com/facebookresearch/faiss/blob/main/faiss/IndexNNDescent.cpp)
 */

// -*- c++ -*-

#include <omp.h>
#include <rnn-descent/IndexRNNDescent.h>

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <queue>
#include <unordered_set>

#ifdef __SSE__
#endif

#include <faiss/IndexFlat.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>

extern "C" {

/* declare BLAS functions, see http://www.netlib.org/clapack/cblas/ */

int sgemm_(const char* transa, const char* transb, FINTEGER* m, FINTEGER* n,
           FINTEGER* k, const float* alpha, const float* a, FINTEGER* lda,
           const float* b, FINTEGER* ldb, float* beta, float* c, FINTEGER* ldc);
}

namespace rnndescent {

using namespace faiss;

using storage_idx_t = NNDescent::storage_idx_t;

/**************************************************************
 * add / search blocks of descriptors
 **************************************************************/

namespace {

/* Wrap the distance computer into one that negates the
   distances. This makes supporting INNER_PRODUCE search easier */

struct NegativeDistanceComputer : DistanceComputer {
    /// owned by this
    DistanceComputer* basedis;

    explicit NegativeDistanceComputer(DistanceComputer* basedis)
        : basedis(basedis) {}

    void set_query(const float* x) override { basedis->set_query(x); }

    /// compute distance of vector i to current query
    float operator()(idx_t i) override { return -(*basedis)(i); }

    /// compute distance between two stored vectors
    float symmetric_dis(idx_t i, idx_t j) override {
        return -basedis->symmetric_dis(i, j);
    }

    ~NegativeDistanceComputer() override { delete basedis; }
};

DistanceComputer* storage_distance_computer(const Index* storage) {
    if (is_similarity_metric(storage->metric_type)) {
        return new NegativeDistanceComputer(storage->get_distance_computer());
    } else {
        return storage->get_distance_computer();
    }
}

}  // namespace

/**************************************************************
 * IndexRNNDescent implementation
 **************************************************************/

IndexRNNDescent::IndexRNNDescent(int d, int K, MetricType metric)
    : Index(d, metric), rnndescent(d), own_fields(false), storage(nullptr) {
    // the default storage is IndexFlat
    storage = new IndexFlat(d, metric);
    own_fields = true;
}

IndexRNNDescent::IndexRNNDescent(Index* storage, int K)
    : Index(storage->d, storage->metric_type),
      rnndescent(storage->d),
      own_fields(false),
      storage(storage) {}

IndexRNNDescent::~IndexRNNDescent() {
    if (own_fields) {
        delete storage;
    }
}

void IndexRNNDescent::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(storage,
                           "Please use IndexNNDescentFlat (or variants) "
                           "instead of IndexNNDescent directly");
    // nndescent structure does not require training
    storage->train(n, x);
    is_trained = true;
}

void IndexRNNDescent::search(idx_t n, const float* x, idx_t k, float* distances,
                             idx_t* labels,
                             const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(!params,
                           "search params not supported for this index");
    FAISS_THROW_IF_NOT_MSG(storage);

    idx_t check_period =
        InterruptCallback::get_period_hint(d * rnndescent.search_L);

    for (idx_t i0 = 0; i0 < n; i0 += check_period) {
        idx_t i1 = std::min(i0 + check_period, n);

#pragma omp parallel
        {
            VisitedTable vt(ntotal);

            DistanceComputer* dis = storage_distance_computer(storage);
            ScopeDeleter1<DistanceComputer> del(dis);

#pragma omp for
            for (idx_t i = i0; i < i1; i++) {
                idx_t* idxi = labels + i * k;
                float* simi = distances + i * k;
                dis->set_query(x + i * d);

                rnndescent.search(*dis, k, idxi, simi, vt);
            }
        }
        InterruptCallback::check();
    }

    if (metric_type == METRIC_INNER_PRODUCT) {
        // we need to revert the negated distances
        for (size_t i = 0; i < k * n; i++) {
            distances[i] = -distances[i];
        }
    }
}

void IndexRNNDescent::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(storage,
                           "Please use IndexNNDescentFlat (or variants) "
                           "instead of IndexNNDescent directly");
    FAISS_THROW_IF_NOT(is_trained);

    if (ntotal != 0) {
        fprintf(stderr,
                "WARNING NNDescent doest not support dynamic insertions,"
                "multiple insertions would lead to re-building the index");
    }

    storage->add(n, x);
    ntotal = storage->ntotal;

    DistanceComputer* dis = storage_distance_computer(storage);
    ScopeDeleter1<DistanceComputer> del(dis);
    rnndescent.build(*dis, ntotal, verbose);
}

void IndexRNNDescent::reset() {
    rnndescent.reset();
    storage->reset();
    ntotal = 0;
}

void IndexRNNDescent::reconstruct(idx_t key, float* recons) const {
    storage->reconstruct(key, recons);
}

}  // namespace rnndescent
