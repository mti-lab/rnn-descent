#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <rnn-descent/RNNDescent.h>

#include <iostream>

namespace rnndescent {

void gen_random(std::mt19937& rng, int* addr, const int size, const int N) {
    for (int i = 0; i < size; ++i) {
        addr[i] = rng() % (N - size);
    }
    std::sort(addr, addr + size);
    for (int i = 1; i < size; ++i) {
        if (addr[i] <= addr[i - 1]) {
            addr[i] = addr[i - 1] + 1;
        }
    }
    int off = rng() % N;
    for (int i = 0; i < size; ++i) {
        addr[i] = (addr[i] + off) % N;
    }
}

// Insert a new point into the candidate pool in ascending order
int insert_into_pool(faiss::nndescent::Neighbor* addr, int size,
                     faiss::nndescent::Neighbor nn) {
    // find the location to insert
    int left = 0, right = size - 1;
    if (addr[left].distance > nn.distance) {
        memmove((char*)&addr[left + 1], &addr[left],
                size * sizeof(faiss::nndescent::Neighbor));
        addr[left] = nn;
        return left;
    }
    if (addr[right].distance < nn.distance) {
        addr[size] = nn;
        return size;
    }
    while (left < right - 1) {
        int mid = (left + right) / 2;
        if (addr[mid].distance > nn.distance)
            right = mid;
        else
            left = mid;
    }
    // check equal ID

    while (left > 0) {
        if (addr[left].distance < nn.distance) break;
        if (addr[left].id == nn.id) return size + 1;
        left--;
    }
    if (addr[left].id == nn.id || addr[right].id == nn.id) return size + 1;
    memmove((char*)&addr[right + 1], &addr[right],
            (size - right) * sizeof(faiss::nndescent::Neighbor));
    addr[right] = nn;
    return right;
}

RNNDescent::RNNDescent(const int d) : d(d) {}

RNNDescent::~RNNDescent() {}

void RNNDescent::init_graph(faiss::DistanceComputer& qdis) {
    graph.reserve(ntotal);
    {
        std::mt19937 rng(random_seed * 6007);
        for (int i = 0; i < ntotal; i++) {
            graph.push_back(faiss::nndescent::Nhood(L, S, rng, (int)ntotal));
        }
    }

#pragma omp parallel
    {
        std::mt19937 rng(random_seed * 7741 + omp_get_thread_num());
#pragma omp for
        for (int i = 0; i < ntotal; i++) {
            std::vector<int> tmp(S);

            gen_random(rng, tmp.data(), S, ntotal);

            for (int j = 0; j < S; j++) {
                int id = tmp[j];
                if (id == i) continue;
                float dist = qdis.symmetric_dis(i, id);

                graph[i].pool.push_back(
                    faiss::nndescent::Neighbor(id, dist, true));
            }
            std::make_heap(graph[i].pool.begin(), graph[i].pool.end());
            graph[i].pool.reserve(L);
        }
    }
}

void RNNDescent::insert_nn(int id, int nn_id, float distance, bool flag) {
    auto& nhood = graph[id];
    {
        std::lock_guard<std::mutex> guard(nhood.lock);
        nhood.pool.emplace_back(nn_id, distance, flag);
    }
}

void RNNDescent::update_neighbors(faiss::DistanceComputer& qdis) {
#pragma omp parallel for schedule(dynamic, 256)
    for (int u = 0; u < ntotal; ++u) {
        auto& nhood = graph[u];
        auto& pool = nhood.pool;
        std::vector<faiss::nndescent::Neighbor> new_pool;
        std::vector<faiss::nndescent::Neighbor> old_pool;
        {
            std::lock_guard<std::mutex> guard(nhood.lock);
            old_pool = pool;
            pool.clear();
        }
        std::sort(old_pool.begin(), old_pool.end());
        old_pool.erase(std::unique(old_pool.begin(), old_pool.end(),
                                   [](faiss::nndescent::Neighbor& a,
                                      faiss::nndescent::Neighbor& b) {
                                       return a.id == b.id;
                                   }),
                       old_pool.end());

        for (auto&& nn : old_pool) {
            bool ok = true;
            for (auto&& other_nn : new_pool) {
                if (!nn.flag && !other_nn.flag) {
                    continue;
                }
                if (nn.id == other_nn.id) {
                    ok = false;
                    break;
                }
                float distance = qdis.symmetric_dis(nn.id, other_nn.id);
                if (distance < nn.distance) {
                    ok = false;
                    insert_nn(other_nn.id, nn.id, distance, true);
                    break;
                }
            }
            if (ok) {
                new_pool.emplace_back(nn);
            }
        }

        for (auto&& nn : new_pool) {
            nn.flag = false;
        }
        {
            std::lock_guard<std::mutex> guard(nhood.lock);
            pool.insert(pool.end(), new_pool.begin(), new_pool.end());
        }
    }
}

void RNNDescent::add_reverse_edges() {
    std::vector<std::vector<faiss::nndescent::Neighbor>> reverse_pools(ntotal);

#pragma omp parallel for
    for (int u = 0; u < ntotal; ++u) {
        for (auto&& nn : graph[u].pool) {
            std::lock_guard<std::mutex> guard(graph[nn.id].lock);
            reverse_pools[nn.id].emplace_back(u, nn.distance, nn.flag);
        }
    }

#pragma omp parallel for
    for (int u = 0; u < ntotal; ++u) {
        auto& pool = graph[u].pool;
        for (auto&& nn : pool) {
            nn.flag = true;
        }
        auto& rpool = reverse_pools[u];
        rpool.insert(rpool.end(), pool.begin(), pool.end());
        pool.clear();
        std::sort(rpool.begin(), rpool.end());
        rpool.erase(std::unique(rpool.begin(), rpool.end(),
                                [](faiss::nndescent::Neighbor& a,
                                   faiss::nndescent::Neighbor& b) {
                                    return a.id == b.id;
                                }),
                    rpool.end());
        if (rpool.size() > R) {
            rpool.resize(R);
        }
    }

#pragma omp parallel for
    for (int u = 0; u < ntotal; ++u) {
        for (auto&& nn : reverse_pools[u]) {
            std::lock_guard<std::mutex> guard(graph[nn.id].lock);
            graph[nn.id].pool.emplace_back(u, nn.distance, nn.flag);
        }
    }

#pragma omp parallel for
    for (int u = 0; u < ntotal; ++u) {
        auto& pool = graph[u].pool;
        std::sort(pool.begin(), pool.end());
        if (pool.size() > R) {
            pool.resize(R);
        }
    }
}

void RNNDescent::build(faiss::DistanceComputer& qdis, const int n,
                       bool verbose) {
    if (verbose) {
        printf("Parameters: S=%d, R=%d, T1=%d, T2=%d\n", S, R, T1, T2);
    }

    ntotal = n;
    init_graph(qdis);

    for (int t1 = 0; t1 < T1; ++t1) {
        if (verbose) {
            std::cout << "Iter " << t1 << " : " << std::flush;
        }
        for (int t2 = 0; t2 < T2; ++t2) {
            update_neighbors(qdis);
            if (verbose) {
                std::cout << "#" << std::flush;
            }
        }

        if (t1 != T1 - 1) {
            add_reverse_edges();
        }

        if (verbose) {
            printf("\n");
        }
    }

#pragma omp parallel for
    for (int u = 0; u < n; ++u) {
        auto& pool = graph[u].pool;
        std::sort(pool.begin(), pool.end());
        pool.erase(std::unique(pool.begin(), pool.end(),
                               [](faiss::nndescent::Neighbor& a,
                                  faiss::nndescent::Neighbor& b) {
                                   return a.id == b.id;
                               }),
                   pool.end());
    }

    offsets.resize(ntotal + 1);
    offsets[0] = 0;
    for (int u = 0; u < ntotal; ++u) {
        offsets[u + 1] = offsets[u] + graph[u].pool.size();
    }

    final_graph.resize(offsets.back(), -1);
#pragma omp parallel for
    for (int u = 0; u < n; ++u) {
        auto& pool = graph[u].pool;
        int offset = offsets[u];
        for (int i = 0; i < pool.size(); ++i) {
            final_graph[offset + i] = pool[i].id;
        }
    }
    std::vector<faiss::nndescent::Nhood>().swap(graph);

    has_built = true;
}

void RNNDescent::search(faiss::DistanceComputer& qdis, const int topk,
                        faiss::idx_t* indices, float* dists,
                        faiss::VisitedTable& vt) const {
    FAISS_THROW_IF_NOT_MSG(has_built, "The index is not build yet.");
    int L = std::max(search_L, topk);

    // candidate pool, the K best items is the result.
    std::vector<faiss::nndescent::Neighbor> retset(L + 1);

    // Randomly choose L points to initialize the candidate pool
    std::vector<int> init_ids(L);
    std::mt19937 rng(random_seed);

    gen_random(rng, init_ids.data(), L, ntotal);
    for (int i = 0; i < L; i++) {
        int id = init_ids[i];
        float dist = qdis(id);
        retset[i] = faiss::nndescent::Neighbor(id, dist, true);
    }

    // Maintain the candidate pool in ascending order
    std::sort(retset.begin(), retset.begin() + L);

    int k = 0;

    // Stop until the smallest position updated is >= L
    while (k < L) {
        int nk = L;

        if (retset[k].flag) {
            retset[k].flag = false;
            int n = retset[k].id;

            int offset = offsets[n];
            int K = std::min(K0, offsets[n + 1] - offset);
            for (int m = 0; m < K; ++m) {
                int id = final_graph[offset + m];
                if (vt.get(id)) continue;

                vt.set(id);
                float dist = qdis(id);
                if (dist >= retset[L - 1].distance) continue;

                faiss::nndescent::Neighbor nn(id, dist, true);
                int r = insert_into_pool(retset.data(), L, nn);

                if (r < nk) nk = r;
            }
        }
        if (nk <= k)
            k = nk;
        else
            ++k;
    }
    for (size_t i = 0; i < topk; i++) {
        indices[i] = retset[i].id;
        dists[i] = retset[i].distance;
    }

    vt.advance();
};

void RNNDescent::reset() {
    has_built = false;
    ntotal = 0;
    final_graph.resize(0);
    offsets.resize(0);
}

}  // namespace rnndescent