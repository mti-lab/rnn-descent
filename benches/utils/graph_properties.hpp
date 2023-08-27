#include <map>
#include <nlohmann/json.hpp>
#include <numeric>
#include <set>
#include <stack>
#include <vector>

struct UnionFind {
    std::vector<int> parents;

    UnionFind(int n) : parents(n) {
        std::iota(parents.begin(), parents.end(), 0);
    }

    bool same(int a, int b) { return root(a) == root(b); }

    void merge(int a, int b) {
        if (same(a, b)) {
            return;
        }
        parents[root(a)] = root(b);
    }

    int root(int a) {
        if (parents[a] == a) {
            return a;
        }
        return parents[a] = root(parents[a]);
    }
};

int count_connected_components(
    const int n, const std::vector<int>& neighbors,
    const std::function<std::tuple<int, int>(int)>& get_range) {
    UnionFind uf(n);
    for (int u = 0; u < n; ++u) {
        auto [left, right] = get_range(u);
        for (int j = left; j < right; ++j) {
            int v = neighbors[j];
            if (v >= 0 && v < n) {
                uf.merge(u, v);
            }
        }
    }

    std::set<int> roots;
    for (int i = 0; i < n; ++i) {
        roots.insert(uf.root(i));
    }
    return roots.size();
}

template <class T, class U>
std::tuple<std::vector<T>, std::vector<U>> map_to_vecs(std::map<T, U>& m) {
    std::vector<T> t;
    std::vector<U> u;
    for (auto&& p : m) {
        t.emplace_back(p.first);
        u.emplace_back(p.second);
    }
    return {t, u};
}

std::map<int, int> outdegree_distribution(
    const int n, const std::vector<int>& neighbors,
    const std::function<std::tuple<int, int>(int)>& get_range) {
    std::map<int, int> dist;
    for (int i = 0; i < n; ++i) {
        auto [left, right] = get_range(i);
        int deg = 0;
        for (int j = left; j < right; ++j) {
            int v = neighbors[j];
            if (v >= 0 && v < n) {
                ++deg;
            }
        }
        ++dist[deg];
    }
    return dist;
}

std::map<int, int> indegree_distribution(
    const int n, const std::vector<int>& neighbors,
    const std::function<std::tuple<int, int>(int)>& get_range) {
    std::map<int, int> dist;
    std::vector<int> indegrees(n);
    for (int i = 0; i < n; ++i) {
        auto [left, right] = get_range(i);
        for (int j = left; j < right; ++j) {
            int v = neighbors[j];
            if (v >= 0 && v < n) {
                ++indegrees[v];
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        ++dist[indegrees[i]];
    }
    return dist;
}

nlohmann::json graph_properties(
    const int n, const std::vector<int>& neighbors,
    const std::function<std::tuple<int, int>(int)>& get_range) {
    nlohmann::json properties;

    properties["connected_components"] =
        count_connected_components(n, neighbors, get_range);

    {
        nlohmann::json indegrees;
        auto m = indegree_distribution(n, neighbors, get_range);
        auto [x, y] = map_to_vecs(m);
        indegrees["degree"] = x;
        indegrees["nvertices"] = y;
        properties["dist_indeg"] = indegrees;
    }

    {
        nlohmann::json outdegrees;
        auto m = outdegree_distribution(n, neighbors, get_range);
        auto [x, y] = map_to_vecs(m);
        outdegrees["degree"] = x;
        outdegrees["nvertices"] = y;
        properties["dist_outdeg"] = outdegrees;

        int total_degrees = 0;
        for (int i = 0; i < m.size(); ++i) {
            total_degrees += x[i] * y[i];
        }
        properties["total_degrees"] = total_degrees;
    }

    return properties;
}
