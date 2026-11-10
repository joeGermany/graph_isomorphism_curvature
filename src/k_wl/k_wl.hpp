#pragma once
#include <algorithm>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace k_wl {
using Matrix = std::vector<std::vector<int>>;
struct Result {
    // One stable color per original vertex (not one per ordered k-tuple).
    std::vector<int> colors;
    int color_count = 0;  // Number of distinct vertex colors.
    int rounds = 0;
};

inline std::pair<std::vector<int>,int> compress(
    const std::vector<std::vector<int>>& keys) {
    std::vector<std::size_t> ids(keys.size());
    std::iota(ids.begin(), ids.end(), 0);
    std::sort(ids.begin(), ids.end(), [&](auto a, auto b) {
        return keys[a] < keys[b];
    });
    std::vector<int> colors(keys.size());
    int count = 0;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (!i || keys[ids[i]] != keys[ids[i-1]]) ++count;
        colors[ids[i]] = count-1;
    }
    return {std::move(colors), count};
}

// Standard separate-coordinate k-WL (k>=2); 1-WL is neighbor color refinement.
// G is a symmetric 0/1 adjacency matrix. Input node colors are optional.
// Computes every ordered k-tuple internally, but returns only the vertex colors
// obtained from the diagonal tuples (v,v,...,v).
// The vertex-color IDs are normalized to 0..color_count-1 in input vertex order.
inline Result run(const Matrix& G, int k,
                  const std::vector<int>& vertex_colors = {},
                  std::size_t max_tuples = 2000000) {
    const std::size_t n = G.size();
    if (k < 1) throw std::invalid_argument("k must be positive");
    if (n == 0) return {};
    if (!vertex_colors.empty() && vertex_colors.size() != n)
        throw std::invalid_argument("wrong vertex color count");
    for (std::size_t u=0;u<n;++u)
        if (G[u].size()!=n) throw std::invalid_argument("adjacency must be square");
    for (std::size_t u=0;u<n;++u) {
        if (G[u][u]) throw std::invalid_argument("self-loops are not supported");
        for (std::size_t v=0;v<n;++v)
            if ((G[u][v]!=0 && G[u][v]!=1) || G[u][v]!=G[v][u])
                throw std::invalid_argument("expected symmetric simple adjacency");
    }
    std::size_t N=1;
    for (int j=0;j<k;++j) {
        if (N > max_tuples/n) throw std::length_error("too many k-tuples");
        N*=n;
    }
    std::vector<std::size_t> powers(k,1);
    for (int j=k-2;j>=0;--j) powers[j]=powers[j+1]*n;

    std::vector<std::vector<int>> atomic(N);
    for (std::size_t id=0;id<N;++id) {
        std::vector<int> t(k);
        for (int j=0;j<k;++j) t[j] = int(id/powers[j]%n);
        auto& key=atomic[id];
        for (int j=0;j<k;++j)
            key.push_back(vertex_colors.empty()?0:vertex_colors[t[j]]);
        for (int j=0;j<k;++j)
            for (int l=0;l<j;++l)
                key.push_back(t[j]==t[l]?2:G[t[j]][t[l]]);
    }
    auto first=compress(atomic);
    auto colors=std::move(first.first);
    int count=first.second;
    int rounds=0;
    for (;;) {
        std::vector<std::vector<int>> keys(N);
        for (std::size_t id=0;id<N;++id) {
            auto& key=keys[id];
            key.reserve(1+std::size_t(k)*(n+1));
            key.push_back(colors[id]);
            for (int j=0;j<k;++j) {
                std::vector<int> bag;
                bag.reserve(n);
                const std::size_t old=id/powers[j]%n;
                const std::size_t base=id-old*powers[j];
                for (std::size_t v=0;v<n;++v) {
                    if (k==1 && !G[id][v]) continue;
                    bag.push_back(colors[base+v*powers[j]]);
                }
                std::sort(bag.begin(),bag.end());
                if (k==1) {
                    key.push_back(int(bag.size()));
                    key.insert(key.end(),n-bag.size(),-1);
                }
                key.insert(key.end(),bag.begin(),bag.end());
            }
        }
        auto next=compress(keys);
        ++rounds;
        if (next.second == count) {
            // A vertex v is represented by the diagonal tuple (v,...,v).
            // Extract only n colors; do not transfer n^k tuple colors to Python.
            std::vector<int> final_vertex_colors(n);
            std::unordered_map<int, int> diagonal_color_ids;
            for (std::size_t v = 0; v < n; ++v) {
                std::size_t diagonal_id = 0;
                for (const auto stride : powers)
                    diagonal_id += v * stride;
                const int tuple_color = next.first[diagonal_id];
                const int next_vertex_color = static_cast<int>(diagonal_color_ids.size());
                const auto [it, inserted] = diagonal_color_ids.emplace(
                    tuple_color, next_vertex_color);
                final_vertex_colors[v] = it->second;
            }
            return {std::move(final_vertex_colors),
                    static_cast<int>(diagonal_color_ids.size()), rounds};
        }
        colors=std::move(next.first);
        count=next.second;
    }
}

// Convenience API when only a color for each input vertex is needed.
inline std::vector<int> vertex_colors(const Matrix& graph, int k,
                                      const std::vector<int>& initial_vertex_colors = {},
                                      std::size_t max_tuples = 2000000) {
    return run(graph, k, initial_vertex_colors, max_tuples).colors;
}
} // namespace k_wl
