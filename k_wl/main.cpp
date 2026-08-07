#include "weisfeiler_lehman.h"
#include "Tuple.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using wl::GraphNode;
using wl::TupleMap;

struct GraphInput {
    int n = 0;
    std::vector<std::tuple<int, int, int>> edges; // u, v, label
};

static std::vector<int> extract_ints(const std::string& line) {
    std::vector<int> out;
    std::regex r("-?\\d+");
    auto begin = std::sregex_iterator(line.begin(), line.end(), r);
    auto end = std::sregex_iterator();

    for (auto it = begin; it != end; ++it) {
        out.push_back(std::stoi(it->str()));
    }

    return out;
}

static GraphInput read_graph(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Could not open graph file: " + path);
    }

    GraphInput G;
    std::string line;
    bool first_graph_line_seen = false;
    bool dimacs = false;

    while (std::getline(in, line)) {
        if (line.empty()) continue;

        std::string trimmed = line;
        trimmed.erase(trimmed.begin(), std::find_if(trimmed.begin(), trimmed.end(),
                    [](unsigned char ch) { return !std::isspace(ch); }));

        if (trimmed.empty()) continue;
        if (trimmed[0] == '#') continue;
        if (trimmed[0] == 'c') continue;

        std::stringstream ss(trimmed);
        std::string first;
        ss >> first;

        if (first == "p") {
            std::string edge_word;
            int n, m;
            ss >> edge_word >> n >> m;
            G.n = n;
            dimacs = true;
            first_graph_line_seen = true;
            continue;
        }

        if (first == "e") {
            int u, v;
            int label = 1;
            ss >> u >> v;
            if (ss >> label) {
                // use supplied label
            }

            if (dimacs) {
                --u;
                --v;
            }

            G.edges.push_back({u, v, label});
            continue;
        }

        std::vector<int> nums = extract_ints(trimmed);

        if (!first_graph_line_seen) {
            if (nums.size() < 2) {
                throw std::runtime_error("First graph line must be: n m");
            }
            G.n = nums[0];
            first_graph_line_seen = true;
            continue;
        }

        if (nums.size() >= 2) {
            int u = nums[0];
            int v = nums[1];
            int label = 1;

            if (nums.size() >= 3) {
                label = nums[2];
            }

            G.edges.push_back({u, v, label});
        }
    }

    if (G.n <= 0) {
        throw std::runtime_error("Could not read number of vertices.");
    }

    return G;
}

static std::vector<std::vector<int>> read_partition(const std::string& path, int n) {
    std::vector<std::vector<int>> partition;

    if (path.empty() || path == "-") {
        return partition;
    }

    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Could not open partition file: " + path);
    }

    std::string line;
    std::vector<std::vector<int>> raw_lines;

    while (std::getline(in, line)) {
        if (line.empty()) continue;

        std::string trimmed = line;
        trimmed.erase(trimmed.begin(), std::find_if(trimmed.begin(), trimmed.end(),
                    [](unsigned char ch) { return !std::isspace(ch); }));

        if (trimmed.empty()) continue;
        if (trimmed[0] == '#') continue;

        std::vector<int> nums = extract_ints(trimmed);
        if (!nums.empty()) {
            raw_lines.push_back(nums);
        }
    }

    // Support files where first line is just number of classes.
    if (!raw_lines.empty() && raw_lines[0].size() == 1) {
        int claimed = raw_lines[0][0];
        if (claimed == static_cast<int>(raw_lines.size()) - 1) {
            raw_lines.erase(raw_lines.begin());
        }
    }

    std::vector<int> seen(n, 0);

    for (const auto& cls : raw_lines) {
        std::vector<int> clean;

        for (int v : cls) {
            if (v < 0 || v >= n) {
                throw std::runtime_error("Partition contains vertex outside range 0..n-1.");
            }

            if (seen[v]) {
                throw std::runtime_error("Partition contains repeated vertex.");
            }

            seen[v] = 1;
            clean.push_back(v);
        }

        if (!clean.empty()) {
            partition.push_back(clean);
        }
    }

    return partition;
}

static std::vector<GraphNode> build_graph_nodes(
    int n,
    const std::vector<std::tuple<int, int, int>>& edges,
    const std::vector<std::vector<int>>& partition,
    bool directed
) {
    std::vector<GraphNode> nodes(n);

    for (int i = 0; i < n; ++i) {
        nodes[i].idx = i;
        nodes[i].color = 0;
    }

    // Same idea as Sage wrapper:
    // partition colors start from 1; missing vertices get color 0.
    for (int c = 0; c < static_cast<int>(partition.size()); ++c) {
        for (int v : partition[c]) {
            nodes[v].color = c + 1;
        }
    }

    for (auto [u, v, label] : edges) {
        if (u < 0 || u >= n || v < 0 || v >= n) {
            throw std::runtime_error("Edge contains vertex outside range 0..n-1.");
        }

        if (label <= 0) {
            throw std::runtime_error("Edge labels must be positive. Label 0 means non-edge internally.");
        }

        nodes[u].adj_list.push_back({v, label});

        if (!directed) {
            nodes[v].adj_list.push_back({u, label});
        }
    }

    return nodes;
}

static std::vector<std::vector<int>> extract_vertex_classes(const TupleMap& coloring, int k) {
    std::map<int, std::vector<int>> by_color;

    for (const auto& item : coloring) {
        const Tuple<int>& tup = item.first;
        int color = item.second;

        if (tup.size() != k) continue;

        int first = tup[0];
        bool diagonal = true;

        for (int i = 1; i < k; ++i) {
            if (tup[i] != first) {
                diagonal = false;
                break;
            }
        }

        if (diagonal) {
            by_color[color].push_back(first);
        }
    }

    std::vector<std::vector<int>> classes;

    for (auto& item : by_color) {
        auto& cls = item.second;
        std::sort(cls.begin(), cls.end());
        classes.push_back(cls);
    }

    std::sort(classes.begin(), classes.end(),
              [](const std::vector<int>& a, const std::vector<int>& b) {
                  if (a.size() != b.size()) return a.size() < b.size();
                  return a < b;
              });

    return classes;
}

static void write_classes(const std::vector<std::vector<int>>& classes, const std::string& out_path) {
    std::ostream* out = &std::cout;
    std::ofstream fout;

    if (!out_path.empty()) {
        fout.open(out_path);
        if (!fout) {
            throw std::runtime_error("Could not open output file: " + out_path);
        }
        out = &fout;
    }

    *out << classes.size() << "\n";

    for (const auto& cls : classes) {
        *out << "[";
        for (size_t i = 0; i < cls.size(); ++i) {
            if (i) *out << ", ";
            *out << cls[i];
        }
        *out << "]\n";
    }
}
static int count_tuple_colors(const TupleMap& coloring) {
    std::set<int> colors;

    for (const auto& item : coloring) {
        colors.insert(item.second);
    }

    return static_cast<int>(colors.size());
}
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr
            << "Usage:\n"
            << "  ./run_wl graph.txt k [partition.txt] [output.txt] [--directed]\n\n"
            << "Graph format:\n"
            << "  n m\n"
            << "  u v\n"
            << "  u v label\n\n"
            << "Partition format:\n"
            << "  one class per line, e.g.\n"
            << "  [0, 2, 4]\n"
            << "  [1, 3]\n\n";
        return 1;
    }

    try {
        std::string graph_path = argv[1];
        int k = std::stoi(argv[2]);

        if (k < 1) {
            throw std::runtime_error("k must be at least 1.");
        }

        std::string partition_path;
        std::string output_path;
        bool directed = false;

        for (int i = 3; i < argc; ++i) {
            std::string arg = argv[i];

            if (arg == "--directed") {
                directed = true;
            } else if (partition_path.empty()) {
                partition_path = arg;
            } else if (output_path.empty()) {
                output_path = arg;
            } else {
                throw std::runtime_error("Too many arguments.");
            }
        }

        GraphInput G = read_graph(graph_path);
        auto partition = read_partition(partition_path, G.n);

        std::cerr << "n = " << G.n << "\n";
        std::cerr << "m = " << G.edges.size() << "\n";
        std::cerr << "k = " << k << "\n";
        std::cerr << "initial partition classes = " << partition.size() << "\n";

        // Warning: original implementation materializes all n^k tuples.
        // So k=2 is okay for many graphs; k=3 can explode quickly.
        auto nodes = build_graph_nodes(G.n, G.edges, partition, directed);

        bool has_vertex_labels = !partition.empty();

        TupleMap coloring = wl::k_WL(nodes, k, has_vertex_labels);

        int num_tuple_colors = count_tuple_colors(coloring);

        std::cerr << "k-tuple colors = " << num_tuple_colors << "\n";
        std::cerr << "k-tuples total = " << coloring.size() << "\n";

        auto vertex_classes = extract_vertex_classes(coloring, k);

        write_classes(vertex_classes, output_path);

        std::cerr << "refined vertex classes = " << vertex_classes.size() << "\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}