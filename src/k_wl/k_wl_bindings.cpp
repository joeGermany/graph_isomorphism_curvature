#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "k_wl.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_k_wl, m) {
    m.doc() = "Standard separate-coordinate k-WL (ordinary 1-WL); vertex colors only";

    m.def("run", [](const k_wl::Matrix& adjacency, int k,
                    const std::vector<int>& vertex_colors,
                    std::size_t max_tuples) {
        std::vector<int> colors;
        {
            py::gil_scoped_release release;
            colors = k_wl::vertex_colors(adjacency, k, vertex_colors, max_tuples);
        }
        return colors;
    },
    py::arg("adjacency"),
    py::arg("k"),
    py::arg("vertex_colors") = std::vector<int>{},
    py::arg("max_tuples") = 2000000,
    R"doc(
        Return one final k-WL color per input vertex, in adjacency matrix order.
        Internally computes all ordered k-tuples, returning only diagonal colors.
        Colors are normalized to integers 0 .. (number of vertex classes - 1).
    )doc");

    // Optional: retain class count and number of refinement rounds.
    m.def("run_with_stats", [](const k_wl::Matrix& adjacency, int k,
                               const std::vector<int>& vertex_colors,
                               std::size_t max_tuples) {
        k_wl::Result result;
        {
            py::gil_scoped_release release;
            result = k_wl::run(adjacency, k, vertex_colors, max_tuples);
        }
        return py::make_tuple(result.colors, result.color_count, result.rounds);
    },
    py::arg("adjacency"),
    py::arg("k"),
    py::arg("vertex_colors") = std::vector<int>{},
    py::arg("max_tuples") = 2000000);
}
