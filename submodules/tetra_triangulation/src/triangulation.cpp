#include "triangulation.h"

#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/compute_average_spacing.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <list>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <limits>

#include "utils/exception.h"

torch::Tensor triangulate(size_t num_points, const float3 *points)
{
    using K = CGAL::Exact_predicates_inexact_constructions_kernel;
    using Vb = CGAL::Triangulation_vertex_base_with_info_3<unsigned int, K>;
    using Tds = CGAL::Triangulation_data_structure_3<Vb>;
    using Triangulation = CGAL::Delaunay_triangulation_3<K, Tds>;
    using PWI = std::pair<Triangulation::Point, unsigned int>;
    std::vector<PWI> L;
    L.reserve(num_points);
    for (std::size_t i = 0; i < num_points; ++i)
    {
        const auto &p3 = points[i];
        L.emplace_back(Triangulation::Point(p3.x, p3.y, p3.z), static_cast<unsigned int>(i));
    }
    using Traits = CGAL::Spatial_sort_traits_adapter_3<K, CGAL::First_of_pair_property_map<PWI>>;
    CGAL::spatial_sort(L.begin(), L.end(), Traits());
#ifndef HIERARACHY
    Triangulation T(L.begin(), L.end());
#else

    Triangulation T;

    Triangulation::Cell_handle hint;
    for (const auto &pi : L)
    {
        auto vh = T.insert(pi.first, hint);
        vh->info() = pi.second;
        hint = vh->cell();
    }
#endif

    auto cells = torch::empty({T.number_of_finite_cells(), 4}, torch::dtype(torch::kInt32).device(torch::kCPU));
    int32_t *out = cells.data_ptr<int32_t>();

    int64_t i = 0;
    for (auto cell : T.finite_cell_handles())
    {
        out[4 * i + 0] = static_cast<int32_t>(cell->vertex(0)->info());
        out[4 * i + 1] = static_cast<int32_t>(cell->vertex(1)->info());
        out[4 * i + 2] = static_cast<int32_t>(cell->vertex(2)->info());
        out[4 * i + 3] = static_cast<int32_t>(cell->vertex(3)->info());
        ++i;
    }
    return cells;
}