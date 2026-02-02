#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>
#include "utils/exception.h"

torch::Tensor triangulate(size_t num_points, const float3 *points);