#pragma once

#include "ops/common/cuda/reduce.cuh"

namespace ninfer::ops {

using common::kWarpSize;
using common::kFullWarpMask;
using common::warp_sum;
using common::warp_reduce_sum;
using common::warp_max;
using common::block_reduce_sum;

} // namespace ninfer::ops
