#include "ops/softmax_attention/dense/packed/launch.h"
#include <stdexcept>

namespace ninfer::ops::detail {

void launch_packed_attention_uniform(const Tensor&, const Tensor&, const Tensor&,
                                     std::int32_t, Tensor&, cudaStream_t) {
    throw std::invalid_argument("packed_softmax_attention uniform: not supported on Volta");
}

} // namespace ninfer::ops::detail
