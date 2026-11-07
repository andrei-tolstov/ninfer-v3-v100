#include "ops/linear_attention/gated_delta_net/replay/launch.h"
#include <stdexcept>

namespace ninfer::ops::detail::gdn {
void launch_record(Operands, CudaExecutionView) {
    throw std::invalid_argument("gdn record launch: not supported on Volta (sm_70)");
}
} // namespace ninfer::ops::detail::gdn
