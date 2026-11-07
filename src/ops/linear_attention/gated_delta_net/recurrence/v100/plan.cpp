#include "ops/linear_attention/gated_delta_net/recurrence/plan.h"
#include <stdexcept>

namespace ninfer::ops::detail::gdn {
Plan select_recurrence_plan(int, int) {
    throw std::invalid_argument("gdn recurrence: not supported on Volta (sm_70)");
}
} // namespace ninfer::ops::detail::gdn
