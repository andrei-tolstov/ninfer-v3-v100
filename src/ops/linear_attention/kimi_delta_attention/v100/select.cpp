#include <stdexcept>

namespace ninfer::ops::detail::kda {
struct Plan {};
Plan select(int, int) {
    throw std::invalid_argument("kimi_delta_attention: not supported on Volta (sm_70)");
}
} // namespace ninfer::ops::detail::kda
