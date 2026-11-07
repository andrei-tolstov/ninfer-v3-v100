#include <stdexcept>
namespace ninfer::ops::detail::qsa {
void launch_sparse_gqa() {
    throw std::invalid_argument("qsa sparse_gqa launch: not supported on Volta (sm_70)");
}
} // namespace ninfer::ops::detail::qsa
