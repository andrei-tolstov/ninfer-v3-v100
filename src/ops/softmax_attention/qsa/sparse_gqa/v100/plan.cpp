#include <stdexcept>
namespace ninfer::ops::detail::qsa {
void select_sparse_gqa_plan() {
    throw std::invalid_argument("qsa sparse_gqa: not supported on Volta (sm_70)");
}
} // namespace ninfer::ops::detail::qsa
