#include <stdexcept>
namespace ninfer::ops::detail::qsa {
void select_indexer_plan() {
    throw std::invalid_argument("qsa indexer: not supported on Volta (sm_70)");
}
} // namespace ninfer::ops::detail::qsa
