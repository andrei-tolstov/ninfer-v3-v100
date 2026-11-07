#include <stdexcept>
namespace ninfer::ops::detail::qsa {
void launch_indexer() {
    throw std::invalid_argument("qsa indexer launch: not supported on Volta (sm_70)");
}
} // namespace ninfer::ops::detail::qsa
