# Compile policy is owned here; each semantic family explicitly contributes sources.
add_library(ninfer_ops SHARED)
ninfer_internal_includes(ninfer_ops)
ninfer_cuda_library(ninfer_ops)
target_link_libraries(ninfer_ops PUBLIC ninfer_core PRIVATE ninfer_ops_non_rdc)

# Non-RDC kernels keep instruction and register-allocation choices within one translation unit.
# This private PIC archive joins the shared library through host launchers.
add_library(ninfer_ops_non_rdc STATIC)
ninfer_internal_includes(ninfer_ops_non_rdc)
ninfer_cuda_non_rdc_archive(ninfer_ops_non_rdc)
target_link_libraries(ninfer_ops_non_rdc PRIVATE ninfer_core CUDA::cudart CUDA::cuda_driver)

if(NINFER_TARGET_VOLTA)
  target_link_libraries(ninfer_ops PRIVATE nvidia::cutlass::cutlass)
  target_link_libraries(ninfer_ops_non_rdc PRIVATE nvidia::cutlass::cutlass)
endif()

target_sources(ninfer_ops PRIVATE weight_input.cpp)
include("${CMAKE_CURRENT_LIST_DIR}/collective/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/common/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/elementwise/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/indexing/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/embedding/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/reduction/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/sampling/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/sequence/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/dynamic_grouped_conv/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/gated_residual/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/mhc/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/kv_cache/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_attention/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/ple/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/normalization/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/rope/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/softmax_attention/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/moe/sources.cmake")

if(NINFER_TARGET_VOLTA)
  target_sources(ninfer_ops PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ops/sparse_moe/prefill/sparse_moe_prefill_plan.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ops/sparse_moe/decode/sparse_moe_decode_kernels.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ops/sparse_moe/decode/sparse_moe_decode_plan.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ops/sparse_moe/small_t/sparse_moe_small_t_kernels.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ops/sparse_moe/small_t/sparse_moe_small_t_plan.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ops/wrapper/sparse_moe.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ops/wrapper/candidate_selector.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ops/wrapper/mtp_round.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ops/candidate_selector/bf16/dflash2_selector_volta.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ops/dynamic_grouped_conv/volta/dflash2_dynamic_conv_volta.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ops/launcher/mtp_round.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ops/launcher/rmsnorm_pack_tail_volta.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ops/dflash2_sm70_stub.cu"
  )
endif()
