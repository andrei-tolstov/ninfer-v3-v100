target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ops/linear/kernels/bf16/bf16_cutlass_sm70.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ops/linear/kernels/bf16/bf16_n256_k5120_volta.cu"
)
