target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ops/linear/kernels/q8/w8_volta_mma_gemm.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ops/linear/kernels/q8/w8_volta_qpn_gemm.cu"
)
