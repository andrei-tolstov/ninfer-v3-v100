target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/q4_a16_mma.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q4_a16_simt.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q4_a16_sliced_k_mma.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q4_a16_gemv.cu"
)

if(NINFER_TARGET_VOLTA)
  target_sources(ninfer_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/q4_volta_mma_gemm.cu"
    "${CMAKE_CURRENT_LIST_DIR}/q4_volta_qpn_gemm.cu"
  )
endif()
