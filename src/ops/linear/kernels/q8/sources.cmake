target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/q8_sliced_k_mma.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8_mma.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8_simt.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8_gemv.cu"
)

if(NINFER_TARGET_VOLTA)
  target_sources(ninfer_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/w8_volta_mma_gemm.cu"
    "${CMAKE_CURRENT_LIST_DIR}/w8_volta_qpn_gemm.cu"
  )
endif()
