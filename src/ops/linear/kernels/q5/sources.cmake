target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/q5_a16_mma.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q5_a16_sliced_k_mma.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q5_a16_simt.cu"
)

if(NINFER_TARGET_VOLTA)
  target_sources(ninfer_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/q5_volta_mma_gemm.cu"
  )
endif()
