target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_a4.cu"
)

if(NINFER_TARGET_VOLTA)
  target_sources(ninfer_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/nvfp4_cutlass_sm70.cu"
    "${CMAKE_CURRENT_LIST_DIR}/nvfp4_prepack_sm70.cu"
    "${CMAKE_CURRENT_LIST_DIR}/nvfp4_volta_mma_gemm.cu"
    "${CMAKE_CURRENT_LIST_DIR}/nvfp4_volta_qpn_gemm.cu"
  )
endif()
