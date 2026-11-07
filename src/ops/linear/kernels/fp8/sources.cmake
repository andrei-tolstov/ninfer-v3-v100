target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/fp8_format.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/fp8_a8.cu"
)

if(NINFER_TARGET_VOLTA)
  target_sources(ninfer_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/fp8_cutlass_sm70.cu"
    "${CMAKE_CURRENT_LIST_DIR}/fp8_prepack_sm70.cu"
    "${CMAKE_CURRENT_LIST_DIR}/fp8_volta_qpn_gemm.cu"
  )
endif()
