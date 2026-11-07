target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/bias.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/linear_split.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/operands.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/bf16/shapes/n14336_k5120/a16.cu"
  "${CMAKE_CURRENT_LIST_DIR}/bf16/shapes/n16384_k5120/a16.cu"
  "${CMAKE_CURRENT_LIST_DIR}/bf16/shapes/n48_k2560/a16.cu"
  "${CMAKE_CURRENT_LIST_DIR}/bf16/shapes/n96_k2560/a16.cu"
  "${CMAKE_CURRENT_LIST_DIR}/bf16/shapes/n1664_k2560/a16.cu"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/shapes/n14336_k5120/a16.cu"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/shapes/n14336_k5120/a8.cu"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/shapes/n16384_k5120/a16.cu"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/shapes/n16384_k5120/a8.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/shapes/n14336_k5120/a16.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/shapes/n14336_k5120/a4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/shapes/n16384_k5120/a16.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/shapes/n16384_k5120/a4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q4_q5/shapes/n14336_k5120/a16.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q4_q5/shapes/n16384_k5120/a16.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/shapes/n14336_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/shapes/n16384_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/shapes/n48_k2560.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/shapes/n96_k2560.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/shapes/n1664_k2560.cu"
)

target_sources(ninfer_ops_non_rdc PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/shapes/n14336_k5120/a4_tma.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/shapes/n16384_k5120/a4_tma.cu"
)

include("${CMAKE_CURRENT_LIST_DIR}/bf16/${NINFER_TARGET}/sources.cmake")

include("${CMAKE_CURRENT_LIST_DIR}/q8/${NINFER_TARGET}/sources.cmake")

if(NINFER_TARGET STREQUAL "dgx_spark")
  include("${CMAKE_CURRENT_LIST_DIR}/mxfp8/dgx_spark/sources.cmake")
  include("${CMAKE_CURRENT_LIST_DIR}/fp8_block_bf16/dgx_spark/sources.cmake")
endif()

if(NINFER_TARGET_VOLTA)
  target_sources(ninfer_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/fp8_attn_input_cutlass_sm70.cu"
    "${CMAKE_CURRENT_LIST_DIR}/fp8_attn_input_volta_qpn.cu"
    "${CMAKE_CURRENT_LIST_DIR}/q4_q5_attn_input_cutlass_sm70.cu"
  )
endif()
