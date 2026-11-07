target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ops/linear_attention/gated_delta_net/input_conv/fp8_gdn_input_cutlass_sm70.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ops/linear_attention/gated_delta_net/input_conv/fp8_gdn_input_volta_qpn.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ops/linear_attention/gated_delta_net/input_conv/q4_q5_gdn_input_cutlass_sm70.cu"
)
