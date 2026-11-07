target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/host/operands.cpp"
)

if(NINFER_TARGET_VOLTA)
  target_sources(ninfer_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/dflash2_rmsnorm_rope_volta.cu"
  )
endif()
