
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SelectV2TilingData)
  TILING_DATA_FIELD_DEF(uint32_t, block_size);
  TILING_DATA_FIELD_DEF(uint32_t, core_size);   
  TILING_DATA_FIELD_DEF(uint32_t, core_remain);   
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 30, shapeInf);      
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 4, y_shape);  
  TILING_DATA_FIELD_DEF(uint8_t, ALIGN_NUM); 
  TILING_DATA_FIELD_DEF(bool, boardCast); 
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SelectV2, SelectV2TilingData)
}
