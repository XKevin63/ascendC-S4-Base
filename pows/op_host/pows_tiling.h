
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(PowsTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, block_size);
    TILING_DATA_FIELD_DEF(uint32_t, core_size);   
    TILING_DATA_FIELD_DEF(uint32_t, core_remain);   
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 20, shapeInf);       
    TILING_DATA_FIELD_DEF(uint8_t, ALIGN_NUM); 
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Pows, PowsTilingData)
}
