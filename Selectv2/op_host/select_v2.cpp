
#include "select_v2_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"


namespace optiling {
const uint32_t BLOCK_SIZE = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    SelectV2TilingData tiling;
    uint32_t y_dim[4] = {0};
    const gert::StorageShape* y_shape = context->GetOutputShape(0);       
    for (int j = 0; j < y_shape->GetStorageShape().GetDimNum(); j++) {  
        y_dim[j] = y_shape->GetStorageShape().GetDim(j);                   
    } 
    tiling.set_y_shape(y_dim);

    uint32_t sizeofdatatype;
    int32_t NUM = 24;

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto socVersion = ascendcPlatform.GetSocVersion();
    uint64_t ub_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size); 
    auto aivNum = ascendcPlatform.GetCoreNum();
    uint32_t input_num=3;
    //用于存维度数
    uint32_t shapeInf[3 * 10] = {};
    uint32_t inputLength[3] = {};
    uint32_t length = 0;  
    // 获取最大维度数
    for (int i = 0; i < input_num; ++i) 
        length = std::max<uint32_t>(length, context->GetInputShape(i)->GetStorageShape().GetDimNum());
    // 获取输入形状，维度数，以及输入数据大小
    for (int i = 0; i < input_num; ++i) { 
        const gert::StorageShape* shape = context->GetInputShape(i);
        inputLength[i] = context->GetInputTensor(i)->GetShapeSize();        
        shapeInf[i*4+0]=shape->GetStorageShape().GetDimNum();             
        for (int j = 1; j <= shape->GetStorageShape().GetDimNum(); j++) {  
            shapeInf[i*4+j] = shape->GetStorageShape().GetDim(j-1);                   
        } 
    }  
    // 获取最大元素值
    uint32_t total_length = 0;
    for (int i = 0; i < input_num; ++i) {  
        total_length = std::max<uint32_t>(total_length, context->GetInputTensor(i)->GetShapeSize());
    }    
    //判断是否需要广播
    bool boardCast = 0;    
    if (inputLength[0] != total_length ||       
        inputLength[1] != total_length ||
        inputLength[2] != total_length) {    
        boardCast = 1;  
    }  
    tiling.set_boardCast(boardCast);
    //用于存数据元素个数
    uint32_t totalLength = total_length;
    
    // 获取第一个输入的数据类型。
    auto inputx1 = context->GetInputTensor(1)->GetDataType();
    uint32_t tmp_x = 2;
    // 开始条件判断：根据数据类型设置不同的参数。
    // 如果数据类型是 INT8 (ge::DT_INT8)。
    if(inputx1 == ge::DT_INT8){
        // 设置数据类型大小为 1 字节。
        sizeofdatatype = 1;
        // 将经验参数 NUM 调整为 15。
        NUM = 13;
    // 如果数据类型是 FP16 (ge::DT_FLOAT16)。
    }else if(inputx1 == ge::DT_FLOAT16){
        // 设置数据类型大小为 2 字节。
        sizeofdatatype = 2;
        // 将经验参数 NUM 调整为 9。
        NUM = 7;
    // 如果数据类型是 INT32 (ge::DT_INT32)。
    } else if (inputx1 == ge::DT_INT32) {
        // 设置数据类型大小为 4 字节。
        sizeofdatatype = 4;
        // 将经验参数 NUM 调整为 11 (9+2)。
        NUM = 8+2;
        // 将调整因子 tmp_x 设为 1。
        tmp_x = 1;
    // 对于其他数据类型，如 FP32。
    } else if (inputx1 == ge::DT_FLOAT) {
        // 默认设置数据类型大小为 2 字节。
        sizeofdatatype = 2;
        // 默认将经验参数 NUM 调整为 9。
        NUM = 8;
    } else { 
        // 默认设置数据类型大小为 2 字节。
        sizeofdatatype = 2;
        // 默认将经验参数 NUM 调整为 9。
        NUM = 7;
    }

    // 计算 ALIGN_NUM：一个 BLOCK_SIZE (32 字节) 可以容纳多少个当前数据类型的元素。
    // 这是数据对齐的基本单位（元素个数）。
    uint8_t ALIGN_NUM = BLOCK_SIZE / sizeofdatatype;
    // 计算 tiling_size：这是 Tiling 的核心参数之一。
    // 它估算在 UB 内存限制下，单次处理（或单核单次迭代）可以处理多少个 BLOCK_SIZE 块。
    // (总 UB 大小 / 每个块大小 / 调整因子 tmp_x) / 经验参数 NUM
    uint32_t tiling_size = ((ub_size) / BLOCK_SIZE / tmp_x) / NUM;
    // 调整 tiling_size：如果 tiling_size 大于 8，则将其向下取整到最近的 8 的倍数。
    // 这通常是为了匹配硬件的向量处理能力（例如，一次处理 8 个向量）。
    tiling_size = tiling_size <= 8 ? tiling_size : tiling_size / 8 * 8;
    // 计算 block_size：将 tiling_size（块的数量）乘以 ALIGN_NUM（每块的元素数），
    // 得到核函数内部处理的基本数据块的大小（单位：元素数量）。
    uint32_t block_size = tiling_size * ALIGN_NUM;
    // 调整要使用的 AI Core 数量 aivNum：
    // 取“物理可用核心数”和“总工作量 / 每个块的大小 所需的块数”中的较小值。
    // 防止分配的核心数超过实际工作所需的数量。
    aivNum = (aivNum < totalLength / block_size) ? aivNum : (totalLength / block_size);
    aivNum = aivNum >= 1 ? aivNum : 1;

    // 计算 core_size：每个 AI Core 负责处理的主要（对齐）部分的数据量（单位：元素数量）。
    // (总元素数 / 使用的核心数) 先得到每个核心大致处理量，
    // 然后除以 (ALIGN_NUM * 8) 再乘以 (ALIGN_NUM * 8)，实现向下对齐到 ALIGN_NUM * 8 的倍数。
    // ALIGN_NUM * 8 可能是一个更优的处理粒度。
    uint32_t core_size = (totalLength / aivNum) / (ALIGN_NUM * 8) * (ALIGN_NUM * 8);
    // 计算 core_remain：总元素数量减去所有核心处理的对齐部分的总和，
    // 得到剩余的、未能被对齐处理的元素数量。这些通常需要特殊处理。
    uint32_t core_remain = totalLength - aivNum * core_size;

    // 将计算出的 ALIGN_NUM 保存到 tiling 对象中。
    tiling.set_ALIGN_NUM(ALIGN_NUM);

    // 将计算出的核内处理块大小 block_size 保存到 tiling 对象中。
    tiling.set_block_size(block_size);

    // 将计算出的每个核心主要处理量 core_size 保存到 tiling 对象中。
    tiling.set_core_size(core_size);

    // 将计算出的剩余元素数量 core_remain 保存到 tiling 对象中。
    tiling.set_core_remain(core_remain);

    // 将之前获取并存储的输入形状信息 shapeInf 保存到 tiling 对象中。
    tiling.set_shapeInf(shapeInf);

    context->SetBlockDim(aivNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    // 获取用于设置 Workspace（工作空间，临时内存）大小的数组指针。参数 1 表示需要 1 个 Workspace 区域。
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);

    // 设置第一个（也是唯一一个）Workspace 的大小为 0。
    // 表示根据当前的 Tiling 计算，该算子执行不需要额外的临时内存。
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}
} 


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    // const gert::Shape* x2_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *context->GetOutputShape(0);
    return GRAPH_SUCCESS;
}
// static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
// {
// const auto inputDataType = context->GetInputDataType(1);
// context->SetOutputDataType(0, inputDataType);
// return ge::GRAPH_SUCCESS;
// }
}


namespace ops {
class SelectV2 : public OpDef {
public:
    explicit SelectV2(const char* name) : OpDef(name)
    {
        this->Input("condition")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        // this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(SelectV2);
}
