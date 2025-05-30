#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;     // 使用双缓冲 (每个队列有 2 个 Buffer)

template<typename TYPE_X1, typename TYPE_X2, typename TYPE_Y> class Kernel_Powsx {
public:
    __aicore__ inline Kernel_Powsx() {}

    // ALIGN_NUM：一个 BLOCK_SIZE (32 字节) 可以容纳多少个当前数据类型的元素。
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                uint8_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain)
    {
        this->blockLength = core_size + core_remain;
        this->tileLength = block_size;
        // 将blocklength长度对齐为32字节
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0); 

        x1Gm.SetGlobalBuffer((__gm__ TYPE_X1*)x1, this->blockLength);
        x2Gm.SetGlobalBuffer((__gm__ TYPE_X2*)x2, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y, this->blockLength);

        // 计算需要处理的 Tile 数量 (tileNum)。如果 blockLength 不能被 tileLength 整除，则加 1 处理剩余部分。
        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);

        pipe.InitBuffer(inQueueX1, BUFFER_NUM, this->tileLength * sizeof(TYPE_X1));
        pipe.InitBuffer(inQueueX2, BUFFER_NUM, this->tileLength * sizeof(TYPE_X2));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(TYPE_Y));

        if constexpr (std::is_same_v<TYPE_Y, float16_t>) {
            pipe.InitBuffer(B_x1, this->tileLength * sizeof(float32_t));
            pipe.InitBuffer(B_x2, this->tileLength * sizeof(float32_t));
            pipe.InitBuffer(B_y, this->tileLength * sizeof(float32_t));
        }
        else if constexpr (std::is_same_v<TYPE_Y, float32_t>) {
            pipe.InitBuffer(B_x1, this->tileLength * sizeof(float32_t));
        }
        else if constexpr (std::is_same_v<TYPE_Y, bfloat16_t>) {
            pipe.InitBuffer(B_x1, this->tileLength * sizeof(float));
            pipe.InitBuffer(B_x2, this->tileLength * sizeof(float));
            pipe.InitBuffer(B_y, this->tileLength * sizeof(float));
            pipe.InitBuffer(f_x1_16, this->tileLength * sizeof(float));
            pipe.InitBuffer(f_x2_16, this->tileLength * sizeof(float));
            pipe.InitBuffer(f_y_16, this->tileLength * sizeof(float));
        }


    }

    // 核心处理函数：实现标准的三级双缓冲流水线 (CopyIn -> Compute -> CopyOut)
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        // 循环处理除了最后一个 Tile 之外的所有完整 Tile。
        for (int32_t i = 0; i < loopCount-1; i++) {
            // AscendC::printf("++++++++++++++++++++++++++++++this is times:[%d/%d] loop+++++++++++++++++++++++++++++++\n", i, loopCount-2);
            CopyIn(i, this->tileLength);
            Compute(i, this->tileLength);
            CopyOut(i, this->tileLength);
        }

        uint32_t length = this->blockLength - this->tileLength * (loopCount - 1);
        // AscendC::printf("++++++++++++++++++++++++++++++this is times:[%d/%d] loop+++++++++++++++++++++++++++++++\n", loopCount-1, loopCount-1);
        CopyIn(loopCount - 1, (length + 31) / 32 * 32);
        Compute(loopCount - 1, (length + 31) / 32 * 32);
        // 拷贝最后一个 Tile 的计算结果回 GM。
        // 注意：拷贝长度向上对齐到 32 的倍数，因为 DataCopy 输出通常要求对齐。
        // 即使计算只产生了 length 个有效结果，也会拷贝对齐后的长度，多余部分是无效数据，
        // 但因为 GM 空间是按对齐后的 blockLength 分配的，所以写这些无效数据是安全的。
        CopyOut(loopCount - 1, (length + 31) / 32 * 32);
    }

private:
    // 搬入函数 (GM -> UB)
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length)
    {
        AscendC::LocalTensor<TYPE_X1> x1Local = inQueueX1.AllocTensor<TYPE_X1>();
        AscendC::LocalTensor<TYPE_X2> x2Local = inQueueX2.AllocTensor<TYPE_X2>();

        AscendC::DataCopy(x1Local, x1Gm[progress * this->tileLength], length);
        AscendC::DataCopy(x2Local, x2Gm[progress * this->tileLength], length);
        
        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
    }

    __aicore__ inline void Compute(int32_t progress, uint32_t length) 
    {
        AscendC::LocalTensor<TYPE_X1> x1Local = inQueueX1.DeQue<TYPE_X1>();
        AscendC::LocalTensor<TYPE_X2> x2Local = inQueueX2.DeQue<TYPE_X2>();
        AscendC::LocalTensor<TYPE_Y> yLocal = outQueueY.AllocTensor<TYPE_Y>();

        if constexpr (std::is_same_v<TYPE_Y, float32_t>) {
            // AscendC::printf("-------------------------------this is float32_t compute-------------------------------\n");
            auto ln_x1 = B_x1.Get<float32_t>();
            AscendC::Ln(ln_x1, x1Local, length);
            AscendC::Mul(ln_x1, x2Local, ln_x1, length);
            AscendC::Exp(yLocal, ln_x1, length);
        }
        else if constexpr (std::is_same_v<TYPE_Y, float16_t>) {
            // AscendC::printf("-------------------------------this is float16_t compute-------------------------------\n");
            auto tmp_x1 = B_x1.Get<float32_t>();
            auto tmp_x2 = B_x2.Get<float32_t>();
            auto tmp_y = B_y.Get<float32_t>();
            AscendC::Cast(tmp_x1, x1Local, AscendC::RoundMode::CAST_NONE, length);
            AscendC::Cast(tmp_x2, x2Local, AscendC::RoundMode::CAST_NONE, length);
            AscendC::Ln(tmp_x1, tmp_x1, length);
            AscendC::Mul(tmp_x1, tmp_x2, tmp_x1, length);
            AscendC::Exp(tmp_y, tmp_x1, length);
            AscendC::Cast(yLocal, tmp_y, AscendC::RoundMode::CAST_NONE, length);

        }
        else if constexpr (std::is_same_v<TYPE_Y, bfloat16_t>){
            // AscendC::printf("-------------------------------this is bf16_t compute-------------------------------\n");
            auto tmp_x1 = B_x1.Get<float>();
            auto tmp_x2 = B_x2.Get<float>();
            auto tmp_y = B_y.Get<float>();
            AscendC::Cast(tmp_x1, x1Local, AscendC::RoundMode::CAST_NONE, length);
            AscendC::Cast(tmp_x2, x2Local, AscendC::RoundMode::CAST_NONE, length);
            AscendC::DumpTensor(tmp_x1, 321333, length);
            AscendC::Ln(tmp_x1, tmp_x1, length);
            AscendC::Mul(tmp_x1, tmp_x1, tmp_x2, length);
            AscendC::Exp(tmp_y, tmp_x1, length);
            AscendC::Cast(yLocal, tmp_y, AscendC::RoundMode::CAST_ROUND, length);
        }

        
        outQueueY.EnQue<TYPE_Y>(yLocal);
        inQueueX1.FreeTensor(x1Local);
        inQueueX2.FreeTensor(x2Local);
    }

    // 搬出函数 (UB -> GM)
    __aicore__ inline void CopyOut(int32_t progress, uint32_t length)
    {
        AscendC::LocalTensor<TYPE_Y> yLocal = outQueueY.DeQue<TYPE_Y>();

        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, length);
        outQueueY.FreeTensor(yLocal);
    }

private:
    // 固定变量
    uint32_t blockLength, tileNum, tileLength;  

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX1;          
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX2;         
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;     

    AscendC::GlobalTensor<TYPE_X1> x1Gm; 
    AscendC::GlobalTensor<TYPE_X2> x2Gm;        
    AscendC::GlobalTensor<TYPE_Y> yGm;     
    //
    AscendC::TBuf<AscendC::QuePosition::VECCALC> B_x1, B_x2, B_y;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> f_x1_16, f_x2_16, f_y_16;
};


template<typename TYPE_X1, typename TYPE_X2, typename TYPE_Y> class KernelPows_Broadcast {
    public:
        __aicore__ inline KernelPows_Broadcast() {}
    
        __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                    uint8_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain, uint32_t shapeInf[2*10])
        {
             // 确定最大维度数
            int32_t x1DimNum = static_cast<int32_t>(shapeInf[0 * 10 + 0]);
            int32_t x2DimNum = static_cast<int32_t>(shapeInf[1 * 10 + 0]);
            maxDimNum = x1DimNum;
            if (x2DimNum > maxDimNum) maxDimNum = x2DimNum;
            // 将shapeInf转换为shape，最后一行存每个维度最大值
            for(int tensor_idx = maxDimNum-1; tensor_idx >= 0; tensor_idx--) {
                this->shapes[0][tensor_idx] = (maxDimNum - x1DimNum - tensor_idx) > 0 ? 1 : shapeInf[0*10 + tensor_idx+1];
                this->shapes[1][tensor_idx] = (maxDimNum - x2DimNum - tensor_idx) > 0 ? 1 : shapeInf[1*10 + tensor_idx+1];
                this->shapes[2][tensor_idx] = this->shapes[0][tensor_idx];
                if (this->shapes[1][tensor_idx] > this->shapes[2][tensor_idx]) this->shapes[2][tensor_idx] = this->shapes[1][tensor_idx];
            }

            for (int i = 0; i < 3; i++) {
                strides[i][maxDimNum - 1] = 1;
            }
            for (int i = maxDimNum - 2; i >= 0; i--) {
                strides[0][i] = strides[0][i + 1] * shapes[0][i + 1];
                strides[1][i] = strides[1][i + 1] * shapes[1][i + 1];
                strides[2][i] = strides[2][i + 1] * shapes[2][i + 1];
            }
            
            this->blockLength = core_size + core_remain;
            // 将blocklength长度对齐为32字节
            this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0); 
            x1Gm.SetGlobalBuffer((__gm__ TYPE_X1*)x1, this->blockLength);
            x2Gm.SetGlobalBuffer((__gm__ TYPE_X2*)x2, this->blockLength);
            yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y, this->blockLength);
            
            this->tileLength = 1;
            if constexpr (std::is_same_v<TYPE_Y, float16_t>) {
                pipe.InitBuffer(B_x1, this->tileLength * sizeof(float32_t));
                pipe.InitBuffer(B_x2, this->tileLength * sizeof(float32_t));
                pipe.InitBuffer(B_y, this->tileLength * sizeof(float32_t));
                auto tmp_x1 = B_x1.Get<float32_t>();
                auto tmp_x2 = B_x2.Get<float32_t>();
                auto tmp_y = B_y.Get<float32_t>();
            }
            else if constexpr (std::is_same_v<TYPE_Y, float32_t>) {
                pipe.InitBuffer(B_x1, this->tileLength * sizeof(float32_t));
                pipe.InitBuffer(B_x2, this->tileLength * sizeof(float32_t));
                pipe.InitBuffer(B_y, this->tileLength * sizeof(float32_t));
                auto tmp_x1 = B_x1.Get<float32_t>();
                auto tmp_x2 = B_x2.Get<float32_t>();
                auto tmp_y = B_y.Get<float32_t>();
            }
            else if constexpr (std::is_same_v<TYPE_Y, bfloat16_t>) {
                pipe.InitBuffer(B_x1, this->tileLength * sizeof(float));
                pipe.InitBuffer(B_x2, this->tileLength * sizeof(float));
                pipe.InitBuffer(B_y, this->tileLength * sizeof(bfloat16_t));
                auto tmp_x1 = B_x1.Get<float>();
                auto tmp_x2 = B_x2.Get<float>();
                auto tmp_y = B_y.Get<bfloat16_t>();
            }


        }
    
        __aicore__ inline void Process()
        {
            for (int i = 0; i < blockLength; i++) {
                int x1Offset = 0;
                int x2Offset = 0;
                int outOffset = 0;
                for (int j = 0; j < maxDimNum; j++) {
                    int index = i / strides[2][j] % shapes[2][j];
                    x1Offset += (index % shapes[0][j]) * strides[0][j];
                    x2Offset += (index % shapes[1][j]) * strides[1][j];
                    outOffset += index * strides[2][j];
                }
                TYPE_X1 x1 = x1Gm.GetValue(x1Offset);
                TYPE_X2 x2 = x2Gm.GetValue(x2Offset);

                if constexpr (std::is_same_v<TYPE_Y, float16_t>) {
                    auto tmp_x1 = B_x1.Get<float32_t>();
                    auto tmp_x2 = B_x2.Get<float32_t>();
                    auto tmp_y = B_y.Get<float32_t>();
                    tmp_x1.SetValue(0, static_cast<float32_t>(x1));
                    tmp_x2.SetValue(0, static_cast<float32_t>(x2));
                    AscendC::Ln(tmp_x1, tmp_x1, tileLength);
                    AscendC::Mul(tmp_x1, tmp_x2, tmp_x1, tileLength);
                    AscendC::Exp(tmp_y, tmp_x1, tileLength);
                    yGm.SetValue(outOffset, static_cast<float16_t>(tmp_y.GetValue(0)));
                }
                else if constexpr (std::is_same_v<TYPE_Y, float32_t>) {
                    auto tmp_x1 = B_x1.Get<float32_t>();
                    auto tmp_x2 = B_x2.Get<float32_t>();
                    auto tmp_y = B_y.Get<float32_t>();
                    tmp_x1.SetValue(0, x1);
                    tmp_x2.SetValue(0, x2);
                    AscendC::Ln(tmp_x1, tmp_x1, tileLength);
                    AscendC::Mul(tmp_x1, tmp_x2, tmp_x1, tileLength);
                    AscendC::Exp(tmp_y, tmp_x1, tileLength);
                    yGm.SetValue(outOffset, tmp_y.GetValue(0));
                }
                else if constexpr (std::is_same_v<TYPE_Y, bfloat16_t>) {
                    auto tmp_x1 = B_x1.Get<float>();
                    auto tmp_x2 = B_x2.Get<float>();
                    auto tmp_y = B_y.Get<bfloat16_t>();
                    tmp_y.SetValue(0, x1);
                    AscendC::Cast(tmp_x1, tmp_y, AscendC::RoundMode::CAST_NONE, tileLength);
                    tmp_y.SetValue(0, x2);
                    AscendC::Cast(tmp_x2, tmp_y, AscendC::RoundMode::CAST_NONE, tileLength);
                    AscendC::Ln(tmp_x1, tmp_x1, tileLength);
                    AscendC::Mul(tmp_x1, tmp_x2, tmp_x1, tileLength);
                    AscendC::Exp(tmp_x1, tmp_x1, tileLength);
                    AscendC::Cast(tmp_y, tmp_x1, AscendC::RoundMode::CAST_ROUND, tileLength);
                    yGm.SetValue(outOffset, tmp_y.GetValue(0));
                }
                
            }
        }

    
    private:
        AscendC::TPipe pipe;
        // 固定变量
        uint32_t blockLength, tileLength;  
        
        AscendC::GlobalTensor<TYPE_X1> x1Gm; 
        AscendC::GlobalTensor<TYPE_X2> x2Gm;        
        AscendC::GlobalTensor<TYPE_Y> yGm;     
              
        int64_t shapes[3][10];   
        int64_t strides[3][10];  
        int32_t maxDimNum; // 最大维度数
        AscendC::TBuf<AscendC::QuePosition::VECCALC> B_x1, B_x2, B_y;

};






extern "C" __global__ __aicore__ void pows(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    if (TILING_KEY_IS(1)) {
        Kernel_Powsx<DTYPE_X1, DTYPE_X2, DTYPE_Y> op; 
        op.Init(x1, x2, y,
            tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        KernelPows_Broadcast<DTYPE_X1, DTYPE_X2, DTYPE_Y> op; 
        op.Init(x1, x2, y,
            tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain, tiling_data.shapeInf);
        op.Process();
    }
}