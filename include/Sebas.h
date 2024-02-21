#pragma once

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <type_traits>
#include <iostream>
#include <stdexcept>
#include "config.h"
#include "utils.h"

__global__ void initCurandStates(curandState_t *state, int n, unsigned long long seed)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        curand_init(seed, id, 0, &state[id]);
    }
}

template <typename T>
__global__ void RVDKernel(T *matrix, int row, int col)
{
    // 计算行号和列号
    int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
    int colIdx = blockIdx.x * blockDim.x + threadIdx.x;

    // 确保线程操作的索引在合理范围内
    if (rowIdx < row && colIdx < col)
    {
        // 左下矩阵的索引位置。因为矩阵是列优先，所以要用colIdx计算列的偏移。
        int indexLeftBottom = colIdx * (2 * row) + rowIdx + row;

        // 右上矩阵相应的位置需要将矩阵的列偏移量加上col来实现对矩阵右半部的访问
        int indexRightTop = (colIdx + col) * (2 * row) + rowIdx;

        // 右下矩阵位置
        int indexRightBottom = (colIdx + col) * (2 * row) + rowIdx + row;

        // 左上矩阵到右下矩阵的复制
        matrix[indexRightBottom] = matrix[colIdx * (2 * row) + rowIdx];

        // 左下矩阵取相反数填充到右上矩阵
        matrix[indexRightTop] = -matrix[indexLeftBottom];
    }
}

#define TRY_CATCH_CALL(call)                            \
    try                                                 \
    {                                                   \
        call;                                           \
    }                                                   \
    catch (const std::runtime_error &e)                 \
    {                                                   \
        std::cerr << "Exception caught in destructor: " \
                  << e.what() << std::endl;             \
    }

template <typename T>
class Sebas
{
private:
    cudaStream_t stream;         // CUDA流
    curandGenerator_t gen;       // cuRAND生成器
    curandState_t *d_states;     // cuRAND状态 用于curand_kernel
    cublasHandle_t cublasHandle; // cuBLAS句柄

    // 私有化构造函数
    Sebas()
    {
        static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value,
                      "Sebas only supports float or double as template parameters.");
        // 创建CUDA流
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        // 创建cuRAND生成器
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW);
        curandSetStream(gen, stream);
        // 创建cuBLAS句柄
        cublasCreate(&cublasHandle);
        cublasSetStream(cublasHandle, stream);

        // 初始化cuRAND状态数组
        CUDA_CHECK(cudaMalloc((void **)&d_states, MAX_CURAND_STATE * sizeof(curandState_t)));
        initCurandStates<<<256, 256>>>(d_states, MAX_CURAND_STATE, time(NULL));
        cudaDeviceSynchronize();
    }

    // 私有化析构函数
    ~Sebas()
    {
        // 销毁CUDA流
        cudaStreamDestroy(stream);
        // 销毁cuBLAS句柄
        cublasDestroy(cublasHandle);
        // 销毁cuRAND生成器
        curandDestroyGenerator(gen);
    }

    // 私有化拷贝构造函数和赋值运算符
    Sebas(const Sebas &) = delete;
    Sebas &operator=(const Sebas &) = delete;

public:
    // 获取类的唯一实例
    static Sebas &getInstance()
    {
        static Sebas instance;
        return instance;
    }

    // 获取CUDA流
    inline cudaStream_t getStream() const
    {
        return stream;
    }

    // 获取cuRAND生成器
    inline curandGenerator_t getCurandGenerator() const
    {
        return gen;
    }

    // 获取cuBLAS句柄
    inline cublasHandle_t getCublasHandle() const
    {
        return cublasHandle;
    }

    void normalDistribution(T *A, int n, T mean, T stddev)
    {
        if constexpr (std::is_same<T, float>::value)
        {
            CURAND_CHECK(curandGenerateNormal(gen, A, n, mean, stddev));
        }
        else if constexpr (std::is_same<T, double>::value)
        {
            CURAND_CHECK(curandGenerateNormalDouble(gen, A, n, mean, stddev));
        }
    }
    void uniformIntDistribution(unsigned int *A, int n, unsigned int low, unsigned int high)
    {
        CURAND_CHECK(curandGenerate(gen, A, n));

        thrust::transform(thrust::device, A, A + n, A,
                          [=] __device__(int val)
                          { return val % (high - low + 1) + low; });
    }

    // used in real domain detection channel generation
    void RVD(T *A, int row, int col)
    {
        if constexpr (std::is_same<T, float>::value)
        {
            CURAND_CHECK(curandGenerateNormal(gen, A, 2 * row * col, 0, std::sqrt(0.5)));
        }
        else if constexpr (std::is_same<T, double>::value)
        {
            CURAND_CHECK(curandGenerateNormalDouble(gen, A, 2 * row * col, 0, std::sqrt(0.5)));
        }
        static dim3 threadsPerBlock(32, 32);
        static dim3 numBlocks((col + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (row + threadsPerBlock.y - 1) / threadsPerBlock.y);

        RVDKernel<<<numBlocks, threadsPerBlock>>>(A, row, col);
    }
};
