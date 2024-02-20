#pragma once

#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

#include <iostream>
#include <stdexcept>

#include "utils.h"

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

template<typename T>
class Sebas
{
private:
    cudaStream_t stream;         // CUDA流
    curandGenerator_t gen;       // cuRAND生成器
    cublasHandle_t cublasHandle; // cuBLAS句柄
public:
    // 构造函数
    Sebas()
    {
        // 创建CUDA流
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        // 创建cuRAND生成器
        CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW));
        CURAND_CHECK(curandSetStream(gen, stream));
        // 创建cuBLAS句柄
        CUBLAS_CHECK(cublasCreate(&cublasHandle));
        CUBLAS_CHECK(cublasSetStream(cublasHandle, stream));
    }

    // 析构函数
    ~Sebas()
    {
        // 销毁CUDA流
        TRY_CATCH_CALL(CUDA_CHECK(cudaStreamDestroy(stream)));

        // 销毁cuBLAS句柄
        TRY_CATCH_CALL(CUBLAS_CHECK(cublasDestroy(cublasHandle)));

        // 销毁cuRAND生成器
        TRY_CATCH_CALL(CURAND_CHECK(curandDestroyGenerator(gen)));
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



    void normalDistribution(T *A, int n, T mean, T stddev);
    void uniformIntDistribution(unsigned int *A, int n, unsigned int low, unsigned int high);
 
};

template<>
inline void Sebas<float>::normalDistribution(float *A, int n, float mean, float stddev)
{
    CURAND_CHECK(curandGenerateNormal(gen, A, n, mean, stddev));
}

template<>
inline void Sebas<double>::normalDistribution(double *A, int n, double mean, double stddev)
{
    CURAND_CHECK(curandGenerateNormalDouble(gen, A, n, mean, stddev));
}

template<typename T>
void Sebas<T>::uniformIntDistribution(unsigned int *A, int n, unsigned int low, unsigned int high)
{
    CURAND_CHECK(curandGenerate(gen, A, n));

    thrust::transform(thrust::device, A, A + n, A,
                      [=] __device__(int val) { return val % (high - low + 1) + low; });
}

 