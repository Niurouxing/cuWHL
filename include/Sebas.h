#pragma once

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_vector.h>
#include <type_traits>
#include <iostream>
#include <stdexcept>
#include "config.h"
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

        // 左上矩阵的索引位置
        int indexLeftTop = colIdx * (2 * row) + rowIdx;

        // 左上矩阵到右下矩阵的复制
        matrix[indexRightBottom] = matrix[indexLeftTop];
        // matrix[indexRightBottom] = 0;

        // 左下矩阵取相反数填充到右上矩阵
        matrix[indexRightTop] = -matrix[indexLeftBottom];
        // matrix[indexRightTop] = 0;
    }
}

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
        initialize();
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
        // 释放cuRAND状态数组
        cudaFree(d_states);
    }

    // 私有化拷贝构造函数和赋值运算符
    Sebas(const Sebas &) = delete;
    Sebas &operator=(const Sebas &) = delete;

    __device__ inline float normal(curandState_t *state, float mean, float stddev)
    {
        return curand_normal(state) * stddev + mean;
    }

    __device__ inline double normal(curandState_t *state, double mean, double stddev)
    {
        return curand_normal_double(state) * stddev + mean;
    }

    inline void generateNormal(curandGenerator_t gen, float *A, int n, float mean, float stddev)
    {
        CURAND_CHECK(curandGenerateNormal(gen, A, n, mean, stddev));
    }

    inline void generateNormal(curandGenerator_t gen, double *A, int n, double mean, double stddev)
    {
        CURAND_CHECK(curandGenerateNormalDouble(gen, A, n, mean, stddev));
    }

    inline void gemv(float *A, int row, int cow, bool transpose, float alpha, float *x, float beta, float *y)
    {

        const float ALPHA = alpha;
        const float BETA = beta;
        cublasSgemv(cublasHandle, transpose ? CUBLAS_OP_T : CUBLAS_OP_N, row, cow, &ALPHA, A, row, x, 1, &BETA, y, 1);
    }

    inline void gemv(double *A, int row, int cow, bool transpose, double alpha, double *x, double beta, double *y)
    {

        const double ALPHA = alpha;
        const double BETA = beta;
        cublasDgemv(cublasHandle, transpose ? CUBLAS_OP_T : CUBLAS_OP_N, row, cow, &ALPHA, A, row, x, 1, &BETA, y, 1);
    }

    inline void gemv(cuComplex *A, int row, int cow, bool transpose, cuComplex alpha, cuComplex *x, cuComplex beta, cuComplex *y)
    {

        const cuComplex ALPHA = alpha;
        const cuComplex BETA = beta;
        cublasCgemv(cublasHandle, transpose ? CUBLAS_OP_T : CUBLAS_OP_N, row, cow, &ALPHA, A, row, x, 1, &BETA, y, 1);
    }

    inline void gemv(cuDoubleComplex *A, int row, int cow, bool transpose, cuDoubleComplex alpha, cuDoubleComplex *x, cuDoubleComplex beta, cuDoubleComplex *y)
    {

        const cuDoubleComplex ALPHA = alpha;
        const cuDoubleComplex BETA = beta;
        cublasZgemv(cublasHandle, transpose ? CUBLAS_OP_T : CUBLAS_OP_N, row, cow, &ALPHA, A, row, x, 1, &BETA, y, 1);
    }

 


public:
    void initialize()
    {
        // 创建CUDA流
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        // 创建cuRAND生成器 
        CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
        // 初始化cuRAND状态数组
        CUDA_CHECK(cudaMalloc((void **)&d_states, MAX_CURAND_STATE * sizeof(curandState_t)));
        unsigned long long seed = time(NULL);
        thrust::for_each(thrust::device, d_states, d_states + MAX_CURAND_STATE, [seed] __device__(curandState_t & state)
                         {
                             int id = threadIdx.x + blockIdx.x * blockDim.x;
                             curand_init(seed, id, 0, &state); });
        // 创建cuBLAS句柄
        CUBLAS_CHECK(cublasCreate(&cublasHandle));
        cudaDeviceSynchronize();
    }

    // 获取类的唯一实例
    static Sebas &getInstance()
    {
        static Sebas instance;
        return instance;
    }

    // 获取CUDA流
    inline cudaStream_t getStream()
    {
        return stream;
    }

    // 获取cuRAND生成器
    inline curandGenerator_t getCurandGenerator()
    {
        return gen;
    }

    // 获取cuBLAS句柄
    inline cublasHandle_t getCublasHandle()
    {
        return cublasHandle;
    }

    void uniformIntDistribution(unsigned int *A, int n, unsigned int low, unsigned int high)
    {
        CURAND_CHECK(curandGenerate(gen, A, n));

        thrust::transform(thrust::device, A, A + n, A,
                          [high, low] __device__(int val)
                          { return val % (high - low + 1) + low; });
    }

    // used in real domain detection channel generation
    void RVD(data_type *A, int row, int col)
    {
        static dim3 threadsPerBlock(32, 32);
        static dim3 numBlocks((col + threadsPerBlock.x - 1) / threadsPerBlock.x,
                              (row + threadsPerBlock.y - 1) / threadsPerBlock.y);

        RVDKernel<<<numBlocks, threadsPerBlock>>>(A, row, col);
    }

    template <typename T = data_type>
    void normalDistribution(T *A, int n, T mean, T stddev)
    {
        generateNormal(gen, A, n, mean, stddev);
    }

    template <typename T = data_type>
    void complexNormalDistribution(complex_type *A, int n, float mean, float stddev)
    {

        thrust::for_each(thrust::device,
                         thrust::counting_iterator<int>(0),
                         thrust::counting_iterator<int>(n),
                         [A, mean, stddev, states = this->d_states, this] __device__(int idx) mutable
                         {
                             curandState_t *state = &states[idx];
                             A[idx].x = this->normal(state, mean, stddev);
                             A[idx].y = this->normal(state, mean, stddev);
                         });
    }

    void MatrixMultiplyVector(auto *A, int row, int cow, bool transpose, auto alpha, auto *x, auto beta, auto *y)
    {
        gemv(A, row, cow, transpose, alpha, x, beta, y);
    }
};
