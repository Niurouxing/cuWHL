#pragma once

#include <Cons.h>
#include <thrust/transform.h>
#include <thrust/for_each.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>

constexpr float sqrt2 = 1.4142135623730951;

template <typename T>
__global__ void RVDKernel(T *matrix, int row, int col, curandState_t *states)
{
 
    int rowIdx = threadIdx.x;
    int colIdx = blockIdx.x;

    // 确保线程操作的索引在合理范围内
    if (rowIdx < row && colIdx < col)
    {

        int idx = rowIdx + colIdx * row;

        // 产生两个随机数
        float real = curand_normal(&states[idx]) / sqrt2;
        float imag = curand_normal(&states[idx]) / sqrt2;

        // H[rowIdx][colIdx] = real
        // H[rowIdx + row][colIdx] = imag
        // H[rowIdx][colIdx + col] = -imag
        // H[rowIdx + row][colIdx + col] = real

        matrix[rowIdx + 2 * row * colIdx] = real;
        matrix[rowIdx + row + 2 * row * colIdx] = imag;
        matrix[rowIdx + 2 * row * (colIdx + col)] = -imag;
        matrix[rowIdx + row + 2 * row * (colIdx + col)] = real;
    }
}

template <int Tx, int Rx, typename qam = QAM<16, RD>>
class Detection
{
public:
    cudaStream_t stream;
    cublasHandle_t handle;
    curandGenerator_t gen;
    curandState_t *curandStates;

    float SNRdB = 40;
    float Nv = RxAntNum / (pow(10, SNRdB / 10)) / qam::bitLength * 2;

    static constexpr int TxAntNum = Tx;
    static constexpr int RxAntNum = Rx;
    static constexpr int ConSize = qam::ConSize;

    int *TxIndices;

    float *TxSymbols;
    float *RxSymbols;
    float *H;

    inline void setSNRdB(auto dB)
    {
        SNRdB = dB;

        Nv = RxAntNum / (pow(10, SNRdB / 10)) / qam::bitLength * 2;
    }

    // Constructor
    Detection()
    {
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        cublasCreate(&handle);
        cublasSetStream(handle, stream);

        curandInit();

        cudaMalloc(&TxIndices, 2 * TxAntNum * sizeof(int));
        cudaMalloc(&TxSymbols, 2 * TxAntNum * sizeof(float));
        cudaMalloc(&RxSymbols, 2 * RxAntNum * sizeof(float));
        cudaMalloc(&H, 4 * TxAntNum * RxAntNum * sizeof(float));
    }

    void curandInit()
    {
        cudaMalloc(&curandStates, TxAntNum * RxAntNum * sizeof(curandState_t));

        unsigned long long seed = 114514;


        auto* d_states = this->curandStates;
        thrust::for_each(thrust::cuda::par.on(stream),
        thrust::make_counting_iterator(0), thrust::make_counting_iterator(TxAntNum * RxAntNum),
                         [seed,  d_states] __device__(int idx)
                         {
                             curand_init(seed, idx, 0, d_states + idx);
                         });

    }

    // Destructor
    ~Detection()
    {
        cudaFree(TxIndices);
        cudaFree(TxSymbols);
        cudaFree(RxSymbols);
        cudaFree(H);
        cudaFree(curandStates);

        curandDestroyGenerator(gen);
        cublasDestroy(handle);
        cudaStreamDestroy(stream);
    }

    void generate()
    {

        // generate H
        RVDKernel<<<TxAntNum, RxAntNum, 0, stream>>>(H, RxAntNum, TxAntNum, curandStates);


        int *d_TxIndices = this->TxIndices;
        auto *d_states = this->curandStates;
        auto *d_TxSymbols = this->TxSymbols;
        // generate TxSymbols
        thrust::for_each(thrust::cuda::par.on(stream),
        thrust::make_counting_iterator(0), thrust::make_counting_iterator(2 * TxAntNum),
                         [d_TxIndices, d_states, d_TxSymbols] __device__(int idx)
                         {
                             int index = curand(d_states + idx) % qam::ConSize;
                             d_TxIndices[idx] = index;
                             d_TxSymbols[idx] = qam::Cons[index];
                         });

        // generate RxSymbols = H * TxSymbols
        float alpha = 1.0f;
        float beta = 0.0f;
        cublasSgemv(handle, CUBLAS_OP_N, 2 * RxAntNum, 2 * TxAntNum, &alpha, H, 2 * RxAntNum, TxSymbols, 1, &beta, RxSymbols, 1);

        auto *d_RxSymbols = this->RxSymbols;
        float Nv = this->Nv;
        // add noise
        thrust::for_each(thrust::cuda::par.on(stream),
        thrust::make_counting_iterator(0), thrust::make_counting_iterator(2 * RxAntNum),
                         [d_states, d_RxSymbols, Nv] __device__(int idx)
                         {
                             d_RxSymbols[idx] += curand_normal(d_states + idx) * Nv;
                         });

    }
};