#pragma once
#include <cusolverDn.h>
#include "Detection.h"
#include <thrust/sort.h>
#include <thrust/sequence.h>

template <int Tx, int Rx, typename qam = QAM<16, RD>, int dm = 3>
__global__ void beta_kernel(int *sIndex, float *Px, float *H, float *RxSymbols, float *beta, float Nv)
{
    // block index for Rx
    int j = blockIdx.x;
    // thread index for Tx
    int i = threadIdx.x;

    __shared__ float sMean[2 * Tx];
    __shared__ float mean_incoming_all;

    if (i == 0)
    {
        mean_incoming_all = 0.0;
    }
    __syncthreads();

    constexpr auto TxAntNum = Tx;
    constexpr auto RxAntNum = Rx;

    constexpr auto ConSize = qam::ConSize;
    constexpr auto Con0 = *qam::Cons;

    float sGAI = 0;

#pragma unroll
    for (int k = 0; k < dm; k++)
    {
        // sIndex 3D Matrix with shape [dm][2 * TxAntNum][2 * RxAntNum]
        int index = sIndex[(i + 2 * TxAntNum * j) * dm + k];
        // Px 3D Matrix with shape [ConSize][2 * TxAntNum][2 * RxAntNum]
        sGAI += Px[(i + 2 * TxAntNum * j) * ConSize + index] * qam::Cons[index];
    }

    sMean[i] = H[j + i * 2 * RxAntNum] * sGAI;

    // wrong code, will cause thread conflict
    // mean_incoming_all += sMean[i];

    // atomic add
    atomicAdd(&mean_incoming_all, sMean[i]);
    __syncthreads();

    // Compute each beta message;
    float sMean_in = mean_incoming_all - sMean[i];
    float HS_0 = H[j + i * 2 * RxAntNum] * Con0;

#pragma unroll
    for (int k = 0; k < ConSize; k++)
    {
        float HS = H[j + i * 2 * RxAntNum] * qam::Cons[k];
        float diff = RxSymbols[j] - sMean_in - HS;
        float diff0 = RxSymbols[j] - sMean_in - HS_0;

        beta[ConSize * (j + i * 2 * RxAntNum) + k] = (-diff * diff + diff0 * diff0) / Nv * 0.5;
    }
    __syncthreads();
}

__global__ void sum_along_RxAntNum_blocks(float *in, float *out, int dim1, int dim2, int dim3)
{
    int ty = blockIdx.x;  // ConSize block index
    int tx = threadIdx.x; // TxAntNum thread index

    if (tx < dim3 && ty < dim1)
    {
        float sum = 0;

        // 遍历RxAntNum维度进行求和
        for (int k = 0; k < dim2; ++k)
        {
            // 计算列优先存储中的索引
            int idx = ty + (k + dim2 * tx) * dim1;
            sum += in[idx];
        }

        // 输出结果，也为列优先存储
        out[ty + tx * dim1] = sum;
    }
}

template <int Tx, int Rx, typename qam = QAM<16, RD>, int dm = 3>
__global__ void alpha_kernel(float *gamma, float *beta, int *sIndex, float *alpha, float *Px)
{
    // Rx as block index
    int j = blockIdx.x;
    // Tx as thread index
    int i = threadIdx.x;

    constexpr auto ConSize = qam::ConSize;
    constexpr auto Con0 = *qam::Cons;

    // shape [dm][2 * TxAntNum]
    __shared__ float expAlpha[dm * 2 * Tx];

    // shape [dm][2 * TxAntNum]
    __shared__ int s_sIndex[dm * 2 * Tx];

    // copy sIndex to shared memory

#pragma unroll
    for (int k = 0; k < dm; k++)
    {
        s_sIndex[k + i * dm] = sIndex[(i + 2 * Tx * j) * dm + k];
    }

#pragma unroll
    for (int k = 0; k < ConSize; k++)
    {
        alpha[(i + 2 * Tx * j) * ConSize + k] = gamma[i * ConSize + k] - beta[(j + i * 2 * Rx) * ConSize + k];
    }

    float expAlphaSum = 0.0;
#pragma unroll
    for (int k = 0; k < dm; k++)
    {
        // int idx = sIndex[(i + 2 * Tx * j) * dm + k];
        int idx = s_sIndex[k + i * dm];
        // expAlpha[k + i * dm] = exp(alpha[(i + 2 * Tx * j) * ConSize + idx] - alpha[(i + 2 * Tx * j) * ConSize + sIndex[(i + 2 * Tx * j) * dm]]);
        expAlpha[k + i * dm] = exp(alpha[(i + 2 * Tx * j) * ConSize + idx] - alpha[(i + 2 * Tx * j) * ConSize + s_sIndex[i * dm]]);
        expAlphaSum += expAlpha[k + i * dm];
    }

// update Px
#pragma unroll
    for (int k = 0; k < dm; k++)
    {
        // int idx = sIndex[(i + 2 * Tx * j) * dm + k];
        int idx = s_sIndex[k + i * dm];
        Px[(i + 2 * Tx * j) * ConSize + idx] = expAlpha[k + i * dm] / expAlphaSum;
    }
}

template <int Tx, int Rx, typename qam = QAM<16, RD>, int dm = 3>
class BsP
{
public:
    cudaStream_t stream;
    cublasHandle_t cublasHandle;
    cusolverDnHandle_t cusolverHandle;

    int loop = 2;

    static constexpr int TxAntNum = Tx;
    static constexpr int RxAntNum = Rx;

    static constexpr int ConSize = qam::ConSize;

    // alpha 3D Matrix with shape [ConSize][2 * TxAntNum][2 * RxAntNum]
    float *alpha;

    // beta 3D Matrix with shape [ConSize][2 * RxAntNum][2 * TxAntNum]
    float *beta;

    // Px 3D Matrix with shape [ConSize][2 * TxAntNum][2 * RxAntNum]
    float *Px;

    // gamma 2D Matrix with shape [ConSize][2 * TxAntNum]
    float *gamma;

    // sIndex 3D Matrix with shape [dm][2 * TxAntNum][2 * RxAntNum]
    // represents the index of the dm most possible symbols, decided by MMSE
    int *sIndex;

    float *HtH;
    float *HtY;

    int Lwork;

    float *d_work;
    int *d_info;

    int *TxEst;

    // Constructor
    BsP()
    {
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        cublasCreate(&cublasHandle);
        cublasSetStream(cublasHandle, stream);
        cusolverDnCreate(&cusolverHandle);
        cusolverDnSetStream(cusolverHandle, stream);

        cudaMalloc(&alpha, 4 * TxAntNum * RxAntNum * ConSize * sizeof(float));
        cudaMalloc(&beta, 4 * RxAntNum * TxAntNum * ConSize * sizeof(float));
        cudaMalloc(&gamma, 2 * TxAntNum * ConSize * sizeof(float));
        cudaMalloc(&Px, 4 * TxAntNum * RxAntNum * ConSize * sizeof(float));
        cudaMalloc(&sIndex, 4 * TxAntNum * RxAntNum * dm * sizeof(int));
        cudaMalloc(&TxEst, 2 * TxAntNum * sizeof(int));

        cudaMalloc(&HtH, 4 * TxAntNum * TxAntNum * sizeof(float));
        cudaMalloc(&HtY, 2 * TxAntNum * sizeof(float));

        cusolverDnSpotrf_bufferSize(cusolverHandle, CUBLAS_FILL_MODE_UPPER, 2 * Tx, HtH, 2 * Tx, &Lwork);

        cudaMalloc((void **)&d_work, Lwork * sizeof(float));
        cudaMalloc((void **)&d_info, sizeof(int));
    }

    // Destructor
    ~BsP()
    {
        cudaFree(alpha);
        cudaFree(beta);
        cudaFree(gamma);
        cudaFree(Px);
        cudaFree(sIndex);
        cudaFree(TxEst);

        cudaFree(HtH);
        cudaFree(HtY);

        cudaFree(d_work);
        cudaFree(d_info);

        cublasDestroy(cublasHandle);
        cusolverDnDestroy(cusolverHandle);
        cudaStreamDestroy(stream);
    }

    void execute(Detection<Tx, Rx, qam> &det)
    {
        // HtH = H^T * H

        float ALPHA = 1.0;
        float BETA = 0.0;

        cublasStatus_t status = cublasSsyrk(cublasHandle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, 2 * Rx, 2 * Tx, &ALPHA, det.H, 2 * Rx, &BETA, HtH, 2 * Tx);
 

 


        // HtY = H^T * Y
        cublasSgemv(cublasHandle, CUBLAS_OP_T, 2 * Rx, 2 * Tx, &ALPHA, det.H, 2 * Rx, det.RxSymbols, 1, &BETA, HtY, 1);


        cusolverDnSpotrf(cusolverHandle, CUBLAS_FILL_MODE_UPPER, 2 * Tx, HtH, 2 * Tx, d_work, Lwork, d_info);

 
        cusolverDnSpotrs(cusolverHandle, CUBLAS_FILL_MODE_UPPER, 2 * Tx, 1, HtH, 2 * Tx, HtY, 2 * Tx, d_info);

        // usually, use i for TxAntNum, j for RxAntNum, k for ConSize

        float *d_HtH = this->HtH;
        float *d_HtY = this->HtY;
        float *d_Px = this->Px;
        float *d_alpha = this->alpha;
        float *d_gamma = this->gamma;
        int *d_sIndex = this->sIndex;

        thrust::for_each(thrust::cuda::par.on(stream),
                         thrust::make_counting_iterator(0), thrust::make_counting_iterator(2 * Tx),
                         [d_alpha, d_HtH, d_HtY, d_Px, d_sIndex, d_gamma] __device__(int i)
                         {
                             // HtY is already the pre-calculated MMSE estimate via cusolver, no need to calculate it again

                             // MMSEEst = HtHInv * HtY

                             // create temporary distList
                             float distList[ConSize];

                             // find the best index
                             float minDist = 1e10;
                             int bestIndex = 0;
                             for (int k = 0; k < ConSize; k++)
                             {
                                 float dist = abs(qam::Cons[k] - d_HtY[i]);
                                 distList[k] = dist;
                                 d_gamma[i * ConSize + k] = -dist;
                                 if (dist < minDist)
                                 {
                                     minDist = dist;
                                     bestIndex = k;
                                 }
                             }

                             // fill alpha and Px
                             for (int j = 0; j < 2 * RxAntNum; j++)
                             {
                                 // alpha [ConSize][2 * TxAntNum][2 * RxAntNum]

                                 for (int k = 0; k < ConSize; k++)
                                 {
                                     d_alpha[(i + j * 2 * TxAntNum) * ConSize + k] = d_gamma[i * ConSize + k];
                                 }

                                 // Px [ConSize][2 * TxAntNum][2 * RxAntNum]
                                 d_Px[(i + j * 2 * TxAntNum) * ConSize + bestIndex] = 1;
                             }

                             //  initialize temporary minkRes
                             int minkRes[ConSize];
                             thrust::sequence(thrust::device, minkRes, minkRes + ConSize);

                             //  sort by key
                             thrust::sort_by_key(thrust::device, distList, distList + ConSize, minkRes);

                             // fill sIndex
                             for (int j = 0; j < 2 * RxAntNum; j++)
                             {
                                 for (int k = 0; k < dm; k++)
                                 {
                                     // sIndex [dm][2 * TxAntNum][2 * RxAntNum]
                                     d_sIndex[(i + j * 2 * TxAntNum) * dm + k] = minkRes[k];
                                 }
                             }
                         });

        // main loop
        for (int iter = 0; iter < loop; iter++)
        {
            beta_kernel<Tx, Rx, qam, dm><<<2 * RxAntNum, 2 * TxAntNum, 0, stream>>>(sIndex, Px, det.H, det.RxSymbols, beta, det.Nv);

            // sum beta over RxAntNum to get gamma
            sum_along_RxAntNum_blocks<<<ConSize, 2 * TxAntNum, 0, stream>>>(beta, gamma, ConSize, 2 * RxAntNum, 2 * TxAntNum);

            // // memset alpha to 0
            // cudaMemset(alpha, 0, 4 * TxAntNum * RxAntNum * ConSize * sizeof(float));

            // // memset Px to 0
            // cudaMemset(Px, 0, 4 * TxAntNum * RxAntNum * ConSize * sizeof(float));

            alpha_kernel<Tx, Rx, qam, dm><<<2 * RxAntNum, 2 * TxAntNum, 0, stream>>>(gamma, beta, sIndex, alpha, Px);

        }

        // argmax gamma to get TxEst
        auto *d_TxEst = this->TxEst;
        thrust::for_each(thrust::cuda::par.on(stream),
                         thrust::make_counting_iterator(0), thrust::make_counting_iterator(2 * TxAntNum),
                         [d_gamma, d_TxEst] __device__(int i)
                         {
                             float maxVal = -1e10;
                             int maxIdx = 0;
                             for (int j = 0; j < ConSize; j++)
                             {
                                 if (d_gamma[i * ConSize + j] > maxVal)
                                 {
                                     maxVal = d_gamma[i * ConSize + j];
                                     maxIdx = j;
                                 }
                             }
                             d_TxEst[i] = maxIdx;
                         });
    }
};