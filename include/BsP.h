#pragma once
#include <cusolverDn.h>
#include "Detection.h"
#include <thrust/sort.h>
#include <thrust/sequence.h>



template <int Tx, int Rx, typename qam = QAM<16, RD>, int dm = 2>
class BsP
{
public:
    cudaStream_t stream;
    cublasHandle_t cublasHandle;
    cusolverDnHandle_t cusolverHandle;

    int loop = 4;

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

        cudaMalloc(&HtH, 4 * TxAntNum * TxAntNum * sizeof(float));
        cudaMalloc(&HtY, 4 * TxAntNum * sizeof(float));

        cusolverDnSpotrf_bufferSize(cusolverHandle, CUBLAS_FILL_MODE_UPPER, 2 * Rx, HtH, 2 * Tx, &Lwork);

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

        cublasSsyrk(cublasHandle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, 2 * Tx, 2 * Rx, &ALPHA, det.H, 2 * Rx, &BETA, HtH, 2 * Tx);

        // HtY = H^T * Y
        cublasSgemv(cublasHandle, CUBLAS_OP_T, 2 * Rx, 2 * Tx, &ALPHA, det.H, 2 * Rx, det.RxSymbols, 1, &BETA, HtY, 1);

        float *d_HtH = this->HtH;
        float Nv = det.Nv;
        thrust::for_each(thrust::cuda::par.on(stream),
                         thrust::make_counting_iterator(0), thrust::make_counting_iterator(2 * Tx),
                         [d_HtH, Nv] __device__(int idx)
                         {
                             d_HtH[idx * 2 * Tx + idx] += Nv;
                         });

        // solve HtH * x = HtY
        cusolverDnSpotrf(cusolverHandle, CUBLAS_FILL_MODE_UPPER, 2 * Tx, d_HtH, 2 * Tx, d_work, Lwork, d_info);

        cusolverDnSpotrs(cusolverHandle, CUBLAS_FILL_MODE_UPPER, 2 * Tx, 1, d_HtH, 2 * Tx, HtY, 2 * Tx, d_info);

        // usually, use i for TxAntNum, j for RxAntNum, k for ConSize

        // cpu code for reference
        // memset(Px, 0, TxAntNum * RxAntNum * ConSize * sizeof(double));
        // for (int i = 0; i < TxAntNum; i++)
        // {
        //     std::complex<double> MMSEEst = 0;
        //     for (int j = 0; j < TxAntNum; j++)
        //     {
        //         MMSEEst += HtHInv[i * TxAntNum + j] * HtR[j];
        //     }

        //     double minDist = 1e10;
        //     int bestIndex = 0;
        //     for (int k = 0; k < ConSize; k++)
        //     {
        //         double dist = std::abs(ConsComplex[k] - MMSEEst);
        //         distList[k] = dist;
        //         gamma[i * ConSize + k] = -dist;

        //         if (dist < minDist)
        //         {
        //             minDist = dist;
        //             bestIndex = k;
        //         }
        //     }

        //     for (int j = 0; j < RxAntNum; j++)
        //     {
        //         std::copy_n(&gamma[i * ConSize], ConSize, &alpha[(i * RxAntNum + j) * ConSize]);

        //         Px[(i * RxAntNum + j) * ConSize + bestIndex] = 1;
        //     }

        //     mink(distList, ConSize, minkRes, dm);
        //     for (int j = 0; j < RxAntNum; j++)
        //     {
        //         std::copy_n(minkRes, dm, &sIndex[(i * RxAntNum + j) * dm]);
        //     }
        // }

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

            //     // complex cpu code for reference
            //     std::complex<double> mean_incoming_all = 0; // use the name in zwy's code
            //     // double var_incoming_all = 0;  // turn off variance according to wendy's code
            //     for (int i = 0; i < TxAntNum; i++)
            //     {
            //         std::complex<double> sGAI = 0;
            //         // double sGAI_var = 0;
            //         for (int k = 0; k < dm; k++)
            //         {
            //             int index = sIndex[(i * RxAntNum + j) * dm + k];
            //             sGAI += Px[(i * RxAntNum + j) * ConSize + index] * ConsComplex[index];
            //             // sGAI_var += std::abs(Px[i][j][index] * ConsComplex[index]) * std::abs(Px[i][j][index] * ConsComplex[index]);
            //         }

            //         sMean[i] = H[j * TxAntNum + i] * sGAI;
            //         // sVar[i] = std::norm(H[j][i]) * (sGAI_var - std::norm(sGAI));

            //         mean_incoming_all += sMean[i];
            //         // var_incoming_all += sVar[i];
            //     }

            //     // Compute each beta message;
            //     for (int i = 0; i < TxAntNum; i++)
            //     {
            //         std::complex<double> sMean_in = mean_incoming_all - sMean[i];
            //         // double sVar_in = var_incoming_all - sVar[i] + Nv;
            //         std::complex<double> HS_0 = H[j * TxAntNum + i] * ConsComplex[0];

            //         for (int k = 0; k < ConSize; k++)
            //         {
            //             std::complex<double> HS = precomputedHCons[j * TxAntNum * ConSize + i * ConSize + k];
            //             // beta[j][i][k] =  -std::norm(RxSymbols[j] - sMean_in - HS) / sVar_in + std::norm(RxSymbols[j] - sMean_in - HS_0) / sVar_in;
            //             beta[j * TxAntNum * ConSize + i * ConSize + k] = (-std::norm(RxSymbols[j] - sMean_in - HS) + std::norm(RxSymbols[j] - sMean_in - HS_0)) * NvInv * 0.5;
            //         }
            //     }

        }
    }
};