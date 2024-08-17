#pragma once

#include "Sebas.h"


class MMSE
{

float *d_HtH;
int Lwork;

float *d_work;
int *d_info;

float *d_HtY;

bool isInitialized = false; 

public:
// execute MMSE
void execute(auto& det)
{
    if (!isInitialized)
    {
        // malloc d_HtH
        CUDA_CHECK(cudaMalloc((void **)&d_HtH, 4 * sizeof(float) * det.getTxAntNum() * det.getTxAntNum()));


        cusolverDnSpotrf_bufferSize(Sebas::getInstance().getCusolverHandle(), CUBLAS_FILL_MODE_UPPER, 2 * det.getTxAntNum(), d_HtH, 2 * det.getTxAntNum(), &Lwork);

        // malloc d_work
        CUDA_CHECK(cudaMalloc((void **)&d_work, Lwork * sizeof(float)));

        // malloc d_info
        CUDA_CHECK(cudaMalloc((void **)&d_info, sizeof(int)));

        // malloc d_HtY
        CUDA_CHECK(cudaMalloc((void **)&d_HtY, 2 * sizeof(float) * det.getTxAntNum()));

        isInitialized = true;
    }

    static float alpha = 1.0;
    static float beta = 0.0;

    // get the handle
    cublasHandle_t cublasHandle = Sebas::getInstance().getCublasHandle();
    cusolverDnHandle_t cusolverHandle = Sebas::getInstance().getCusolverHandle();

    // get Tx and Rx
    int Tx = det.getTxAntNum();
    int Rx = det.getRxAntNum();

    cublasSsyrk(cublasHandle,
    CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, 2 * Tx, 2 * Rx, &alpha, det.getH(), 2 * Rx, &beta, d_HtH, 2 * Tx);

    // printMatrix(d_HtH, 2 * Tx, 2 * Tx, "HtH");

    // calculate Ht * y
    cublasSgemv(cublasHandle, CUBLAS_OP_T, 2 * Rx, 2 * Tx, &alpha, det.getH(), 2 * Rx, det.getRxSymbols(), 1, &beta, d_HtY, 1);

    // printVector(d_HtY, 2 * Tx, "HtY");

    // solve HtH * x = HtY
    cusolverDnSpotrf(cusolverHandle, CUBLAS_FILL_MODE_UPPER, 2 * Tx, d_HtH, 2 * Tx, d_work, Lwork, d_info);

    cusolverDnSpotrs(cusolverHandle, CUBLAS_FILL_MODE_UPPER, 2 * Tx, 1, d_HtH, 2 * Tx, d_HtY, 2 * Tx, d_info);

    printVector(d_HtY, 2 * Tx, "x");






}



};