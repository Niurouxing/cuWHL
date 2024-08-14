
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include "utils.h"
// #include "Sebas.h"
#include "Cons.h"
#include "Detection.h"

#include <cusolverDn.h>

int main(int argc, char *argv[])
{
    int Tx = 4;
    int Rx = 8;
    Sebas &sebas = Sebas::getInstance();

    Detection<RC::real, ModType::QAM16> detReal(Tx, Rx);
    // Detection<RC::complex, ModType::QAM16> detComplex(Tx, Rx);

    detReal.setSNRdB(20);
    // detComplex.setSNRdB(15);

    detReal.generateChannel();
    // detComplex.generateChannel();

    // copy to host
    printMatrix(detReal.getH(), Rx * 2, Tx * 2, "HReal");
    // printMatrix(detComplex.getH(), Rx, Tx, "HComplex");

    detReal.generateTxSignals();
    // detComplex.generateTxSignals();

    printVector(detReal.getTxIndice(), Tx * 2, "TxIndiceReal");
    // printVector(detComplex.getTxIndice(), Tx, "TxIndiceComplex");

    printVector(detReal.getTxSymbols(), Tx * 2, "TxSymbolsReal");
    // printVector(detComplex.getTxSymbols(), Tx, "TxSymbolsComplex");

    detReal.generateRxSignalsWithNoise();
    // detComplex.generateRxSignalsWithNoise();

    printVector(detReal.getRxSymbols(), Rx * 2, "RxSymbolsReal");
    // printVector(detComplex.getRxSymbols(), Rx, "RxSymbolsComplex");


    // malloc device memory for HtH
    float *d_HtH;
    cudaMalloc((void**)&d_HtH, 4 * Tx * Tx * sizeof(float));

    float alpha = 1.0;
    float beta = 0.0;
    auto res = cublasSsyrk(sebas.getCublasHandle(), CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, 2 * Tx, 2 * Rx, &alpha, detReal.getH(), 2 * Rx, &beta, d_HtH, 2 * Tx);

    if (res != CUBLAS_STATUS_SUCCESS)
    {
        throw std::runtime_error("cublasSsyrk failed");
    }

    printMatrix(d_HtH, 2 * Tx, 2 * Tx, "HtH");


    // cuSolver
    cusolverDnHandle_t cusolverH = NULL;
    cusolverDnCreate(&cusolverH);
    cusolverDnSetStream(cusolverH, sebas.getStream());

    // potrf
    int Lwork;
    cusolverDnSpotrf_bufferSize(cusolverH, CUBLAS_FILL_MODE_UPPER, 2 * Tx, d_HtH, 2 * Tx, &Lwork);

    float *d_work;
    cudaMalloc((void**)&d_work, Lwork * sizeof(float));

    int *d_info;
    cudaMalloc((void**)&d_info, sizeof(int));

    cusolverDnSpotrf(cusolverH, CUBLAS_FILL_MODE_UPPER, 2 * Tx, d_HtH, 2 * Tx, d_work, Lwork, d_info);

    int info;
    cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

    if (info != 0)
    {
        throw std::runtime_error("cusolverDnSpotrf failed");
    }

    printMatrix(d_HtH, 2 * Tx, 2 * Tx, "HtH");

    // calculate Ht * y
    float *d_HtY;
    cudaMalloc((void**)&d_HtY, 2 * Tx * sizeof(float));

    cublasSgemv(sebas.getCublasHandle(), CUBLAS_OP_T, 2 * Rx, 2 * Tx, &alpha, detReal.getH(), 2 * Rx, detReal.getRxSymbols(), 1, &beta, d_HtY, 1);

    printVector(d_HtY, 2 * Tx, "HtY");

    // solve HtH * x = HtY
    cusolverDnSpotrs(cusolverH, CUBLAS_FILL_MODE_UPPER, 2 * Tx, 1, d_HtH, 2 * Tx, d_HtY, 2 * Tx, d_info);

    cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

    if (info != 0)
    {
        throw std::runtime_error("cusolverDnSpotrs failed");
    }

    printVector(d_HtY, 2 * Tx, "x");




    return EXIT_SUCCESS;
}

