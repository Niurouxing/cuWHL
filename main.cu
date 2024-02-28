
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

int main(int argc, char *argv[])
{
    int Tx = 4;
    int Rx = 8;
    Sebas &sebas = Sebas::getInstance();

    Detection<RC::real, ModType::QAM16> detReal(Tx, Rx);
    Detection<RC::complex, ModType::QAM16> detComplex(Tx, Rx);

    detReal.setSNRdB(15);
    detComplex.setSNRdB(15);

    detReal.generateChannel();
    detComplex.generateChannel();

    // copy to host
    printMatrix(detReal.getH(), Rx * 2, Tx * 2, "HReal");
    printMatrix(detComplex.getH(), Rx, Tx, "HComplex");

    detReal.generateTxSignals();
    detComplex.generateTxSignals();

    printVector(detReal.getTxIndice(), Tx * 2, "TxIndiceReal");
    printVector(detComplex.getTxIndice(), Tx, "TxIndiceComplex");

    printVector(detReal.getTxSymbols(), Tx * 2, "TxSymbolsReal");
    printVector(detComplex.getTxSymbols(), Tx, "TxSymbolsComplex");

    detReal.generateRxSignalsWithNoise();
    detComplex.generateRxSignalsWithNoise();

    printVector(detReal.getRxSymbols(), Rx * 2, "RxSymbolsReal");
    printVector(detComplex.getRxSymbols(), Rx, "RxSymbolsComplex");
    return EXIT_SUCCESS;
}

// cudaStream_t stream = sebas.getStream();
// curandGenerator_t gen = NULL;
// curandOrdering_t order = CURAND_ORDERING_PSEUDO_BEST;

// data_type *d_data = nullptr;

// const int n = 10;

// const data_type mean = 1.0f;
// const data_type stddev = 2.0f;

// CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_data), sizeof(data_type) * n));

// // /* Generate n floats on device */
// sebas.normalDistribution(d_data, n, mean, stddev);
// std::vector<data_type> h_data(n, 0);
// // /* Copy data to host */
// CUDA_CHECK(cudaMemcpyAsync(h_data.data(), d_data,
//                            sizeof(data_type) * h_data.size(),
//                            cudaMemcpyDeviceToHost, stream));

// // /* Sync stream */
// CUDA_CHECK(cudaStreamSynchronize(stream));

// printf("normal\n");
// print_vector(h_data);
// printf("=====\n");

// const int low = 0;
// const int high = ConSize - 1;

// unsigned int *d_data2 = nullptr;
// CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_data2), sizeof(unsigned int) * n));

// /* Generate n unsigned ints on device */
// sebas.uniformIntDistribution(d_data2, n, low, high);
// std::vector<unsigned int> h_data2(n, 0);
// /* Copy data to host */
// CUDA_CHECK(cudaMemcpyAsync(h_data2.data(), d_data2,
//                            sizeof(unsigned int) * h_data2.size(),
//                            cudaMemcpyDeviceToHost, stream));

// /* Sync stream */
// CUDA_CHECK(cudaStreamSynchronize(stream));

// printf("uniform int\n");
// print_vector(h_data2);
// printf("=====\n");

// data_type *d_data3 = nullptr;
// CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_data3), sizeof(data_type) * n));

// thrust::transform(thrust::device, d_data2, d_data2 + n, d_data3, [=] __device__(unsigned int x)
//                   { return Cons[x]; });

// std::vector<data_type> h_data3(n, 0);
// /* Copy data to host */
// CUDA_CHECK(cudaMemcpyAsync(h_data3.data(), d_data3,
//                            sizeof(data_type) * h_data3.size(),
//                            cudaMemcpyDeviceToHost, stream));

// /* Sync stream */
// CUDA_CHECK(cudaStreamSynchronize(stream));

// printf("uniform int to float\n");
// print_vector(h_data3);
// printf("=====\n");