
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>
#include "MMSE.h"

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


    auto mmse = MMSE();

    mmse.execute(detReal);



    return EXIT_SUCCESS;
}

