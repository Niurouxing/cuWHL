#pragma once
#include <type_traits>
#include "utils.h"
#include "Cons.h"
#include <cmath>
#include "Sebas.h"
enum class RC
{
    real,
    complex
};

// ConditionallyIncluded Data struct
template <RC rc>
struct DetectionUtils
{
    // None for default
};

// special for real
template <>
struct DetectionUtils<RC::real>
{
    using valueType = data_type;
    const valueType *Cons2;

    int TxAntNum2;
    int RxAntNum2;
};

// special for complex
template <>
struct DetectionUtils<RC::complex>
{
    using valueType = std::conditional_t<std::is_same<data_type, float>::value, cuFloatComplex, cuDoubleComplex>;
};

template <RC rc>
class Detection : public DetectionUtils<rc>
{
public:
    using valueType = typename DetectionUtils<rc>::valueType;
    int TxAntNum;
    int RxAntNum;
    int ModType;
    int ConSize;
    int bitLength;
    valueType SNRdB;
    valueType Nv;
    valueType sqrtNvDiv2;
    valueType NvInv;

    const valueType *Cons;
    const bool *bitCons;

    unsigned int *TxIndice;
    bool *TxBits;

    valueType *TxSymbols;
    valueType *RxSymbols;
    valueType *H;

    Detection(int TxAntNum, int RxAntNum, int ModType, double SNRdB)
    {
        this->TxAntNum = TxAntNum;
        this->TxAntNum2 = TxAntNum * 2;
        this->RxAntNum = RxAntNum;
        this->ModType = ModType;
        this->SNRdB = SNRdB;

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&this->TxBits), sizeof(unsigned int) * this->TxAntNum * this->ModType));

        this->Nv = static_cast<data_type>(TxAntNum * RxAntNum / (pow(10, SNRdB / 10) * ModType * TxAntNum));
        this->sqrtNvDiv2 = std::sqrt(Nv / 2);
        this->NvInv = 1 / Nv;

        if constexpr (rc == RC::real)
        {
            switch (ModType)
            {
            case 2:
                this->ConSize = 2;
                this->bitLength = 1;

                this->Cons = realConsMod2;
                this->Cons2 = realCons2Mod2;
                this->bitCons = realBitConsMod2;
                break;
            case 4:
                this->ConSize = 4;
                this->bitLength = 2;

                this->Cons = realConsMod4;
                this->Cons2 = realCons2Mod4;
                this->bitCons = realBitConsMod4;
                break;
            case 6:
                this->ConSize = 8;
                this->bitLength = 3;

                this->Cons = realConsMod6;
                this->Cons2 = realCons2Mod6;
                this->bitCons = realBitConsMod6;
                break;
            case 8:
                this->ConSize = 16;
                this->bitLength = 4;

                this->Cons = realConsMod8;
                this->Cons2 = realCons2Mod8;
                this->bitCons = realBitConsMod8;
                break;
            }

            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&this->TxIndice), sizeof(unsigned int) * this->TxAntNum2));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&this->TxSymbols), sizeof(data_type) * this->TxAntNum2));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&this->RxSymbols), sizeof(data_type) * this->RxAntNum));

            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&this->H), sizeof(data_type) * this->TxAntNum2 * this->RxAntNum2));
        }
        else
        {
            switch (ModType)
            {
            case 2:
                if (complexConsInitMod2 == false)
                {
                    complex_type complexConsMod2_host[4] = {
                        make_complex(normfMod2, normfMod2),
                        make_complex(normfMod2, -normfMod2),
                        make_complex(-normfMod2, -normfMod2),
                        make_complex(-normfMod2, +normfMod2)};
                    CUDA_CHECK(cudaMemcpyToSymbol(complexConsMod2, complexConsMod2_host, sizeof(complex_type) * 4));
                    complexConsInitMod2 = true;
                }

                this->ConSize = 4;
                this->bitLength = 2;

                this->Cons = complexConsMod2;
                this->bitCons = complexBitConsMod2;
                break;
            case 4:
                if (complexConsInitMod4 == false)
                {
                    complex_type complexConsMod4_host[16] = {
                        make_complex(-1 * normfMod4, -1 * normfMod4),
                        make_complex(-1 * normfMod4, -3 * normfMod4),
                        make_complex(-1 * normfMod4, 1 * normfMod4),
                        make_complex(-1 * normfMod4, 3 * normfMod4),
                        make_complex(-3 * normfMod4, -1 * normfMod4),
                        make_complex(-3 * normfMod4, -3 * normfMod4),
                        make_complex(-3 * normfMod4, 1 * normfMod4),
                        make_complex(-3 * normfMod4, 3 * normfMod4),
                        make_complex(1 * normfMod4, -1 * normfMod4),
                        make_complex(1 * normfMod4, -3 * normfMod4),
                        make_complex(1 * normfMod4, 1 * normfMod4),
                        make_complex(1 * normfMod4, 3 * normfMod4),
                        make_complex(3 * normfMod4, -1 * normfMod4),
                        make_complex(3 * normfMod4, -3 * normfMod4),
                        make_complex(3 * normfMod4, 1 * normfMod4),
                        make_complex(3 * normfMod4, 3 * normfMod4)};
                    CUDA_CHECK(cudaMemcpyToSymbol(complexConsMod4, complexConsMod4_host, sizeof(complex_type) * 16));
                    complexConsInitMod4 = true;
                }

                this->ConSize = 16;
                this->bitLength = 4;

                this->Cons = complexConsMod4;
                this->bitCons = complexBitConsMod4;
                break;
            case 6:
                if (complexConsInitMod6 == false)
                {
                    complex_type complexConsMod6_host[64] = {
                        make_complex(-3 * normfMod6, -3 * normfMod6),
                        make_complex(-3 * normfMod6, -1 * normfMod6),
                        make_complex(-3 * normfMod6, -5 * normfMod6),
                        make_complex(-3 * normfMod6, -7 * normfMod6),
                        make_complex(-3 * normfMod6, 3 * normfMod6),
                        make_complex(-3 * normfMod6, 1 * normfMod6),
                        make_complex(-3 * normfMod6, 5 * normfMod6),
                        make_complex(-3 * normfMod6, 7 * normfMod6),
                        make_complex(-1 * normfMod6, -3 * normfMod6),
                        make_complex(-1 * normfMod6, -1 * normfMod6),
                        make_complex(-1 * normfMod6, -5 * normfMod6),
                        make_complex(-1 * normfMod6, -7 * normfMod6),
                        make_complex(-1 * normfMod6, 3 * normfMod6),
                        make_complex(-1 * normfMod6, 1 * normfMod6),
                        make_complex(-1 * normfMod6, 5 * normfMod6),
                        make_complex(-1 * normfMod6, 7 * normfMod6),
                        make_complex(-5 * normfMod6, -3 * normfMod6),
                        make_complex(-5 * normfMod6, -1 * normfMod6),
                        make_complex(-5 * normfMod6, -5 * normfMod6),
                        make_complex(-5 * normfMod6, -7 * normfMod6),
                        make_complex(-5 * normfMod6, 3 * normfMod6),
                        make_complex(-5 * normfMod6, 1 * normfMod6),
                        make_complex(-5 * normfMod6, 5 * normfMod6),
                        make_complex(-5 * normfMod6, 7 * normfMod6),
                        make_complex(-7 * normfMod6, -3 * normfMod6),
                        make_complex(-7 * normfMod6, -1 * normfMod6),
                        make_complex(-7 * normfMod6, -5 * normfMod6),
                        make_complex(-7 * normfMod6, -7 * normfMod6),
                        make_complex(-7 * normfMod6, 3 * normfMod6),
                        make_complex(-7 * normfMod6, 1 * normfMod6),
                        make_complex(-7 * normfMod6, 5 * normfMod6),
                        make_complex(-7 * normfMod6, 7 * normfMod6),
                        make_complex(3 * normfMod6, -3 * normfMod6),
                        make_complex(3 * normfMod6, -1 * normfMod6),
                        make_complex(3 * normfMod6, -5 * normfMod6),
                        make_complex(3 * normfMod6, -7 * normfMod6),
                        make_complex(3 * normfMod6, 3 * normfMod6),
                        make_complex(3 * normfMod6, 1 * normfMod6),
                        make_complex(3 * normfMod6, 5 * normfMod6),
                        make_complex(3 * normfMod6, 7 * normfMod6),
                        make_complex(1 * normfMod6, -3 * normfMod6),
                        make_complex(1 * normfMod6, -1 * normfMod6),
                        make_complex(1 * normfMod6, -5 * normfMod6),
                        make_complex(1 * normfMod6, -7 * normfMod6),
                        make_complex(1 * normfMod6, 3 * normfMod6),
                        make_complex(1 * normfMod6, 1 * normfMod6),
                        make_complex(1 * normfMod6, 5 * normfMod6),
                        make_complex(1 * normfMod6, 7 * normfMod6),
                        make_complex(5 * normfMod6, -3 * normfMod6),
                        make_complex(5 * normfMod6, -1 * normfMod6),
                        make_complex(5 * normfMod6, -5 * normfMod6),
                        make_complex(5 * normfMod6, -7 * normfMod6),
                        make_complex(5 * normfMod6, 3 * normfMod6),
                        make_complex(5 * normfMod6, 1 * normfMod6),
                        make_complex(5 * normfMod6, 5 * normfMod6),
                        make_complex(5 * normfMod6, 7 * normfMod6),
                        make_complex(7 * normfMod6, -3 * normfMod6),
                        make_complex(7 * normfMod6, -1 * normfMod6),
                        make_complex(7 * normfMod6, -5 * normfMod6),
                        make_complex(7 * normfMod6, -7 * normfMod6),
                        make_complex(7 * normfMod6, 3 * normfMod6),
                        make_complex(7 * normfMod6, 1 * normfMod6),
                        make_complex(7 * normfMod6, 5 * normfMod6),
                        make_complex(7 * normfMod6, 7 * normfMod6),
                    };
                    CUDA_CHECK(cudaMemcpyToSymbol(complexConsMod6, complexConsMod6_host, sizeof(complex_type) * 64));
                    complexConsInitMod6 = true;
                }
                this->ConSize = 64;
                this->bitLength = 6;

                this->Cons = complexConsMod6;
                this->bitCons = complexBitConsMod6;
                break;
            case 8:
                if (complexConsInitMod8 == false)
                {
                    complex_type complexConsMod8_host[256] = {
                        make_complex(-5 * normfMod8, -5 * normfMod8),
                        make_complex(-5 * normfMod8, -7 * normfMod8),
                        make_complex(-5 * normfMod8, -3 * normfMod8),
                        make_complex(-5 * normfMod8, -1 * normfMod8),
                        make_complex(-5 * normfMod8, -11 * normfMod8),
                        make_complex(-5 * normfMod8, -9 * normfMod8),
                        make_complex(-5 * normfMod8, -13 * normfMod8),
                        make_complex(-5 * normfMod8, -15 * normfMod8),
                        make_complex(-5 * normfMod8, 5 * normfMod8),
                        make_complex(-5 * normfMod8, 7 * normfMod8),
                        make_complex(-5 * normfMod8, 3 * normfMod8),
                        make_complex(-5 * normfMod8, 1 * normfMod8),
                        make_complex(-5 * normfMod8, 11 * normfMod8),
                        make_complex(-5 * normfMod8, 9 * normfMod8),
                        make_complex(-5 * normfMod8, 13 * normfMod8),
                        make_complex(-5 * normfMod8, 15 * normfMod8),
                        make_complex(-7 * normfMod8, -5 * normfMod8),
                        make_complex(-7 * normfMod8, -7 * normfMod8),
                        make_complex(-7 * normfMod8, -3 * normfMod8),
                        make_complex(-7 * normfMod8, -1 * normfMod8),
                        make_complex(-7 * normfMod8, -11 * normfMod8),
                        make_complex(-7 * normfMod8, -9 * normfMod8),
                        make_complex(-7 * normfMod8, -13 * normfMod8),
                        make_complex(-7 * normfMod8, -15 * normfMod8),
                        make_complex(-7 * normfMod8, 5 * normfMod8),
                        make_complex(-7 * normfMod8, 7 * normfMod8),
                        make_complex(-7 * normfMod8, 3 * normfMod8),
                        make_complex(-7 * normfMod8, 1 * normfMod8),
                        make_complex(-7 * normfMod8, 11 * normfMod8),
                        make_complex(-7 * normfMod8, 9 * normfMod8),
                        make_complex(-7 * normfMod8, 13 * normfMod8),
                        make_complex(-7 * normfMod8, 15 * normfMod8),
                        make_complex(-3 * normfMod8, -5 * normfMod8),
                        make_complex(-3 * normfMod8, -7 * normfMod8),
                        make_complex(-3 * normfMod8, -3 * normfMod8),
                        make_complex(-3 * normfMod8, -1 * normfMod8),
                        make_complex(-3 * normfMod8, -11 * normfMod8),
                        make_complex(-3 * normfMod8, -9 * normfMod8),
                        make_complex(-3 * normfMod8, -13 * normfMod8),
                        make_complex(-3 * normfMod8, -15 * normfMod8),
                        make_complex(-3 * normfMod8, 5 * normfMod8),
                        make_complex(-3 * normfMod8, 7 * normfMod8),
                        make_complex(-3 * normfMod8, 3 * normfMod8),
                        make_complex(-3 * normfMod8, 1 * normfMod8),
                        make_complex(-3 * normfMod8, 11 * normfMod8),
                        make_complex(-3 * normfMod8, 9 * normfMod8),
                        make_complex(-3 * normfMod8, 13 * normfMod8),
                        make_complex(-3 * normfMod8, 15 * normfMod8),
                        make_complex(-1 * normfMod8, -5 * normfMod8),
                        make_complex(-1 * normfMod8, -7 * normfMod8),
                        make_complex(-1 * normfMod8, -3 * normfMod8),
                        make_complex(-1 * normfMod8, -1 * normfMod8),
                        make_complex(-1 * normfMod8, -11 * normfMod8),
                        make_complex(-1 * normfMod8, -9 * normfMod8),
                        make_complex(-1 * normfMod8, -13 * normfMod8),
                        make_complex(-1 * normfMod8, -15 * normfMod8),
                        make_complex(-1 * normfMod8, 5 * normfMod8),
                        make_complex(-1 * normfMod8, 7 * normfMod8),
                        make_complex(-1 * normfMod8, 3 * normfMod8),
                        make_complex(-1 * normfMod8, 1 * normfMod8),
                        make_complex(-1 * normfMod8, 11 * normfMod8),
                        make_complex(-1 * normfMod8, 9 * normfMod8),
                        make_complex(-1 * normfMod8, 13 * normfMod8),
                        make_complex(-1 * normfMod8, 15 * normfMod8),
                        make_complex(-11 * normfMod8, -5 * normfMod8),
                        make_complex(-11 * normfMod8, -7 * normfMod8),
                        make_complex(-11 * normfMod8, -3 * normfMod8),
                        make_complex(-11 * normfMod8, -1 * normfMod8),
                        make_complex(-11 * normfMod8, -11 * normfMod8),
                        make_complex(-11 * normfMod8, -9 * normfMod8),
                        make_complex(-11 * normfMod8, -13 * normfMod8),
                        make_complex(-11 * normfMod8, -15 * normfMod8),
                        make_complex(-11 * normfMod8, 5 * normfMod8),
                        make_complex(-11 * normfMod8, 7 * normfMod8),
                        make_complex(-11 * normfMod8, 3 * normfMod8),
                        make_complex(-11 * normfMod8, 1 * normfMod8),
                        make_complex(-11 * normfMod8, 11 * normfMod8),
                        make_complex(-11 * normfMod8, 9 * normfMod8),
                        make_complex(-11 * normfMod8, 13 * normfMod8),
                        make_complex(-11 * normfMod8, 15 * normfMod8),
                        make_complex(-9 * normfMod8, -5 * normfMod8),
                        make_complex(-9 * normfMod8, -7 * normfMod8),
                        make_complex(-9 * normfMod8, -3 * normfMod8),
                        make_complex(-9 * normfMod8, -1 * normfMod8),
                        make_complex(-9 * normfMod8, -11 * normfMod8),
                        make_complex(-9 * normfMod8, -9 * normfMod8),
                        make_complex(-9 * normfMod8, -13 * normfMod8),
                        make_complex(-9 * normfMod8, -15 * normfMod8),
                        make_complex(-9 * normfMod8, 5 * normfMod8),
                        make_complex(-9 * normfMod8, 7 * normfMod8),
                        make_complex(-9 * normfMod8, 3 * normfMod8),
                        make_complex(-9 * normfMod8, 1 * normfMod8),
                        make_complex(-9 * normfMod8, 11 * normfMod8),
                        make_complex(-9 * normfMod8, 9 * normfMod8),
                        make_complex(-9 * normfMod8, 13 * normfMod8),
                        make_complex(-9 * normfMod8, 15 * normfMod8),
                        make_complex(-13 * normfMod8, -5 * normfMod8),
                        make_complex(-13 * normfMod8, -7 * normfMod8),
                        make_complex(-13 * normfMod8, -3 * normfMod8),
                        make_complex(-13 * normfMod8, -1 * normfMod8),
                        make_complex(-13 * normfMod8, -11 * normfMod8),
                        make_complex(-13 * normfMod8, -9 * normfMod8),
                        make_complex(-13 * normfMod8, -13 * normfMod8),
                        make_complex(-13 * normfMod8, -15 * normfMod8),
                        make_complex(-13 * normfMod8, 5 * normfMod8),
                        make_complex(-13 * normfMod8, 7 * normfMod8),
                        make_complex(-13 * normfMod8, 3 * normfMod8),
                        make_complex(-13 * normfMod8, 1 * normfMod8),
                        make_complex(-13 * normfMod8, 11 * normfMod8),
                        make_complex(-13 * normfMod8, 9 * normfMod8),
                        make_complex(-13 * normfMod8, 13 * normfMod8),
                        make_complex(-13 * normfMod8, 15 * normfMod8),
                        make_complex(-15 * normfMod8, -5 * normfMod8),
                        make_complex(-15 * normfMod8, -7 * normfMod8),
                        make_complex(-15 * normfMod8, -3 * normfMod8),
                        make_complex(-15 * normfMod8, -1 * normfMod8),
                        make_complex(-15 * normfMod8, -11 * normfMod8),
                        make_complex(-15 * normfMod8, -9 * normfMod8),
                        make_complex(-15 * normfMod8, -13 * normfMod8),
                        make_complex(-15 * normfMod8, -15 * normfMod8),
                        make_complex(-15 * normfMod8, 5 * normfMod8),
                        make_complex(-15 * normfMod8, 7 * normfMod8),
                        make_complex(-15 * normfMod8, 3 * normfMod8),
                        make_complex(-15 * normfMod8, 1 * normfMod8),
                        make_complex(-15 * normfMod8, 11 * normfMod8),
                        make_complex(-15 * normfMod8, 9 * normfMod8),
                        make_complex(-15 * normfMod8, 13 * normfMod8),
                        make_complex(-15 * normfMod8, 15 * normfMod8),
                        make_complex(5 * normfMod8, -5 * normfMod8),
                        make_complex(5 * normfMod8, -7 * normfMod8),
                        make_complex(5 * normfMod8, -3 * normfMod8),
                        make_complex(5 * normfMod8, -1 * normfMod8),
                        make_complex(5 * normfMod8, -11 * normfMod8),
                        make_complex(5 * normfMod8, -9 * normfMod8),
                        make_complex(5 * normfMod8, -13 * normfMod8),
                        make_complex(5 * normfMod8, -15 * normfMod8),
                        make_complex(5 * normfMod8, 5 * normfMod8),
                        make_complex(5 * normfMod8, 7 * normfMod8),
                        make_complex(5 * normfMod8, 3 * normfMod8),
                        make_complex(5 * normfMod8, 1 * normfMod8),
                        make_complex(5 * normfMod8, 11 * normfMod8),
                        make_complex(5 * normfMod8, 9 * normfMod8),
                        make_complex(5 * normfMod8, 13 * normfMod8),
                        make_complex(5 * normfMod8, 15 * normfMod8),
                        make_complex(7 * normfMod8, -5 * normfMod8),
                        make_complex(7 * normfMod8, -7 * normfMod8),
                        make_complex(7 * normfMod8, -3 * normfMod8),
                        make_complex(7 * normfMod8, -1 * normfMod8),
                        make_complex(7 * normfMod8, -11 * normfMod8),
                        make_complex(7 * normfMod8, -9 * normfMod8),
                        make_complex(7 * normfMod8, -13 * normfMod8),
                        make_complex(7 * normfMod8, -15 * normfMod8),
                        make_complex(7 * normfMod8, 5 * normfMod8),
                        make_complex(7 * normfMod8, 7 * normfMod8),
                        make_complex(7 * normfMod8, 3 * normfMod8),
                        make_complex(7 * normfMod8, 1 * normfMod8),
                        make_complex(7 * normfMod8, 11 * normfMod8),
                        make_complex(7 * normfMod8, 9 * normfMod8),
                        make_complex(7 * normfMod8, 13 * normfMod8),
                        make_complex(7 * normfMod8, 15 * normfMod8),
                        make_complex(3 * normfMod8, -5 * normfMod8),
                        make_complex(3 * normfMod8, -7 * normfMod8),
                        make_complex(3 * normfMod8, -3 * normfMod8),
                        make_complex(3 * normfMod8, -1 * normfMod8),
                        make_complex(3 * normfMod8, -11 * normfMod8),
                        make_complex(3 * normfMod8, -9 * normfMod8),
                        make_complex(3 * normfMod8, -13 * normfMod8),
                        make_complex(3 * normfMod8, -15 * normfMod8),
                        make_complex(3 * normfMod8, 5 * normfMod8),
                        make_complex(3 * normfMod8, 7 * normfMod8),
                        make_complex(3 * normfMod8, 3 * normfMod8),
                        make_complex(3 * normfMod8, 1 * normfMod8),
                        make_complex(3 * normfMod8, 11 * normfMod8),
                        make_complex(3 * normfMod8, 9 * normfMod8),
                        make_complex(3 * normfMod8, 13 * normfMod8),
                        make_complex(3 * normfMod8, 15 * normfMod8),
                        make_complex(1 * normfMod8, -5 * normfMod8),
                        make_complex(1 * normfMod8, -7 * normfMod8),
                        make_complex(1 * normfMod8, -3 * normfMod8),
                        make_complex(1 * normfMod8, -1 * normfMod8),
                        make_complex(1 * normfMod8, -11 * normfMod8),
                        make_complex(1 * normfMod8, -9 * normfMod8),
                        make_complex(1 * normfMod8, -13 * normfMod8),
                        make_complex(1 * normfMod8, -15 * normfMod8),
                        make_complex(1 * normfMod8, 5 * normfMod8),
                        make_complex(1 * normfMod8, 7 * normfMod8),
                        make_complex(1 * normfMod8, 3 * normfMod8),
                        make_complex(1 * normfMod8, 1 * normfMod8),
                        make_complex(1 * normfMod8, 11 * normfMod8),
                        make_complex(1 * normfMod8, 9 * normfMod8),
                        make_complex(1 * normfMod8, 13 * normfMod8),
                        make_complex(1 * normfMod8, 15 * normfMod8),
                        make_complex(11 * normfMod8, -5 * normfMod8),
                        make_complex(11 * normfMod8, -7 * normfMod8),
                        make_complex(11 * normfMod8, -3 * normfMod8),
                        make_complex(11 * normfMod8, -1 * normfMod8),
                        make_complex(11 * normfMod8, -11 * normfMod8),
                        make_complex(11 * normfMod8, -9 * normfMod8),
                        make_complex(11 * normfMod8, -13 * normfMod8),
                        make_complex(11 * normfMod8, -15 * normfMod8),
                        make_complex(11 * normfMod8, 5 * normfMod8),
                        make_complex(11 * normfMod8, 7 * normfMod8),
                        make_complex(11 * normfMod8, 3 * normfMod8),
                        make_complex(11 * normfMod8, 1 * normfMod8),
                        make_complex(11 * normfMod8, 11 * normfMod8),
                        make_complex(11 * normfMod8, 9 * normfMod8),
                        make_complex(11 * normfMod8, 13 * normfMod8),
                        make_complex(11 * normfMod8, 15 * normfMod8),
                        make_complex(9 * normfMod8, -5 * normfMod8),
                        make_complex(9 * normfMod8, -7 * normfMod8),
                        make_complex(9 * normfMod8, -3 * normfMod8),
                        make_complex(9 * normfMod8, -1 * normfMod8),
                        make_complex(9 * normfMod8, -11 * normfMod8),
                        make_complex(9 * normfMod8, -9 * normfMod8),
                        make_complex(9 * normfMod8, -13 * normfMod8),
                        make_complex(9 * normfMod8, -15 * normfMod8),
                        make_complex(9 * normfMod8, 5 * normfMod8),
                        make_complex(9 * normfMod8, 7 * normfMod8),
                        make_complex(9 * normfMod8, 3 * normfMod8),
                        make_complex(9 * normfMod8, 1 * normfMod8),
                        make_complex(9 * normfMod8, 11 * normfMod8),
                        make_complex(9 * normfMod8, 9 * normfMod8),
                        make_complex(9 * normfMod8, 13 * normfMod8),
                        make_complex(9 * normfMod8, 15 * normfMod8),
                        make_complex(13 * normfMod8, -5 * normfMod8),
                        make_complex(13 * normfMod8, -7 * normfMod8),
                        make_complex(13 * normfMod8, -3 * normfMod8),
                        make_complex(13 * normfMod8, -1 * normfMod8),
                        make_complex(13 * normfMod8, -11 * normfMod8),
                        make_complex(13 * normfMod8, -9 * normfMod8),
                        make_complex(13 * normfMod8, -13 * normfMod8),
                        make_complex(13 * normfMod8, -15 * normfMod8),
                        make_complex(13 * normfMod8, 5 * normfMod8),
                        make_complex(13 * normfMod8, 7 * normfMod8),
                        make_complex(13 * normfMod8, 3 * normfMod8),
                        make_complex(13 * normfMod8, 1 * normfMod8),
                        make_complex(13 * normfMod8, 11 * normfMod8),
                        make_complex(13 * normfMod8, 9 * normfMod8),
                        make_complex(13 * normfMod8, 13 * normfMod8),
                        make_complex(13 * normfMod8, 15 * normfMod8),
                        make_complex(15 * normfMod8, -5 * normfMod8),
                        make_complex(15 * normfMod8, -7 * normfMod8),
                        make_complex(15 * normfMod8, -3 * normfMod8),
                        make_complex(15 * normfMod8, -1 * normfMod8),
                        make_complex(15 * normfMod8, -11 * normfMod8),
                        make_complex(15 * normfMod8, -9 * normfMod8),
                        make_complex(15 * normfMod8, -13 * normfMod8),
                        make_complex(15 * normfMod8, -15 * normfMod8),
                        make_complex(15 * normfMod8, 5 * normfMod8),
                        make_complex(15 * normfMod8, 7 * normfMod8),
                        make_complex(15 * normfMod8, 3 * normfMod8),
                        make_complex(15 * normfMod8, 1 * normfMod8),
                        make_complex(15 * normfMod8, 11 * normfMod8),
                        make_complex(15 * normfMod8, 9 * normfMod8),
                        make_complex(15 * normfMod8, 13 * normfMod8),
                        make_complex(15 * normfMod8, 15 * normfMod8),
                    };
                    CUDA_CHECK(cudaMemcpyToSymbol(complexConsMod8, complexConsMod8_host, sizeof(complex_type) * 256));
                    complexConsInitMod8 = true;
                }
                this->ConSize = 256;
                this->bitLength = 8;

                this->Cons = complexConsMod8;
                this->bitCons = complexBitConsMod8;
                break;
            }
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&this->TxIndice), sizeof(unsigned int) * this->TxAntNum));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&this->TxSymbols), sizeof(valueType) * this->TxAntNum));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&this->RxSymbols), sizeof(valueType) * this->RxAntNum));

            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&this->H), sizeof(valueType) * this->TxAntNum * this->RxAntNum));
        }
    }

    void generateChannel()
    {
        if constexpr (rc == RC::real)
        {
            Sebas<data_type>::getInstance().normalDistribution(this->H, this->TxAntNum * this->RxAntNum2, 0, std::sqrt(0.5));

        }
        else
        {

        }
    }
};

