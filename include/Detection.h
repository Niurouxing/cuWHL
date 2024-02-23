#pragma once
#include <type_traits>
#include <stdexcept>
#include "utils.h"
#include "Cons.h"
#include <cmath>
#include "Sebas.h"
enum class RC
{
    real,
    complex
};

enum class ModType
{
    QAM4,
    QAM16,
    QAM64,
    QAM256
};

template <RC rc, ModType mod>
class Detection;

// ConditionallyIncluded Data struct
template <RC rc, ModType mod>
struct DetectionUtils
{
    // None for default
    friend class Detection<rc, mod>;
};

// special for real
template <ModType mod>
struct DetectionUtils<RC::real, mod>
{
    friend class Detection<RC::real, mod>;

public:
    using valueType = data_type;
    const valueType *Cons2;

    __device__ inline auto getCons()
    {
        if constexpr (mod == ModType::QAM4)
        {
            return realConsMod2;
        }
        else if constexpr (mod == ModType::QAM16)
        {
            return realConsMod4;
        }
        else if constexpr (mod == ModType::QAM64)
        {
            return realConsMod6;
        }
        else if constexpr (mod == ModType::QAM256)
        {
            return realConsMod8;
        }
    }

    __device__ inline auto getCons2()
    {
        if constexpr (mod == ModType::QAM4)
        {
            return realConsMod2;
        }
        else if constexpr (mod == ModType::QAM16)
        {
            return realConsMod4;
        }
        else if constexpr (mod == ModType::QAM64)
        {
            return realConsMod6;
        }
        else if constexpr (mod == ModType::QAM256)
        {
            return realConsMod8;
        }
    }

    __device__ inline auto getBitCons()
    {
        if constexpr (mod == ModType::QAM4)
        {
            return realBitConsMod2;
        }
        else if constexpr (mod == ModType::QAM16)
        {
            return realBitConsMod4;
        }
        else if constexpr (mod == ModType::QAM64)
        {
            return realBitConsMod6;
        }
        else if constexpr (mod == ModType::QAM256)
        {
            return realBitConsMod8;
        }
    }

    inline int getTxAntNum2()
    {
        return TxAntNum2;
    }

    inline int getRxAntNum2()
    {
        return RxAntNum2;
    }

private:
    int TxAntNum2;
    int RxAntNum2;
};

// special for complex
template <ModType mod>
struct DetectionUtils<RC::complex, mod>
{
    friend class Detection<RC::complex, mod>;

public:
    using valueType = std::conditional_t<std::is_same<data_type, float>::value, cuFloatComplex, cuDoubleComplex>;

    __device__ inline auto getCons()
    {
        if constexpr (mod == ModType::QAM4)
        {
            return complexConsMod2;
        }
        else if constexpr (mod == ModType::QAM16)
        {
            return complexConsMod4;
        }
        else if constexpr (mod == ModType::QAM64)
        {
            return complexConsMod6;
        }
        else if constexpr (mod == ModType::QAM256)
        {
            return complexConsMod8;
        }
    }

    __device__ inline auto getBitCons()
    {
        if constexpr (mod == ModType::QAM4)
        {
            return complexBitConsMod2;
        }
        else if constexpr (mod == ModType::QAM16)
        {
            return complexBitConsMod4;
        }
        else if constexpr (mod == ModType::QAM64)
        {
            return complexBitConsMod6;
        }
        else if constexpr (mod == ModType::QAM256)
        {
            return complexBitConsMod8;
        }
    }
};

template <RC rc, ModType mod>
class Detection : public DetectionUtils<rc, mod>
{
private:
    using valueType = typename DetectionUtils<rc, mod>::valueType;
    int TxAntNum;
    int RxAntNum;
    int ConSize;
    int bitLength;
    data_type SNRdB;
    data_type Nv;
    data_type sqrtNvDiv2;
    data_type NvInv;

    unsigned int *TxIndice;
    bool *TxBits;

    valueType *TxSymbols;
    valueType *RxSymbols;
    valueType *H;

public:
    inline int getTxAntNum()
    {
        return TxAntNum;
    }

    inline int getRxAntNum()
    {
        return RxAntNum;
    }

    inline int getConSize()
    {
        return ConSize;
    }

    inline int getBitLength()
    {
        return bitLength;
    }

    inline void setSNRdB(data_type dB)
    {
        SNRdB = dB;
        if constexpr (rc == RC::real)
        {
            Nv = static_cast<data_type>(TxAntNum * RxAntNum / (pow(10, SNRdB / 10) * this->bitLength * 2 * TxAntNum));
        }
        else
        {
            Nv = static_cast<data_type>(TxAntNum * RxAntNum / (pow(10, SNRdB / 10) * this->bitLength * TxAntNum));
        }

        sqrtNvDiv2 = std::sqrt(Nv / 2);
        NvInv = 1 / Nv;
    }

    inline data_type getSNRdB()
    {
        return SNRdB;
    }

    inline data_type getNv()
    {
        return Nv;
    }

    inline data_type getsqrtNvDiv2()
    {
        return sqrtNvDiv2;
    }

    inline data_type getNvInv()
    {
        return NvInv;
    }

    inline unsigned int *getTxIndice()
    {
        return TxIndice;
    }

    inline bool *getTxBits()
    {
        return TxBits;
    }

    inline valueType *getTxSymbols()
    {
        return TxSymbols;
    }

    inline valueType *getRxSymbols()
    {
        return RxSymbols;
    }

    inline valueType *getH()
    {
        return H;
    }

    Detection(int TxAntNum, int RxAntNum)
    {
        this->TxAntNum = TxAntNum;

        this->RxAntNum = RxAntNum;

        if constexpr (rc == RC::real)
        {
            this->TxAntNum2 = TxAntNum * 2;
            this->RxAntNum2 = RxAntNum * 2;

            if constexpr (mod == ModType::QAM4)
            {
                this->ConSize = 2;
                this->bitLength = 1;
            }
            else if constexpr (mod == ModType::QAM16)
            {
                this->ConSize = 4;
                this->bitLength = 2;
            }

            else if constexpr (mod == ModType::QAM64)
            {
                this->ConSize = 8;
                this->bitLength = 3;
            }
            else if constexpr (mod == ModType::QAM256)
            {
                this->ConSize = 16;
                this->bitLength = 4;
            }
            else
            {
                throw std::runtime_error("ModType not supported");
            }

            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&this->TxBits), sizeof(unsigned int) * this->TxAntNum * this->bitLength * 2));

            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&this->TxIndice), sizeof(unsigned int) * this->TxAntNum2));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&this->TxSymbols), sizeof(data_type) * this->TxAntNum2));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&this->RxSymbols), sizeof(data_type) * this->RxAntNum2));

            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&this->H), sizeof(data_type) * this->TxAntNum2 * this->RxAntNum2));
        }
        else
        {
            if constexpr (mod == ModType::QAM4)
            {

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
            }
            else if constexpr (mod == ModType::QAM16)

            {
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
            }

            else if constexpr (mod == ModType::QAM64)
            {
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
            }

            else if constexpr (mod == ModType::QAM256)
            {
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
            }

            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&this->TxBits), sizeof(unsigned int) * this->TxAntNum * this->bitLength));

            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&this->TxIndice), sizeof(unsigned int) * this->TxAntNum));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&this->TxSymbols), sizeof(valueType) * this->TxAntNum));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&this->RxSymbols), sizeof(valueType) * this->RxAntNum));

            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&this->H), sizeof(valueType) * this->TxAntNum * this->RxAntNum));
        }
    }

    void generateChannel()
    {
        static auto &sebas = Sebas::getInstance();
        if constexpr (rc == RC::real)
        {
            sebas.normalDistribution(this->H, this->TxAntNum * this->RxAntNum2, (float)0, (float)std::sqrt(0.5));
            sebas.RVD(this->H, this->RxAntNum, this->TxAntNum);
        }
        else
        {
            sebas.complexNormalDistribution(this->H, this->TxAntNum * this->RxAntNum, 0, std::sqrt(0.5));
        }
    }

    void generateTxSignals()
    {
        static auto &sebas = Sebas::getInstance();

        auto TxAntNumToUse = 0;
        if constexpr (rc == RC::real)
        {
            TxAntNumToUse = this->TxAntNum2;
        }
        else
        {
            TxAntNumToUse = this->TxAntNum;
        }
        sebas.uniformIntDistribution(this->TxIndice, TxAntNumToUse, 0, this->ConSize - 1);

        thrust::transform(thrust::device, this->TxIndice, this->TxIndice + TxAntNumToUse, this->TxSymbols, [this] __device__(unsigned int x)
                          { return this->getCons()[x]; });
    }
};
