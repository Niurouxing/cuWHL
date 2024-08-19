#pragma once
#include <cuComplex.h>
#include <stddef.h>

struct RD {};
struct CD {};

template<int N, typename T = RD>
struct QAM;

static constexpr __constant__ float QAM16RD[4] = {-0.31622776601683794, -0.9486832980505138, 0.31622776601683794, 0.9486832980505138};

// QAM specializations
template<>
struct QAM<16, RD>
{
    using type = float;
    static constexpr int ConSize = 4;
    static constexpr int bitLength = 2;
    static constexpr auto* Cons = QAM16RD;
};

static constexpr __constant__ float QAM64RD[8] = {-0.4629100498862757, -0.1543033499620919, -0.7715167498104595, -1.0801234497346432, 0.1543033499620919, 0.4629100498862757, 0.7715167498104595, 1.0801234497346432};

template<>
struct QAM<64, RD>
{
    using type = float;
    static constexpr int ConSize = 8;
    static constexpr int bitLength = 3;
    static constexpr auto* Cons = QAM64RD;
};

static constexpr __constant__ cuComplex QAM256RD[16] = {-0.3834824944236852, -0.5368754921931592, -0.2300894966542111, -0.07669649888473704, -0.8436614877321074, -0.6902684899626333, -0.9970544855015815, -1.1504474832710556, 0.3834824944236852, 0.5368754921931592, 0.2300894966542111, 0.07669649888473704, 0.8436614877321074, 0.6902684899626333, 0.9970544855015815, 1.1504474832710556};

template<>
struct QAM<256, RD>
{
    using type = cuComplex;
    static constexpr int ConSize = 16;
    static constexpr int bitLength = 4;
    static constexpr auto* Cons = QAM256RD;
};
