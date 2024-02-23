#pragma once

#define PRECISION 0 // 0 for float, 1 for double
#define MAX_CURAND_STATE 256*256 // preallocate curand states

#if PRECISION == 0
    using data_type=float;
    using complex_type = cuFloatComplex;
    #define make_complex make_cuComplex
#elif PRECISION == 1
    using data_type=double;
    using complex_type = cuDoubleComplex;
    #define make_complex make_cuDoubleComplex
#endif
