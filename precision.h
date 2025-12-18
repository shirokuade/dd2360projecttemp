#ifndef PRECISION_H
#define PRECISION_H
//==============================================================================================
// Precision configuration for ray tracer
// Define USE_DOUBLE_PRECISION before including any headers to control precision
// Default: true (double precision for maximum accuracy)
//==============================================================================================

#ifndef USE_DOUBLE_PRECISION
#define USE_DOUBLE_PRECISION true  // Default to double for backward compatibility
#endif

#if USE_DOUBLE_PRECISION
    typedef double real_t;
    #define REAL_CONST(x) x
    #define REAL_EPSILON 1e-8
    #define REAL_INF 1e30
    #define REAL_SQRT(x) sqrt(x)
    #define REAL_FABS(x) fabs(x)
    #define REAL_FMIN(x, y) fmin(x, y)
    #define REAL_FMAX(x, y) fmax(x, y)
#else
    typedef float real_t;
    #define REAL_CONST(x) x##f
    #define REAL_EPSILON 1e-6f
    #define REAL_INF 1e30f
    #define REAL_SQRT(x) sqrtf(x)
    #define REAL_FABS(x) fabsf(x)
    #define REAL_FMIN(x, y) fminf(x, y)
    #define REAL_FMAX(x, y) fmaxf(x, y)
#endif

#endif
