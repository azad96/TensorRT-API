#ifndef LOGS_H
#define LOGS_H

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

/**
 * Execute a CUDA call and print out any errors
 * @return the original cudaError_t result
 * @ingroup util
 */
#define CUDA(x)				cudaCheckError((x), #x, __FILE__, __LINE__)

/**
 * LOG_CUDA string.
 * @ingroup util
 */
#define LOG_CUDA "[cuda]   "


inline cudaError_t cudaCheckError(cudaError_t retval, const char* txt, const char* file, int line )
{
#if !defined(CUDA_TRACE)
    if( retval == cudaSuccess)
        return cudaSuccess;
#endif

    printf(LOG_CUDA "%s\n", txt);

    if( retval != cudaSuccess )
    {
        printf(LOG_CUDA "   %s (error %u) (hex 0x%02X)\n", cudaGetErrorString(retval), retval, retval);
        printf(LOG_CUDA "   %s:%i\n", file, line);
    }

    return retval;
}

#endif