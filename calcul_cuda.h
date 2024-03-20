#ifndef __CALCUL_CUDA_H__
#define __CALCUL_CUDA_H__

#include <cuda.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

void matrixMulOnDevice(float * matriceResult, float * matriceLeft, float * matriceRight, int width);

#ifdef __cplusplus
}
#endif

#endif
