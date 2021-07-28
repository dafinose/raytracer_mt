#pragma once

#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "device_launch_parameters.h"
#include "vec3.cuh"
#include <vector_functions.h>


#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line);

int cuda_main(cudaGraphicsResource* resource, int x, int y, int samples);
__global__ void render(float* fb, int max_x, int max_y);

