#pragma once

#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "device_launch_parameters.h"
#include "vec3.cuh"


int cuda_main();
__global__ void render(float* fb, int max_x, int max_y);
