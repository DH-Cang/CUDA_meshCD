#pragma once
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void vectorAddition(const float* a, const float* b, float* c, int size);

int here();