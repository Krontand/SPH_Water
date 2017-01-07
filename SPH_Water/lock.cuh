#pragma once

#include "CUDA_core.cuh"
#include "cuda_runtime.h"

struct Lock
{
	int *mutex;
	Lock();
	~Lock();

	__device__ void lock();
	__device__ void unlock();

};


