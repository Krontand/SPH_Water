#include "lock.cuh"

Lock::Lock()
{
	int state = 0;
	HANDLE_ERROR(cudaMalloc((void**)&mutex, sizeof(int)));
	HANDLE_ERROR(cudaMemcpy(mutex, &state, sizeof(int), cudaMemcpyHostToDevice));
}

Lock::~Lock()
{
	cudaFree(mutex);
}

__device__ void Lock::lock()
{
	while (atomicCAS(mutex, 0, 1) != 0);
}

__device__ void Lock::unlock()
{
	atomicExch(mutex, 0);
}
