#pragma once

#include <stdlib.h>
#include "Logger.h"

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


static void HandleError(cudaError_t err,
	const char *file,
	int line) {
	if (err != cudaSuccess) {
		LOG_ERROR("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(1);
	}
}

void CUDA_init()
{
	cudaDeviceProp prop;
	int dev;
	memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 1;
	prop.minor = 0;
	HANDLE_ERROR(cudaChooseDevice(&dev, &prop));
}

