#pragma once

#include "CUDA_core.cuh"

void HandleError(cudaError_t err, const char *file,	int line) 
{
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
	prop.major = 3;
	prop.minor = 0;
	HANDLE_ERROR(cudaChooseDevice(&dev, &prop));
	HANDLE_ERROR(cudaGLSetGLDevice(dev));

	HANDLE_ERROR(cudaGetDeviceProperties(&prop, dev));
	LoggerWrite("\n\nИнформация о GPU\n");
	LoggerWrite("Имя: %s\n", prop.name);
	LoggerWrite("Вычислительные возможности: %d.%d\n", prop.major, prop.minor);
	LoggerWrite("Тактовая частота: %.2fМГц\n", prop.clockRate/1000.0);
	LoggerWrite("Глобальной памяти: %dМБ\n", prop.totalGlobalMem/1048576);
	LoggerWrite("Константной памяти: %dКб\n", prop.totalConstMem/1024);
	LoggerWrite("Memory clock rate: %dМГц\n", prop.memoryClockRate / 1024);
	LoggerWrite("Количество мультипроцессоров: %ld\n", prop.multiProcessorCount);
	LoggerWrite("Разделяемая память на один МП: %dКб\n", prop.sharedMemPerBlock/1024);
	LoggerWrite("Регистров на один МП (32-bit registers available): %d\n", prop.regsPerBlock);
	LoggerWrite("Макс. количество потоков(threads) в блоке: %d\n", prop.maxThreadsPerBlock);
	LoggerWrite("Макс. количество потоков по измерениям: (%d, %d, %d)\n",
		prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
	LoggerWrite("Макс. размеры сетки: (%d, %d, %d)\n\n",
		prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
}

