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
	LoggerWrite("\n\n���������� � GPU\n");
	LoggerWrite("���: %s\n", prop.name);
	LoggerWrite("�������������� �����������: %d.%d\n", prop.major, prop.minor);
	LoggerWrite("�������� �������: %.2f���\n", prop.clockRate/1000.0);
	LoggerWrite("���������� ������: %d��\n", prop.totalGlobalMem/1048576);
	LoggerWrite("����������� ������: %d��\n", prop.totalConstMem/1024);
	LoggerWrite("Memory clock rate: %d���\n", prop.memoryClockRate / 1024);
	LoggerWrite("���������� �����������������: %ld\n", prop.multiProcessorCount);
	LoggerWrite("����������� ������ �� ���� ��: %d��\n", prop.sharedMemPerBlock/1024);
	LoggerWrite("��������� �� ���� �� (32-bit registers available): %d\n", prop.regsPerBlock);
	LoggerWrite("����. ���������� �������(threads) � �����: %d\n", prop.maxThreadsPerBlock);
	LoggerWrite("����. ���������� ������� �� ����������: (%d, %d, %d)\n",
		prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
	LoggerWrite("����. ������� �����: (%d, %d, %d)\n\n",
		prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
}

