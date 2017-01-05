#pragma once

#include <stdlib.h>
#include <Windows.h>
#include "Logger.h"
#include "cuda_runtime.h"
#include "cuda_gl_interop.h"

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

void HandleError(cudaError_t err, const char *file, int line);

void CUDA_init();
