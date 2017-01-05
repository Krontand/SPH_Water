#pragma once

#include "cuda_runtime.h"

void calculateParticles(float *particles, float dt, int MESH_VERTEX_COUNT, int blocks, int threads);

__global__ void calculateParticle(float *particles, float dt);
