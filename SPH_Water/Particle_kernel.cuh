#pragma once

#include "cuda_runtime.h"
#include "ParticleHashTable.cuh"
#include "vec3.cuh"

void calculateParticles(float *particles, ParticleHashTable hash, ParticleData *data,
						float dt, int MESH_VERTEX_COUNT, int blocks, int threads);

__global__ void calculateParticle(float *particles, ParticleData *data, 
							      ParticleHashTable hash, float dt, int count);
