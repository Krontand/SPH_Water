#pragma once

#include "cuda_runtime.h"
#include "vec3.cuh"

typedef struct
{
	vec3 velocity;
	vec3 position;
} ParticleData;

void calculateParticles(float *particles, ParticleData *data, 
						float dt, int MESH_VERTEX_COUNT, 
						int blocks, int threads);

__global__ void calculateParticle(float *particles, ParticleData *data, float dt, int count);
