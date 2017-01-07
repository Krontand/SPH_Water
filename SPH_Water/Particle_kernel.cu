#include "Particle_kernel.cuh"
#include "Logger.h"
void calculateParticles(float *particles, ParticleHashTable hash, ParticleData *data, 
						float dt, int MESH_VERTEX_COUNT, 
						int blocks, int threads)
{
//	LOG_DEBUG("%d  %d\n", blocks, threads);
	clear_table<<<blocks, threads>>>(hash);
	generate_hashtable<<<blocks, threads>>>(hash, data);
	calculateParticle<<<blocks, threads>>>(particles, data, hash, dt, MESH_VERTEX_COUNT);
}

__global__ void calculateParticle(float *particles, ParticleData *data, ParticleHashTable hash, float dt, int count)
{
	vec3 oldpos;
	const vec3 g = vec3(0.0, -9.81, 0.0);
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < count)
	{
		//this->hash->generate_hashtable(this->data);
		data[i].velocity = data[i].velocity + g * dt;
		oldpos = data[i].position;
		data[i].position = data[i].position + data[i].velocity * dt;

		//	this->doubleDensityRelaxation();

		data[i].velocity = (data[i].position - oldpos) / dt;

		if (data[i].position.y < -1.2)
			data[i].velocity.y = fabs(data[i].velocity.y);

		particles[6 * i + 0] = data[i].position.x;
		particles[6 * i + 1] = data[i].position.y;
		particles[6 * i + 2] = data[i].position.z;
	}
}
