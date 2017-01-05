#include "Particle_kernel.cuh"

void calculateParticles(float * particles, float dt, int MESH_VERTEX_COUNT, int blocks, int threads)
{
	calculateParticle <<< blocks, threads >>> (particles, dt);
}

__global__ void calculateParticle(float *particles, float dt)
{
/*	vec3 oldpos;
	//this->hash->generate_hashtable(this->data);
	for (int i = 0; i < MESH_VERTEX_COUNT; i++)
	{
		data[i].velocity = data[i].velocity + this->g * dt;
		oldpos = data[i].position;
		data[i].position = data[i].position + data[i].velocity * dt;

		//	this->doubleDensityRelaxation();

		data[i].velocity = (data[i].position - oldpos) / dt;

		if (data[i].position.y < -1.2)
			data[i].velocity.y = fabs(data[i].velocity.y);
	}
	for (int i = 0; i < MESH_VERTEX_COUNT; i++)
	{
		triangleMesh[6 * i + 0] = data[i].position.x;
		triangleMesh[6 * i + 1] = data[i].position.y;
		triangleMesh[6 * i + 2] = data[i].position.z;
	}*/
}
