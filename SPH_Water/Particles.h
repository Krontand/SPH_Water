#pragma once

#include <vector>
#include "cuda_runtime.h"
#include "math/mathgl.h"
#include "Particle_kernel.cuh"
#include "ParticleHashTable.cuh"
#include "CUDA_core.cuh"


typedef std::vector<ParticleData> ParticleList;

class Particles
{
public:
	Particles();
	~Particles();

	void update_particles(float *particles, float dt);

	// ���������� ������
	const static int MESH_VERTEX_COUNT = 186624;

	// ������ ��������� � ������ ��� Opengl
	float *triangleMesh;
	ParticleData data[MESH_VERTEX_COUNT];

private:
	ParticleHashTable hash;
	ParticleData *dev_data;

	int xcount;
	int ycount;
	int zcount;
	float scale;
	float h;

	void doubleDensityRelaxation();

	const vec3 g = vec3(0.0, -9.81, 0.0);
};

