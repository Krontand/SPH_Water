#pragma once

#include <vector>
#include "cuda_runtime.h"
#include "math/mathgl.h"
#include "Particle_kernel.cuh"

typedef struct
{
	vec3 velocity;
	vec3 position;
} ParticleData;

typedef std::vector<ParticleData> ParticleList;

class Particles
{
public:
	Particles();
	~Particles();

	void update_particles(float *particles, float dt);

	// количество частиц
	const static int MESH_VERTEX_COUNT = 186624;

	// массив координат и цветов для Opengl
	float *triangleMesh;
	ParticleData data[MESH_VERTEX_COUNT];

private:
	int xcount;
	int ycount;
	int zcount;
	float scale;
	float h;

	void doubleDensityRelaxation();

	const vec3 g = vec3(0.0, -9.81, 0.0);
};

