#pragma once

#include <vector>
#include "math/vec3.h"

typedef struct
{
	vec3 velocity;
	vec3 position;
} ParticleData;

class Particles
{
public:
	Particles();
	~Particles();

	void update_particles();

	// количество вершин в нашей геометрии, у нас простой треугольник
	int MESH_VERTEX_COUNT = 125000;

	// подготовим данные для вывода треугольника, всего 3 вершины
	float *triangleMesh;
	std::vector<ParticleData> data;

private:
	int xcount;
	int ycount;
	int zcount;
	float scale;

	const vec3 g = vec3(0.0, 9.81, 0.0);
};

