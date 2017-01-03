#pragma once

#include "ParticleHashTable.h"

class Particles
{
public:
	Particles();
	~Particles();

	void update_particles(float dt);

	// ���������� ������
	int MESH_VERTEX_COUNT = 23328;

	// ������ ��������� � ������ ��� Opengl
	float *triangleMesh;
	ParticleList data;

private:
	ParticleHashTable *hash;

	int xcount;
	int ycount;
	int zcount;
	float scale;
	float h;

	void doubleDensityRelaxation();

	const vec3 g = vec3(0.0, -9.81, 0.0);
};

