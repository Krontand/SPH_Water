#include "stdafx.h"
#include "Particles.h"


Particles::Particles()
{
	triangleMesh = new float[MESH_VERTEX_COUNT * 6];
	int i = 0;

	xcount = 72;
	ycount = 36;
	zcount = 72;
	scale = 1.6;

	h = 1 / xcount;

	ParticleData buf;
	buf.velocity.set(0, 0, 0);

	for (int x = 0; x < xcount; x++)
	{
		for (int y = 0; y < ycount; y++)
		{
			for (int z = 0; z < zcount; z++)
			{
				triangleMesh[6 * i + 0] = scale * (x / (float)xcount - 0.5);
				triangleMesh[6 * i + 1] = scale * (y / (float)ycount - 0.5);
				triangleMesh[6 * i + 2] = scale * (z / (float)zcount - 0.5);
				triangleMesh[6 * i + 3] = 0.0;
				triangleMesh[6 * i + 4] = 0.4;
				triangleMesh[6 * i + 5] = 1.0;
				buf.position.x = triangleMesh[6 * i + 0];
				buf.position.y = triangleMesh[6 * i + 1];
				buf.position.z = triangleMesh[6 * i + 2];
				data[i] = buf;
				i++;
			}
		}
	}
	HANDLE_ERROR(cudaMalloc((void**)&dev_data, MESH_VERTEX_COUNT * 6 * sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(dev_data, 
							data, 
							MESH_VERTEX_COUNT * 6 * sizeof(float),
							cudaMemcpyHostToDevice));

	init_table(hash, 186629, h);
}


Particles::~Particles()
{
	delete triangleMesh;
}

void Particles::update_particles(float *particles, float dt)
{
	dt /= 5;
	const int threads = 512;
	int blocks = (MESH_VERTEX_COUNT + 511) / 512;

	calculateParticles(particles, this->hash, this->dev_data, dt, MESH_VERTEX_COUNT, blocks, threads);
}

void Particles::doubleDensityRelaxation()
{
/*	int *nbours = new int[hash->depth() * 50];
	int num;
	float ro;
	float ro_near;
	for (auto particle = this->data.begin(); particle != this->data.end(); particle++)
	{
		ro = 0;
		ro_near = 0;
		num = hash->get_neighbours(nbours, particle->position, this->h);
	}*/
}


/*
Algorithm 1: Simulation step.
1. foreach particle i
2. // apply gravity
3. vi ← vi + ∆tg
4. // modify velocities with pairwise viscosity impulses
5. applyViscosity // (Section 5.3)
6. foreach particle i
7. // save previous position
8. xprevi ← xi
9. // advance to predicted position
10. xi ← xi + ∆tvi
11. // add and remove springs, change rest lengths
12. adjustSprings // (Section 5.2)
13. // modify positions according to springs,
14. // double density relaxation, and collisions
15. applySpringDisplacements // (Section 5.1)
16. doubleDensityRelaxation // (Section 4)
17. resolveCollisions // (Section 6)
18. foreach particle i
19. // use previous position to compute next velocity
20. vi ←(xi − xprevi) / ∆t

Particle-based Viscoelastic Fluid Simulation
http://www.ligum.umontreal.ca/Clavet-2005-PVFS/pvfs.pdf

*/