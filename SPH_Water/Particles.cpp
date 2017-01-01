#include "stdafx.h"
#include "Particles.h"


Particles::Particles()
{
	triangleMesh = new float[MESH_VERTEX_COUNT * 6];
	int i = 0;

	xcount = 50;
	ycount = 50;
	zcount = 50;
	scale = 1.3;

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
				i++;
				this->data.push_back(buf);
			}
		}
	}

	this->hash = new ParticleHashTable(125003, 0.02);
}


Particles::~Particles()
{
	delete triangleMesh;
}

void Particles::update_particles(float dt)
{
	dt /= 5;
	vec3 oldpos;
	this->hash->generate_hashtable(this->data);
	for (auto particle = this->data.begin(); particle != this->data.end(); particle++)
	{
		particle->velocity = particle->velocity + this->g * dt;
		oldpos = particle->position;
		particle->position = particle->position + particle->velocity * dt;

		//this->doubleDensityRelaxation();

		particle->velocity = (particle->position - oldpos) / dt;

		if (particle->position.y < -1.2)
			particle->velocity.y = fabs(particle->velocity.y);
	}
	for (int i = 0; i < MESH_VERTEX_COUNT; i++)
	{
		triangleMesh[6 * i + 0] = data[i].position.x;
		triangleMesh[6 * i + 1] = data[i].position.y;
		triangleMesh[6 * i + 2] = data[i].position.z;
	}
}

void Particles::doubleDensityRelaxation()
{
	float ro;
	float ro_near;
	for (auto particle = this->data.begin(); particle != this->data.end(); particle++)
	{
		ro = 0;
		ro_near = 0;
	}
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