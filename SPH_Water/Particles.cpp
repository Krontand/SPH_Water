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
				i++;
				buf.position.x = triangleMesh[6 * i + 0];
				buf.position.y = triangleMesh[6 * i + 1];
				buf.position.z = triangleMesh[6 * i + 2];
				buf.velocity.set(0, 0, 0);
				this->data.push_back(buf);
			}
		}
	}
}


Particles::~Particles()
{
	delete triangleMesh;
}

void Particles::update_particles()
{
}
