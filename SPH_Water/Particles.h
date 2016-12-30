#pragma once
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

private:
	int xcount;
	int ycount;
	int zcount;
	float scale;
};

