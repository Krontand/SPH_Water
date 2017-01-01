#pragma once

#include <vector>
#include "math/vec3.h"
#include <omp.h>
#include "Logger.h"

typedef int* Entry;

typedef struct
{
	vec3 velocity;
	vec3 position;
} ParticleData;

typedef std::vector<ParticleData> ParticleList;

class ParticleHashTable
{
public:
	ParticleHashTable(size_t size, float step);
	~ParticleHashTable();

	int insert(int index, vec3 position);
	int get_neighbours(int *neighbours, vec3 position, float h);
	int generate_hashtable(ParticleList &data);

private:
	void clear_table();
	int hash_func(int x, int y, int z);

	float step;
	float step1;
	size_t size;
	Entry *table;
	const int depth = 4;

	// Простые числа для хеш-функции
	const int a = 73856093;
	const int b = 19349663;
	const int c = 83492791;
};

