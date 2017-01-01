#include "stdafx.h"
#include "ParticleHashTable.h"


ParticleHashTable::ParticleHashTable(size_t size, float step)
{
	this->step = step;
	this->step1 = 1 / step;
	this->size = size;
	this->table = new Entry[size];

	for (int i = 0; i < size; i++)
	{
		table[i] = (int *)malloc((depth + 1) * sizeof(int));
		table[i][0] = 0;
	}
}

ParticleHashTable::~ParticleHashTable()
{
	delete this->table;
}

int ParticleHashTable::insert(int index, vec3 position)
{
	int ind = hash_func(position.x * step1, position.y * step1, position.z * step1);
	int &n = this->table[ind][0];
	if (n < depth)
	{
		n++;
		this->table[ind][n] = index;
	}
	return ind;
}

int ParticleHashTable::get_neighbours(int *neighbours, vec3 position, float h)
{
	int x = position.x / step;
	int y = position.y / step;
	int z = position.z / step;
	int ind;
	int num = 0;

	for (int i = x - 1; i < x + 2; i++)
		for (int j = y - 1; j < y + 2; j++)
			for (int k = z - 1; k < z + 2; k++)
			{
				ind = hash_func(i, j, k);
				int n = table[ind][0];
				for (int l = 1; l <= n; l++)
					neighbours[num++] = table[ind][l];
			}
	return num;
}

int ParticleHashTable::generate_hashtable(ParticleList &data)
{
	this->clear_table();
	int n = data.size();
	for (int i = 0; i < n; i++)
	{
		this->insert(i, data[i].position);
	}
	return 0;
}

void ParticleHashTable::clear_table()
{
	for (int i = 0; i < this->size; i++)
		this->table[i][0] = 0; 
}

int ParticleHashTable::hash_func(int x, int y, int z)
{
	return (x * this->a ^ y * this->b ^ z * this->c) % this->size;
}
