#include "stdafx.h"
#include "ParticleHashTable.cuh"

Lock::Lock(int count)
{
	int *state = (int*)calloc(count, sizeof(int));
	HANDLE_ERROR(cudaMalloc((void**)&mutex, count * sizeof(int)));
	HANDLE_ERROR(cudaMemcpy(mutex, state, count * sizeof(int), cudaMemcpyHostToDevice));
	free(state);
}

Lock::~Lock()
{
	cudaFree(mutex);
}

__device__ void Lock::lock(int n)
{
	while (atomicCAS(&(mutex[n]), 0, 1) != 0);
}

__device__ void Lock::unlock(int n)
{
	atomicExch(&(mutex[n]), 0);
}



void init_table(ParticleHashTable table, size_t size, float step)
{
	table.step = step;
	table.step1 = 1 / step;
	table.count = size;

	HANDLE_ERROR(cudaMalloc((void**)&(table.entries), size * sizeof(Entry*)));
	HANDLE_ERROR(cudaMemset(table.entries, 0, size * sizeof(Entry*)));
	HANDLE_ERROR(cudaMalloc((void**)&(table.pool), size * sizeof(Entry)));

	Lock *_lock = new Lock(size);
	HANDLE_ERROR(cudaMalloc((void**)&(table.lock), sizeof(Lock)));
	HANDLE_ERROR(cudaMemcpy(table.lock, _lock, sizeof(Lock), cudaMemcpyHostToDevice));
	delete _lock;
}

__device__ void insert(ParticleHashTable table, int index, vec3 position)
{
	return;
}

__device__ int get_neighbours(ParticleHashTable table, int *neighbours, vec3 position, float h)
{
/*	int x = position.x / step;
	int y = position.y / step;
	int z = position.z / step;
	int ind;
	int num = 0;

	for (int i = x - 1; i < x + 2; i++)
		for (int j = y - 1; j < y + 2; j++)
			for (int k = z - 1; k < z + 2; k++)
			{
				ind = d * hash_func(i, j, k);
				int n = table[ind];
				for (int l = 1; l <= n; l++)
					neighbours[num++] = table[ind + 1];
			}
	return num;*/
	return 0;
}

__global__ void clear_table(ParticleHashTable table)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < table.count)
	{
		table.entries[tid]->key = 0;
		table.entries[tid]->next = 0;
		table.entries[tid]->value = 0;
	}
}

__device__ size_t hash_func(int x, int y, int z)
{
	return (x * 73856093 ^ y * 19349663 ^ z * 83492791);
}

__global__ void generate_hashtable(ParticleHashTable table, ParticleData *data)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < table.count)
	{
		unsigned int key = hash_func(data[tid].position.x * table.step1,
			data[tid].position.y * table.step1,
			data[tid].position.z * table.step1);
		size_t hashValue = key % table.count;
		for (int i = 0; i < 32; i++)
		{
			if ((tid % 32) == i)
			{
				Entry *location = &(table.pool[tid]);
				location->key = key;
				location->value = tid;
				table.lock->lock(hashValue);
				location->next = table.entries[hashValue];
				table.entries[hashValue] = location;
				table.lock->unlock(hashValue);
			}
		}
	}
}
