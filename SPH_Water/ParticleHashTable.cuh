#pragma once

#include <vector>
#include "vec3.cuh"
#include "Logger.h"
#include "CUDA_core.cuh"

struct Lock
{
	int *mutex;
	Lock(int count);
	~Lock();

	__device__ void lock(int n);
	__device__ void unlock(int n);

};

struct Entry
{
	unsigned int key;
	int value;
	Entry *next;
};

typedef struct
{
	vec3 velocity;
	vec3 position;
} ParticleData;

struct ParticleHashTable
{
	float step;
	float step1;
	Lock *lock;
	size_t count;
	Entry **entries;
	Entry *pool;
	Entry *firstFree;
};

void init_table(ParticleHashTable table, size_t size, float step);

__device__ void insert(ParticleHashTable table, int index, vec3 position);
__device__ int get_neighbours(ParticleHashTable table, int *neighbours, vec3 position, float h);

__global__ void clear_table(ParticleHashTable table);
__device__ size_t hash_func(int x, int y, int z);

__global__ void generate_hashtable(ParticleHashTable table, ParticleData *data);