#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include "cuda_runtime.h"

struct vec2;
struct vec3;
struct vec4;
struct quat;
struct mat2;
struct mat3;
struct mat4;

struct vec3
{
	union
	{
		struct { float x, y, z; };
		float v[3];
	};

	__host__ __device__ vec3() {}

	__host__ __device__ void set(const float *f) { x = f[0];    y = f[1];    z = f[2]; }
	__host__ __device__ void set(float x, float y, float z) { this->x = x; this->y = y; this->z = z; }
	__host__ __device__ void set(const vec3 &v) { x = v.x;     y = v.y;     z = v.z; }

	__host__ __device__ void set(const vec2 &v);
	__host__ __device__ void set(const vec4 &v);

	__host__ __device__ vec3(const float *f) { set(f); }
	__host__ __device__ vec3(float x, float y, float z) { set(x, y, z); }
	__host__ __device__ vec3(const vec3 &v) { set(v); }
	__host__ __device__ vec3(const vec2 &v) { set(v); }
	__host__ __device__ vec3(const vec4 &v) { set(v); }

	__host__ __device__ vec3& operator=(const vec3 &v) { set(v); return *this; }
	__host__ __device__ vec3& operator=(const vec2 &v) { set(v); return *this; }
	__host__ __device__ vec3& operator=(const vec4 &v) { set(v); return *this; }

	__host__ __device__ float operator[](int i) { return v[i]; }
	__host__ __device__ float operator[](int i) const { return v[i]; }

	__host__ __device__ const vec3 operator-() const { return vec3(-x, -y, -z); }

	__host__ __device__ const vec3 operator+(const vec3 &v) const { return vec3(x + v.x, y + v.y, z + v.z); }
	__host__ __device__ const vec3 operator-(const vec3 &v) const { return vec3(x - v.x, y - v.y, z - v.z); }
	__host__ __device__ const vec3 operator*(float f)       const { return vec3(x * f, y * f, z * f); }
	__host__ __device__ const vec3 operator/(float f)       const { return vec3(x / f, y / f, z / f); }

	__host__ __device__ vec3& operator+=(const vec3 &v) { x += v.x; y += v.y; z += v.z; return *this; }
	__host__ __device__ vec3& operator-=(const vec3 &v) { x -= v.x; y -= v.y; z -= v.z; return *this; }
	__host__ __device__ vec3& operator*=(float f) { x *= f;   y *= f;   z *= f;   return *this; }
	__host__ __device__ vec3& operator/=(float f) { x /= f;   y /= f;   z /= f;   return *this; }
};

__host__ __device__ inline float dot(const vec3& v1, const vec3 &v2)
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

__host__ __device__ inline const vec3 cross(const vec3 &v1, const vec3 &v2)
{
	return vec3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}

__host__ __device__ inline float length(const vec3 &v)
{
	return sqrtf(dot(v, v));
}

__host__ __device__ inline float distance(const vec3 &v1, const vec3 &v2)
{
	return length(v1 - v2);
}

__host__ __device__ inline const vec3 normalize(const vec3 &v)
{
	return v / length(v);
}

#endif /* VEC3_H */
