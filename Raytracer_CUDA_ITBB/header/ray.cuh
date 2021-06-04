#pragma once
#include "vec3.cuh"

class ray
{
public:
	__device__ ray() {}
	__device__ ray(const vec3& a, const vec3& b) { A = a; B = b; }
	__device__ vec3 origin() const { return a; }
	__device__ vec3 direction() const { return b; }
	__device__ vec3 point_at_parameter(float t) const { return A + t * b; }

	vec3 A;
	vec3 B;
};