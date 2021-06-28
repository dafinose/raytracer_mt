#pragma once

#include "ray.cuh"

class itbb_material;

struct itbb_hit_record
{
	float t;
	vec3 p;
	vec3 normal;
	itbb_material* mat_ptr;
};

class itbb_hitable
{
public:
	__device__ virtual bool hit(const ray& r, float t_min, float t_max, itbb_hit_record& rec) const = 0;
};