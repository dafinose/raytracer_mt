#pragma once

#include "itbb_hitable.h"

class itbb_sphere : public itbb_hitable {
public:
	__device__ inline itbb_sphere() {}
	__device__ inline itbb_sphere(vec3 cen, float r, itbb_material* m) : center(cen), radius(r), mat_ptr(m) {};
	__device__ virtual inline bool hit(const ray& r, float t_min, float t_max, itbb_hit_record& rec) const;
	vec3 center;
	float radius;
	itbb_material* mat_ptr;
};

__device__ bool itbb_sphere::hit(const ray& r, float t_min, float t_max, itbb_hit_record& rec) const
{
	vec3 oc = r.origin() - center;
	float a = dot(r.direction(), r.direction());
	float b = dot(oc, r.direction());
	float c = dot(oc, oc) - radius * radius;
	float discriminant = b * b - a * c;
	if (discriminant > 0)
	{
		float temp = (-b - sqrt(discriminant)) / a;
		if (temp < t_max && temp > t_min)
		{
			rec.t = temp;
			rec.p = r.point_at_parameter(rec.t);
			rec.normal = (rec.p - center) / radius;
			rec.mat_ptr = mat_ptr;
			return true;
		}
		temp = (-b + sqrt(discriminant)) / a;
		if (temp < t_max && temp > t_min)
		{
			rec.t = temp;
			rec.p = r.point_at_parameter(rec.t);
			rec.normal = (rec.p - center) / radius;
			rec.mat_ptr = mat_ptr;
			return true;
		}
	}
	return false;
}