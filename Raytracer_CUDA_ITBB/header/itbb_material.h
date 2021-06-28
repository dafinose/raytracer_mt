#pragma once

struct itbb_hit_record;

#include "ray.cuh"
#include "itbb_hitable.h"
#include "util.h"


__device__ inline float itbb_schlick(float cosine, float ref_idx) {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
}

__device__ inline bool itbb_refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted) {
    vec3 uv = unit_vector(v);
    float dt = dot(uv, n);
    float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1 - dt * dt);
    if (discriminant > 0) {
        refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
        return true;
    }
    else
        return false;
}

__device__ inline vec3 itbb_reflect(const vec3& v, const vec3& n) {
    return v - 2.0f * dot(v, n) * n;
}

class itbb_material {
public:
    __device__ virtual bool scatter(const ray& r_in, const itbb_hit_record& rec, vec3& attenuation, ray& scattered) const = 0;
};

class lambertian : public itbb_material {
public:
    __device__ lambertian(const vec3& a) : albedo(a) {}
    __device__ virtual bool scatter(const ray& r_in, const itbb_hit_record& rec, vec3& attenuation, ray& scattered) const {
        vec3 target = rec.p + rec.normal + random_in_unit_sphere();
        scattered = ray(rec.p, target - rec.p);
        attenuation = albedo;
        return true;
    }

    vec3 albedo;
};

class metal : public itbb_material {
public:
    __device__ metal(const vec3& a, float f) : albedo(a) { if (f < 1) fuzz = f; else fuzz = 1; }
    __device__ virtual bool scatter(const ray& r_in, const itbb_hit_record& rec, vec3& attenuation, ray& scattered) const {
        vec3 reflected = itbb_reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere());
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0.0f);
    }
    vec3 albedo;
    float fuzz;
};

class dielectric : public itbb_material {
public:
    __device__ dielectric(float ri) : ref_idx(ri) {}
    __device__ virtual bool scatter(const ray& r_in,
        const itbb_hit_record& rec,
        vec3& attenuation,
        ray& scattered) const {
        vec3 outward_normal;
        vec3 reflected = itbb_reflect(r_in.direction(), rec.normal);
        float ni_over_nt;
        attenuation = vec3(1.0, 1.0, 1.0);
        vec3 refracted;
        float reflect_prob;
        float cosine;
        if (dot(r_in.direction(), rec.normal) > 0.0f) {
            outward_normal = -rec.normal;
            ni_over_nt = ref_idx;
            cosine = dot(r_in.direction(), rec.normal) / r_in.direction().length();
            cosine = sqrt(1.0f - ref_idx * ref_idx * (1 - cosine * cosine));
        }
        else {
            outward_normal = rec.normal;
            ni_over_nt = 1.0f / ref_idx;
            cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
        }
        if (itbb_refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
            reflect_prob = itbb_schlick(cosine, ref_idx);
        else
            reflect_prob = 1.0f;
        if (rand_uniform() < reflect_prob)
            scattered = ray(rec.p, reflected);
        else
            scattered = ray(rec.p, refracted);
        return true;
    }

    float ref_idx;
};