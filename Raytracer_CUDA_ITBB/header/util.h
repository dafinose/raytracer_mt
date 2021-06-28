#pragma once

#include "vec3.cuh"

inline float rand_uniform()
{
    float r = (float)rand() / (float)RAND_MAX;
    return r;
}

inline vec3 random_in_unit_disk() {
    vec3 p;
    do {
        p = 2.0f * vec3(rand_uniform(), rand_uniform(), 0) - vec3(1, 1, 0);
    } while (dot(p, p) >= 1.0f);
    return p;
}

inline vec3 random_in_unit_sphere() {
    vec3 p;
    do {
        p = 2.0f * vec3(rand_uniform(), rand_uniform(), rand_uniform()) - vec3(1, 1, 1);
    } while (p.squared_length() >= 1.0f);
    return p;
}