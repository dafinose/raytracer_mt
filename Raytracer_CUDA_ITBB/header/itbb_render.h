#pragma once

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include "vec3.cuh"

#include "../header/ray.cuh"
#include <time.h>
#include "../header/itbb_sphere.h"
#include "../header/itbb_hitable_list.h"
#include "../header/itbb_camera.h"
#include "../header/itbb_material.h"

using namespace tbb;

struct Render_Pixel
{
	vec3* fb;
	int width;
	int height;
	int ns;
	itbb_camera** cam;
	itbb_hitable** world;

	void operator()(const blocked_range<size_t>& range) const;
};

int itbb_main(void** fb);
