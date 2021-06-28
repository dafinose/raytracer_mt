#include "../header/itbb_render.h"

__device__ vec3 color(const ray& r, itbb_hitable** world) {
	ray cur_ray = r;
	vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
	for (int i = 0; i < 50; i++) {
		itbb_hit_record rec;
		if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
			ray scattered;
			vec3 attenuation;
			if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered)) {
				cur_attenuation *= attenuation;
				cur_ray = scattered;
			}
			else {
				return vec3(0.0, 0.0, 0.0);
			}
		}
		else {
			vec3 unit_direction = unit_vector(cur_ray.direction());
			float t = 0.5f * (unit_direction.y() + 1.0f);
			vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
			return cur_attenuation * c;
		}
	}
	return vec3(0.0, 0.0, 0.0); // exceeded recursion
}

void Render_Pixel::operator()(const blocked_range<size_t>& range) const
{
	for (size_t pixel_index = range.begin(); pixel_index < range.end(); pixel_index++) {
		vec3 col(0, 0, 0);
		for (int s = 0; s < ns; s++) {
			// random bilateral
			int x = pixel_index % width;
			int y = pixel_index / width;
			float u = float(x + rand_uniform()) / float(width);
			float v = float(y + rand_uniform()) / float(height);
			ray r = (*cam)->get_ray(u, v);
			col += color(r, world);
		}
		col /= float(ns);
		col[0] = sqrt(col[0]);
		col[1] = sqrt(col[1]);
		col[2] = sqrt(col[2]);
		fb[pixel_index] = col;
	}
}


void createWorld(itbb_hitable** d_list, itbb_hitable** d_world, itbb_camera** d_camera, int nx, int ny) {

	//curandState local_rand_state = *rand_state;
	d_list[0] = new itbb_sphere(vec3(0, -1000.0, -1), 1000,
		new lambertian(vec3(0.5, 0.5, 0.5)));
	int i = 1;
	for (int a = -11; a < 11; a++) {
		for (int b = -11; b < 11; b++) {
			//float choose_mat = RND;
			float choose_mat = 0.0f;
			//vec3 center(a + RND, 0.2, b + RND);
			vec3 center(a, 0.2, b);
			if (choose_mat < 0.8f) {
				d_list[i++] = new itbb_sphere(center, 0.2,
					//new lambertian(vec3(RND * RND, RND * RND, RND * RND)));
					new lambertian(vec3(0.1f, 0.8f, 0.5f)));
			}
			else if (choose_mat < 0.95f) {
				d_list[i++] = new itbb_sphere(center, 0.2,
					//new metal(vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND));
					new metal(vec3(0.5f * (1.0f), 0.5f * (1.0f), 0.5f * (1.0f)), 0.5f));
			}
			else {
				d_list[i++] = new itbb_sphere(center, 0.2, new dielectric(1.5));
			}
		}
	}
	d_list[i++] = new itbb_sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
	d_list[i++] = new itbb_sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
	d_list[i++] = new itbb_sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
	//*rand_state = local_rand_state;
	*d_world = new itbb_hitable_list(d_list, 22 * 22 + 1 + 3);

	vec3 lookfrom(13, 2, 3);
	vec3 lookat(0, 0, 0);
	float dist_to_focus = 10.0; (lookfrom - lookat).length();
	float aperture = 0.1;
	*d_camera = new itbb_camera(lookfrom,
		lookat,
		vec3(0, 1, 0),
		30.0,
		float(nx) / float(ny),
		aperture,
		dist_to_focus);
}

void free_world(itbb_hitable** d_list, itbb_hitable** d_world, itbb_camera** d_camera) {
	for (int i = 0; i < 22 * 22 + 1 + 3; i++) {
		delete ((itbb_sphere*)d_list[i])->mat_ptr;
		delete d_list[i];
	}
	delete* d_world;
	delete* d_camera;
}

int itbb_main(void** pixeldata) 
{
	// Bilddimensionen
	int nx = 800;
	int ny = 600;
	int ns = 1;

	int num_pixels = nx * ny;
	// Größe des Frambuffers
	size_t fb_size = num_pixels * sizeof(vec3);

	vec3* fb;
	fb = (vec3*)malloc(fb_size);
	*pixeldata = fb;

	//// Variablen für die Szene
	itbb_hitable** list;
	int num_hitables = 22 * 22 + 1 + 3;
	list = (itbb_hitable**)malloc(num_hitables * sizeof(itbb_hitable*));
	//itbb_hitable** world;
	//itbb_camera** camera;

	Render_Pixel rp = {};

	rp.fb = fb;
	rp.width = nx;
	rp.height = ny;
	rp.ns = ns;
	rp.cam = (itbb_camera**)malloc(sizeof(itbb_camera*));
	rp.world = (itbb_hitable**)malloc(sizeof(itbb_hitable*));

	createWorld(list, rp.world, rp.cam, nx, ny);

	clock_t start, stop;
	start = clock();

	// render
	parallel_for(blocked_range<std::size_t>(0, num_pixels), rp);

	stop = clock();
	double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cerr << "took " << timer_seconds << " seconds.\n";

	// Bilddaten übergeben
	//free_world(list, rp.world, &rp.cam);
	//free(list);
	//free(rp.world);
	//free(rp.cam);
	//free(fb);

	return 0;
}