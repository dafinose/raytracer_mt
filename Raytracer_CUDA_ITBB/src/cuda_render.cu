#include "../header/cuda_render.cuh"
#include "../header/ray.cuh"
#include <time.h>
#include "../header/sphere.cuh"
#include "../header/hitable_list.cuh"
#include "../header/camera.cuh"
#include "../header/material.cuh"


void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
	if (result)
	{
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
		cudaDeviceReset();
		exit(99);
	}
}

__device__ vec3 color(const ray& r, hitable** world, curandState* local_rand_state) {
	ray cur_ray = r;
	vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
	for (int i = 0; i < 50; i++) {
		hit_record rec;
		if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
			ray scattered;
			vec3 attenuation;
			if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
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

__global__ void rand_init(curandState* rand_state) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		curand_init(1984, 0, 0, rand_state);
	}
}

__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x + i;
	// Original: Each thread gets same seed, a different sequence number, no offset
	// curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
	curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3* fb, int max_x, int max_y, int ns, camera** cam, hitable** world, curandState* rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x + i;
	curandState local_rand_state = rand_state[pixel_index];
	vec3 col(0, 0, 0);
	for (int s = 0; s < ns; s++) {
		float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
		float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
		ray r = (*cam)->get_ray(u, v, &local_rand_state);
		col += color(r, world, &local_rand_state);
	}
	rand_state[pixel_index] = local_rand_state;
	col /= float(ns);
	col[0] = sqrt(col[0]);
	col[1] = sqrt(col[1]);
	col[2] = sqrt(col[2]);
	fb[pixel_index] = col;
}

#define RND (curand_uniform(&local_rand_state))

__global__ void createWorld(hitable** d_list, hitable** d_world, camera** d_camera, int nx, int ny, curandState* rand_state) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		//curandState local_rand_state = *rand_state;
		d_list[0] = new sphere(vec3(0, -1000.0, -1), 1000,
			new lambertian(vec3(0.5, 0.5, 0.5)));
		int i = 1;
		for (int a = -11; a < 11; a++) {
			for (int b = -11; b < 11; b++) {
				//float choose_mat = RND;
				float choose_mat = 0.0f;
				//vec3 center(a + RND, 0.2, b + RND);
				vec3 center(a, 0.2, b);
				if (choose_mat < 0.8f) {
					d_list[i++] = new sphere(center, 0.2,
						//new lambertian(vec3(RND * RND, RND * RND, RND * RND)));
						new lambertian(vec3(0.1f, 0.8f, 0.5f)));
				}
				else if (choose_mat < 0.95f) {
					d_list[i++] = new sphere(center, 0.2,
						//new metal(vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND));
						new metal(vec3(0.5f * (1.0f), 0.5f * (1.0f), 0.5f * (1.0f)), 0.5f));
				}
				else {
					d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
				}
			}
		}
		d_list[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
		d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
		d_list[i++] = new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
		//*rand_state = local_rand_state;
		*d_world = new hitable_list(d_list, 22 * 22 + 1 + 3);

		vec3 lookfrom(13, 2, 3);
		vec3 lookat(0, 0, 0);
		float dist_to_focus = 10.0; (lookfrom - lookat).length();
		float aperture = 0.1;
		*d_camera = new camera(lookfrom,
			lookat,
			vec3(0, 1, 0),
			30.0,
			float(nx) / float(ny),
			aperture,
			dist_to_focus);
	}
}

__global__ void free_world(hitable** d_list, hitable** d_world, camera** d_camera) {
	for (int i = 0; i < 22 * 22 + 1 + 3; i++) {
		delete ((sphere*)d_list[i])->mat_ptr;
		delete d_list[i];
	}
	delete* d_world;
	delete* d_camera;
}

int cuda_main(cudaGraphicsResource* resource)
{
	// Bilddimensionen
	int nx = 1920;
	int ny = 1080;
	int ns = 10;

	// Threaddimensionen für CUDA
	int tx = 32;
	int ty = 32;

	std::cerr << "Rendering a " << nx << "x" << ny << " image ";
	std::cerr << "in " << tx << "x" << ty << " blocks.\n";

	int num_pixels = nx * ny;
	// Größe des Frambuffers
	size_t fb_size = num_pixels * sizeof(vec3);

	// Framebuffer allokieren
	vec3* fb;
	checkCudaErrors(cudaMalloc((void**)&fb, fb_size));

	// allocate random state
	curandState* d_rand_state;
	checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));
	curandState* d_rand_state2;
	checkCudaErrors(cudaMalloc((void**)&d_rand_state2, 1 * sizeof(curandState)));

	// Variablen für die Szene
	hitable** d_list;
	int num_hitables = 22 * 22 + 1 + 3;
	checkCudaErrors(cudaMalloc((void**)&d_list, num_hitables * sizeof(hitable*)));
	hitable** d_world;
	checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hitable*)));
	camera** d_camera;
	checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));
	createWorld<<<1, 1>>>(d_list, d_world, d_camera, nx, ny, d_rand_state2);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	//texture<float, 2> texRef;

	clock_t start, stop;
	start = clock();

	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty);

	render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	render<<<blocks, threads>>>(fb, nx, ny, ns, d_camera, d_world, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	stop = clock();
	double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cerr << "took " << timer_seconds << " seconds.\n";

	cudaArray* fb_dev_array;
	vec3* device_pointer;
	// Bild uebergeben
	//size_t fb_size_t = num_pixels * sizeof(vec3);
	checkCudaErrors(cudaGraphicsMapResources(1, &resource, 0));
	//checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&fb_dev_array, resource, 0, 0));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**) &device_pointer , &fb_size, resource));

	checkCudaErrors(cudaMemcpy(device_pointer, fb, fb_size, cudaMemcpyDeviceToDevice));
	
	//checkCudaErrors(cudaMemcpy2DToArray(fb_dev_array, 0, 0, fb, nx * sizeof(vec3), nx * sizeof(vec3), ny, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaGraphicsUnmapResources(1, &resource, 0));

	// Aufraeumen
	checkCudaErrors(cudaDeviceSynchronize());
	free_world<<<1, 1>>>(d_list, d_world, d_camera);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaFree(d_camera));
	checkCudaErrors(cudaFree(d_list));
	checkCudaErrors(cudaFree(d_world));
	checkCudaErrors(cudaFree(d_rand_state));
	checkCudaErrors(cudaFree(d_rand_state2));
	checkCudaErrors(cudaFree(fb));

	cudaDeviceReset();

	return 0;
}
