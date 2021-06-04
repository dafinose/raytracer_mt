﻿#include "../header/cuda_render.cuh"
#include "../header/ray.cuh"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
	if (result)
	{
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
		cudaDeviceReset();
		exit(99);
	}
}

__device__ vec3 color(const ray& r) 
{
	vec3 unit_direction = unit_vector(r.direction());
	float t = 0.5f * (unit_direction.y() + 1.0f);
	return (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
}

__global__ void render(float* fb, int max_x, int max_y)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x * 3 + i * 3;
	fb[pixel_index + 0] = float(i) / max_x;
	fb[pixel_index + 1] = float(j) / max_y;
	fb[pixel_index + 2] = 0.2;
}

int cuda_main()
{
	// Bilddimensionen
	int nx = 800;
	int ny = 600;

	// Threaddimensionen für CUDA
	int tx = 8;
	int ty = 8;

	int num_pixels = nx * ny;
	// Größe des Frambuffers
	size_t fb_size = 3 * num_pixels * sizeof(float);

	// Framebuffer allokieren
	float* fb;
	checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty);
	render<<<blocks, threads>>>(fb, nx, ny);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// Bild ausgeben
	std::cout << "P3\n" << nx << " " << ny << "\n255\n";
	for (int j = 0; j < ny; j++)
	{
		for (int i = 0; i < nx; i++)
		{
			size_t pixel_index = j * 3 * nx + i * 3;
			float r = fb[pixel_index + 0];
			float g = fb[pixel_index + 1];
			float b = fb[pixel_index + 2];
			int ir = int(255.99 * r);
			int ig = int(255.99 * g);
			int ib = int(255.99 * b);
			//std::cout << ir << " " << ig << " " << ib << "\n";
		}
	}
	checkCudaErrors(cudaFree(fb));

	return 0;
}