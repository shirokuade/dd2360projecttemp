//==============================================================================================
// Originally written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================

#include "rtweekend.h"
#include "camera.h"
#include "hittable.h"
#include "hittable_list.h"
#include "material.h"
#include "quad.h"
#include "texture.h"
#include <iostream>
#include <curand_kernel.h>
#include "vec3.h"
#include "ray.h"
#include "material.h"

// Error checking macro
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

// Initialize rendering

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;

    // Each pixel has its own random seed
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}


// Creates Cornell Box
__global__ void create_world(hittable **d_list, hittable **d_world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int i = 0;
        material *red   = new lambertian( new constant_texture(vec3(0.65, 0.05, 0.05)) );
        material *white = new lambertian( new constant_texture(vec3(0.73, 0.73, 0.73)) );
        material *green = new lambertian( new constant_texture(vec3(0.12, 0.45, 0.15)) );
        material *light = new diffuse_light( new constant_texture(vec3(15, 15, 15)) );

        d_list[i++] = new flip_normals(new yz_rect(0, 555, 0, 555, 555, green));
        d_list[i++] = new yz_rect(0, 555, 0, 555, 0, red);
        d_list[i++] = new xz_rect(213, 343, 227, 332, 554, light);
        d_list[i++] = new flip_normals(new xz_rect(0, 555, 0, 555, 555, white));
        d_list[i++] = new xz_rect(0, 555, 0, 555, 0, white);
        d_list[i++] = new flip_normals(new xy_rect(0, 555, 0, 555, 555, white));
        
        d_list[i++] = new translate(new rotate_y(new box(vec3(0, 0, 0), vec3(165, 165, 165), white), -18), vec3(130,0,65));
        d_list[i++] = new translate(new rotate_y(new box(vec3(0, 0, 0), vec3(165, 330, 165), white),  15), vec3(265,0,295));
        
        *d_world = new hittable_list(d_list, i);
    }
}


__device__ vec3 ray_color(const ray& r, hittable **world, curandState *local_rand_state) {
    ray cur_ray = r;
    vec3 cur_attenuation(1.0, 1.0, 1.0);
    vec3 cur_emitted(0.0, 0.0, 0.0);

    // Changed from recursive to for-loop based for GPU
    for(int i = 0; i < 50; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, interval(0.001, 1e30), rec)) {
            vec3 emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);
            cur_emitted += cur_attenuation * emitted; // Accumulate emission
            
            vec3 attenuation;
            ray scattered;
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            } else {
                return cur_emitted;
            }
        } else {
            return cur_emitted; // Hit nothing (black background)
        }
    }
    return cur_emitted; // Exceeded max depth
}

__global__ void render(vec3 *fb, int max_x, int max_y, int ns, CameraData cam, hittable **world, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if((i >= max_x) || (j >= max_y)) return;

    int pixel_index = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0,0,0);

    for(int s=0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        
        ray r = get_ray(cam, u, v, &local_rand_state);
        col += ray_color(r, world, &local_rand_state);
    }

    rand_state[pixel_index] = local_rand_state;

    col /= float(ns);
    // Gamma correction
    col = vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));
    fb[pixel_index] = col;
}

int main() {
    int nx = 600; // Increased resolution
    int ny = 600;
    int ns = 100; // Samples per pixel
    int tx = 8;
    int ty = 8;

    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples.\n";

    // IMPORTANT: Set limits FIRST, before any CUDA allocations or kernel launches
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024 * 1024 * 256)); // 256MB heap
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitStackSize, 16384)); // 16KB stack per thread for deep call chains

    // Allocate Framebuffer
    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(vec3);
    vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // Allocate Random State
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState)));

    // Init Random State
    dim3 blocks(nx/tx + 1, ny/ty + 1);
    dim3 threads(tx, ty);
    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    hittable **d_list;
    checkCudaErrors(cudaMalloc((void **)&d_list, 20 * sizeof(hittable *))); // Array for list items
    hittable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hittable *)));     // Pointer to the list itself

    create_world<<<1, 1>>>(d_list, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Setup Camera on Host
    vec3 lookfrom(278, 278, -800);
    vec3 lookat(278, 278, 0);
    float dist_to_focus = 10.0;
    float aperture = 0.0;
    float vfov = 40.0;
    
    camera_host cam_host(lookfrom, lookat, vec3(0,1,0), vfov, float(nx)/float(ny), aperture, dist_to_focus, 0.0, 1.0, nx, ny);

    // Render
    render<<<blocks, threads>>>(fb, nx, ny, ns, cam_host.data, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Output
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j * nx + i;
            int ir = int(255.99 * fb[pixel_index].x());
            int ig = int(255.99 * fb[pixel_index].y());
            int ib = int(255.99 * fb[pixel_index].z());
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    // Freeing memory

    checkCudaErrors(cudaFree(fb));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));

    return 0;
}
