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

//==============================================================================
// COMPILE-TIME CONFIGURATION (cannot be changed at runtime)
//==============================================================================
#define USE_DOUBLE_PRECISION false // Use double (64-bit) precision. Set to false for float (32-bit)
//==============================================================================

#include "rtweekend.h"
#include "camera.h"
#include "hittable.h"
#include "hittable_list.h"
#include "bvh.h"
#include "material.h"
#include "quad.h"
#include "texture.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <sys/stat.h>
#include <cstdlib>
#include <curand_kernel.h>
#include <algorithm>
#include <map>
#include "vec3.h"
#include "ray.h"
#include "material.h"

//==============================================================================
// RUNTIME CONFIGURATION
//==============================================================================
struct Config {
    // Rendering settings
    bool progressive_render = false;
    int samples_per_pixel = 100;
    int image_width = 600;
    int image_height = 600;

    // CUDA optimization settings
    int block_size_x = 32;
    int block_size_y = 8;
    int num_streams = 4;

    // Memory settings
    bool use_pinned_memory = true;
    bool use_constant_memory = true;

    // Acceleration settings
    bool bvh_optim = false;

    // Persistent threads settings
    bool use_persistent_threads = false;
    bool use_tiled_persistent = false;  // Tiled version (better cache locality)
    int tile_size_x = 16;               // Tile width for tiled persistent
    int tile_size_y = 16;               // Tile height for tiled persistent
    int num_blocks = 256;               // Number of blocks for persistent/tiled kernels
};

// Trim whitespace from string
std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, last - first + 1);
}

// Parse boolean value
bool parse_bool(const std::string& value) {
    std::string v = trim(value);
    std::transform(v.begin(), v.end(), v.begin(), ::tolower);
    return (v == "true" || v == "1" || v == "yes");
}

// Load configuration from file
Config load_config(const std::string& filename) {
    Config config;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Warning: Could not open " << filename << ", using default values.\n";
        return config;
    }

    std::string line;
    while (std::getline(file, line)) {
        // Skip comments and empty lines
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;

        // Find '=' separator
        size_t pos = line.find('=');
        if (pos == std::string::npos) continue;

        std::string key = trim(line.substr(0, pos));
        std::string value = trim(line.substr(pos + 1));

        // Remove inline comments
        size_t comment_pos = value.find('#');
        if (comment_pos != std::string::npos) {
            value = trim(value.substr(0, comment_pos));
        }

        // Parse configuration values
        if (key == "PROGRESSIVE_RENDER") {
            config.progressive_render = parse_bool(value);
        } else if (key == "SAMPLES_PER_PIXEL") {
            config.samples_per_pixel = std::stoi(value);
        } else if (key == "IMAGE_WIDTH") {
            config.image_width = std::stoi(value);
        } else if (key == "IMAGE_HEIGHT") {
            config.image_height = std::stoi(value);
        } else if (key == "BLOCK_SIZE_X") {
            config.block_size_x = std::stoi(value);
        } else if (key == "BLOCK_SIZE_Y") {
            config.block_size_y = std::stoi(value);
        } else if (key == "NUM_STREAMS") {
            config.num_streams = std::stoi(value);
        } else if (key == "USE_PINNED_MEMORY") {
            config.use_pinned_memory = parse_bool(value);
        } else if (key == "USE_CONSTANT_MEMORY") {
            config.use_constant_memory = parse_bool(value);
        } else if (key == "BVH_OPTIM") {
            config.bvh_optim = parse_bool(value);
        } else if (key == "USE_PERSISTENT_THREADS") {
            config.use_persistent_threads = parse_bool(value);
        } else if (key == "USE_TILED_PERSISTENT") {
            config.use_tiled_persistent = parse_bool(value);
        } else if (key == "TILE_SIZE_X") {
            config.tile_size_x = std::stoi(value);
        } else if (key == "TILE_SIZE_Y") {
            config.tile_size_y = std::stoi(value);
        } else if (key == "NUM_BLOCKS") {
            config.num_blocks = std::stoi(value);
        }
    }

    file.close();
    return config;
}

//==============================================================================

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

// POD struct for constant memory (no constructors allowed in __constant__)
struct CameraDataPOD {
    real_t origin[3];
    real_t lower_left_corner[3];
    real_t horizontal[3];
    real_t vertical[3];
    real_t u[3], v[3], w[3];
    real_t time0, time1;
    real_t lens_radius;
    int image_width;
    int image_height;
};

// Constant memory for camera data (64KB limit, CameraDataPOD is small)
__constant__ CameraDataPOD d_camera_pod;

// Helper to copy CameraData to POD struct
void copyToPOD(CameraDataPOD& pod, const CameraData& cam) {
    pod.origin[0] = cam.origin.x(); pod.origin[1] = cam.origin.y(); pod.origin[2] = cam.origin.z();
    pod.lower_left_corner[0] = cam.lower_left_corner.x(); pod.lower_left_corner[1] = cam.lower_left_corner.y(); pod.lower_left_corner[2] = cam.lower_left_corner.z();
    pod.horizontal[0] = cam.horizontal.x(); pod.horizontal[1] = cam.horizontal.y(); pod.horizontal[2] = cam.horizontal.z();
    pod.vertical[0] = cam.vertical.x(); pod.vertical[1] = cam.vertical.y(); pod.vertical[2] = cam.vertical.z();
    pod.u[0] = cam.u.x(); pod.u[1] = cam.u.y(); pod.u[2] = cam.u.z();
    pod.v[0] = cam.v.x(); pod.v[1] = cam.v.y(); pod.v[2] = cam.v.z();
    pod.w[0] = cam.w.x(); pod.w[1] = cam.w.y(); pod.w[2] = cam.w.z();
    pod.time0 = cam.time0;
    pod.time1 = cam.time1;
    pod.lens_radius = cam.lens_radius;
    pod.image_width = cam.image_width;
    pod.image_height = cam.image_height;
}

// Device function to get ray from POD camera data
__device__ ray get_ray_from_pod(const CameraDataPOD& cam, real_t s, real_t t, curandState *local_rand_state) {
    vec3 origin(cam.origin[0], cam.origin[1], cam.origin[2]);
    vec3 lower_left(cam.lower_left_corner[0], cam.lower_left_corner[1], cam.lower_left_corner[2]);
    vec3 horizontal(cam.horizontal[0], cam.horizontal[1], cam.horizontal[2]);
    vec3 vertical(cam.vertical[0], cam.vertical[1], cam.vertical[2]);
    vec3 u(cam.u[0], cam.u[1], cam.u[2]);
    vec3 v(cam.v[0], cam.v[1], cam.v[2]);

    vec3 rd = cam.lens_radius * random_in_unit_disk(local_rand_state);
    vec3 offset = u * rd.x() + v * rd.y();
    real_t time = cam.time0 + curand_uniform(local_rand_state) * (cam.time1 - cam.time0);

    return ray(origin + offset,
               lower_left + s * horizontal + t * vertical - origin - offset,
               time);
}

// Initialize rendering - optimized with better thread utilization
__global__ void render_init(int max_x, int max_y, curandState *rand_state, int y_offset) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y + y_offset;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;

    // Each pixel has its own random seed - use better seed mixing
    curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);
}


// Creates Cornell Box - with optional BVH acceleration
__global__ void create_world(hittable **d_list, hittable **d_world, bool use_bvh) {
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

        if (use_bvh) {
            // Build BVH tree for O(log n) intersection tests
            *d_world = new bvh_tree(d_list, i);
        } else {
            // Use linear list for O(n) intersection tests
            *d_world = new hittable_list(d_list, i);
        }
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

// Fast render kernel using constant memory for camera
__global__ void render(vec3 *fb, int max_x, int max_y, int ns, hittable **world, curandState *rand_state, int y_offset, int y_size) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int local_j = threadIdx.y + blockIdx.y * blockDim.y;
    int j = local_j + y_offset;

    if((i >= max_x) || (local_j >= y_size) || (j >= max_y)) return;

    int pixel_index = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0,0,0);

    for(int s=0; s < ns; s++) {
        real_t u = real_t(i + curand_uniform(&local_rand_state)) / real_t(max_x);
        real_t v = real_t(j + curand_uniform(&local_rand_state)) / real_t(max_y);

        // Use constant memory camera (POD version)
        ray r = get_ray_from_pod(d_camera_pod, u, v, &local_rand_state);
        col += ray_color(r, world, &local_rand_state);
    }

    rand_state[pixel_index] = local_rand_state;

    col /= real_t(ns);
    // Gamma correction
    col = vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));
    fb[pixel_index] = col;
}

// Render kernel with CameraData passed as parameter (for non-constant memory mode)
__global__ void render_param(vec3 *fb, int max_x, int max_y, int ns, CameraData cam, hittable **world, curandState *rand_state, int y_offset, int y_size) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int local_j = threadIdx.y + blockIdx.y * blockDim.y;
    int j = local_j + y_offset;

    if((i >= max_x) || (local_j >= y_size) || (j >= max_y)) return;

    int pixel_index = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0,0,0);

    for(int s=0; s < ns; s++) {
        real_t u = real_t(i + curand_uniform(&local_rand_state)) / real_t(max_x);
        real_t v = real_t(j + curand_uniform(&local_rand_state)) / real_t(max_y);

        ray r = get_ray(cam, u, v, &local_rand_state);
        col += ray_color(r, world, &local_rand_state);
    }

    rand_state[pixel_index] = local_rand_state;

    col /= real_t(ns);
    // Gamma correction
    col = vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));
    fb[pixel_index] = col;
}

// Persistent threads render kernel - threads grab work from atomic counter
__global__ void render_persistent(vec3 *fb, int max_x, int max_y, int ns,
                                   hittable **world, curandState *rand_state,
                                   int *work_counter, int total_pixels) {
    // Each thread loops, grabbing pixels until all work is done
    while (true) {
        // Atomically grab next pixel index
        int pixel_index = atomicAdd(work_counter, 1);

        // Exit if no more work
        if (pixel_index >= total_pixels) return;

        // Convert linear index to (i, j) coordinates
        int i = pixel_index % max_x;
        int j = pixel_index / max_x;

        // Load random state for this pixel
        curandState local_rand_state = rand_state[pixel_index];
        vec3 col(0,0,0);

        // Render all samples for this pixel
        for (int s = 0; s < ns; s++) {
            real_t u = real_t(i + curand_uniform(&local_rand_state)) / real_t(max_x);
            real_t v = real_t(j + curand_uniform(&local_rand_state)) / real_t(max_y);

            ray r = get_ray_from_pod(d_camera_pod, u, v, &local_rand_state);
            col += ray_color(r, world, &local_rand_state);
        }

        // Save random state back
        rand_state[pixel_index] = local_rand_state;

        // Normalize and gamma correct
        col /= real_t(ns);
        col = vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));
        fb[pixel_index] = col;
    }
}

// Tiled persistent threads render kernel - thread blocks grab tiles, threads cooperate on pixels
// This reduces atomic contention and improves cache locality

__global__ void render_tiled_persistent(vec3 *fb, int max_x, int max_y, int ns,
                                         hittable **world, curandState *rand_state,
                                         int *tile_counter, int tiles_x, int tiles_y,
                                         int tile_size_x, int tile_size_y) {
    // Shared variable for tile index - only one thread per block does atomic
    __shared__ int tile_idx;

    // Total number of tiles
    int total_tiles = tiles_x * tiles_y;

    // Thread's position within the block (using 2D indexing for better coalescing)
    int local_x = threadIdx.x;
    int local_y = threadIdx.y;

    // Loop: each iteration, the block grabs a new tile
    while (true) {
        // Thread (0,0) grabs the next tile atomically
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            tile_idx = atomicAdd(tile_counter, 1);
        }

        // All threads wait for thread 0 to get the tile index
        __syncthreads();

        // Exit if no more tiles
        if (tile_idx >= total_tiles) return;

        // Calculate tile's top-left corner
        int tile_x = (tile_idx % tiles_x) * tile_size_x;
        int tile_y = (tile_idx / tiles_x) * tile_size_y;

        // Calculate this thread's pixel coordinates
        int pixel_x = tile_x + local_x;
        int pixel_y = tile_y + local_y;

        // Sync before potentially exiting to ensure all threads participate in next iteration
        __syncthreads();

        // Bounds check - some threads may be outside image for edge tiles
        if (pixel_x >= max_x || pixel_y >= max_y) continue;

        int pixel_index = pixel_y * max_x + pixel_x;

        // Load random state for this pixel
        curandState local_rand_state = rand_state[pixel_index];
        vec3 col(0,0,0);

        // Render all samples for this pixel
        for (int s = 0; s < ns; s++) {
            real_t u = real_t(pixel_x + curand_uniform(&local_rand_state)) / real_t(max_x);
            real_t v = real_t(pixel_y + curand_uniform(&local_rand_state)) / real_t(max_y);

            ray r = get_ray_from_pod(d_camera_pod, u, v, &local_rand_state);
            col += ray_color(r, world, &local_rand_state);
        }

        // Save random state back
        rand_state[pixel_index] = local_rand_state;

        // Normalize and gamma correct
        col /= real_t(ns);
        col = vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));
        fb[pixel_index] = col;
    }
}

// Progressive render kernel - renders a single sample and accumulates (for progress visualization)
__global__ void render_sample(vec3 *accum_fb, int max_x, int max_y, hittable **world, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if((i >= max_x) || (j >= max_y)) return;

    int pixel_index = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];

    // Generate one sample using constant memory camera (POD version)
    real_t u = real_t(i + curand_uniform(&local_rand_state)) / real_t(max_x);
    real_t v = real_t(j + curand_uniform(&local_rand_state)) / real_t(max_y);

    ray r = get_ray_from_pod(d_camera_pod, u, v, &local_rand_state);
    vec3 col = ray_color(r, world, &local_rand_state);

    // Accumulate (no averaging yet - that happens when saving frame)
    accum_fb[pixel_index] += col;

    rand_state[pixel_index] = local_rand_state;
}

// Save a frame as PPM file
void save_frame_ppm(const std::string& filename, vec3* fb, int nx, int ny, int num_samples) {
    std::ofstream file(filename);
    file << "P3\n" << nx << " " << ny << "\n255\n";

    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j * nx + i;
            // Average by number of samples and apply gamma correction
            vec3 col = fb[pixel_index] / float(num_samples);
            col = vec3(sqrt(fmax(0.0, col[0])), sqrt(fmax(0.0, col[1])), sqrt(fmax(0.0, col[2])));

            int ir = int(255.99 * fmin(1.0, col.x()));
            int ig = int(255.99 * fmin(1.0, col.y()));
            int ib = int(255.99 * fmin(1.0, col.z()));
            file << ir << " " << ig << " " << ib << "\n";
        }
    }
    file.close();
}

int main() {
    // Load configuration from file
    Config cfg = load_config("config.conf");

    int nx = cfg.image_width;
    int ny = cfg.image_height;
    int ns = cfg.samples_per_pixel;
    int tx = cfg.block_size_x;
    int ty = cfg.block_size_y;
    int num_streams = cfg.num_streams;

    std::cerr << "=== Ray Tracer Configuration ===\n";
    std::cerr << "Image: " << nx << "x" << ny << " with " << ns << " samples\n";
    std::cerr << "Progressive rendering: " << (cfg.progressive_render ? "ENABLED (slower)" : "DISABLED (fast)") << "\n";
    std::cerr << "Block size: " << tx << "x" << ty << " = " << (tx*ty) << " threads/block\n";
    std::cerr << "Streams: " << num_streams << "\n";
    std::cerr << "Pinned memory: " << (cfg.use_pinned_memory ? "YES" : "NO") << "\n";
    std::cerr << "Constant memory: " << (cfg.use_constant_memory ? "YES" : "NO") << "\n";
    std::cerr << "BVH acceleration: " << (cfg.bvh_optim ? "YES (O(log n))" : "NO (O(n))") << "\n";
    std::cerr << "Persistent threads: " << (cfg.use_persistent_threads ? "YES (per-pixel)" : "NO") << "\n";
    std::cerr << "Tiled persistent: " << (cfg.use_tiled_persistent ? "YES" : "NO");
    if (cfg.use_tiled_persistent) std::cerr << " (" << cfg.tile_size_x << "x" << cfg.tile_size_y << " tiles)";
    std::cerr << "\n";
    std::cerr << "Precision: " << (USE_DOUBLE_PRECISION ? "double (64-bit)" : "float (32-bit)") << "\n";
    std::cerr << "================================\n";

    // IMPORTANT: Set limits FIRST, before any CUDA allocations or kernel launches
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024 * 1024 * 256)); // 256MB heap
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitStackSize, 16384)); // 16KB stack per thread for deep call chains

    // Allocate Framebuffer
    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(vec3);

    vec3 *d_fb;        // Device framebuffer
    vec3 *h_fb;        // Host framebuffer

    // Allocate device memory
    checkCudaErrors(cudaMalloc((void **)&d_fb, fb_size));

    // Allocate host memory (pinned for faster transfers if enabled)
    if (cfg.use_pinned_memory) {
        checkCudaErrors(cudaHostAlloc((void **)&h_fb, fb_size, cudaHostAllocDefault));
    } else {
        h_fb = new vec3[num_pixels];
    }

    // Initialize device framebuffer to zero
    checkCudaErrors(cudaMemset(d_fb, 0, fb_size));

    // Allocate Random State
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState)));

    // Create CUDA streams (dynamic allocation)
    std::vector<cudaStream_t> streams(num_streams);
    for (int s = 0; s < num_streams; s++) {
        checkCudaErrors(cudaStreamCreate(&streams[s]));
    }

    // Init Random State (can be done in parallel with streams)
    dim3 threads(tx, ty);
    int rows_per_stream = (ny + num_streams - 1) / num_streams;

    for (int s = 0; s < num_streams; s++) {
        int y_offset = s * rows_per_stream;
        int y_size = std::min(rows_per_stream, ny - y_offset);
        if (y_size <= 0) continue;

        dim3 blocks((nx + tx - 1) / tx, (y_size + ty - 1) / ty);
        render_init<<<blocks, threads, 0, streams[s]>>>(nx, ny, d_rand_state, y_offset);
    }

    // Synchronize all streams
    for (int s = 0; s < num_streams; s++) {
        checkCudaErrors(cudaStreamSynchronize(streams[s]));
    }
    checkCudaErrors(cudaGetLastError());

    hittable **d_list;
    checkCudaErrors(cudaMalloc((void **)&d_list, 20 * sizeof(hittable *))); // Array for list items
    hittable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hittable *)));     // Pointer to the list itself

    create_world<<<1, 1>>>(d_list, d_world, cfg.bvh_optim);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Setup Camera on Host
    vec3 lookfrom(278, 278, -800);
    vec3 lookat(278, 278, 0);
    float dist_to_focus = 10.0;
    float aperture = 0.0;
    float vfov = 40.0;

    camera_host cam_host(lookfrom, lookat, vec3(0,1,0), vfov, float(nx)/float(ny), aperture, dist_to_focus, 0.0, 1.0, nx, ny);

    // Copy camera data to constant memory (using POD struct) if enabled
    CameraDataPOD cam_pod;
    if (cfg.use_constant_memory) {
        copyToPOD(cam_pod, cam_host.data);
        checkCudaErrors(cudaMemcpyToSymbol(d_camera_pod, &cam_pod, sizeof(CameraDataPOD)));
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    if (cfg.progressive_render) {
        // Progressive rendering with time-based frame capture
        // Remove old frames directory and create fresh one
        system("rm -rf frames");
        mkdir("frames", 0777);
        std::cerr << "Progress frames will be saved to 'frames/' directory.\n";

        auto last_frame_time = start_time;
        int frame_count = 0;

        std::cerr << "Starting progressive render...\n";

        dim3 blocks((nx + tx - 1) / tx, (ny + ty - 1) / ty);

        for (int sample = 1; sample <= ns; sample++) {
            // Render one sample
            render_sample<<<blocks, threads>>>(d_fb, nx, ny, d_world, d_rand_state);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());

            auto current_time = std::chrono::high_resolution_clock::now();
            double elapsed_since_last_frame = std::chrono::duration<double>(current_time - last_frame_time).count();
            double total_elapsed = std::chrono::duration<double>(current_time - start_time).count();

            // Save a frame every ~1 second OR on the last sample
            if (elapsed_since_last_frame >= 1.0 || sample == ns) {
                // Copy framebuffer to host
                checkCudaErrors(cudaMemcpy(h_fb, d_fb, fb_size, cudaMemcpyDeviceToHost));

                std::ostringstream filename;
                filename << "frames/frame_" << std::setfill('0') << std::setw(5) << frame_count << ".ppm";

                save_frame_ppm(filename.str(), h_fb, nx, ny, sample);

                std::cerr << "\rSample " << sample << "/" << ns
                          << " | Frame " << frame_count
                          << " | Time: " << std::fixed << std::setprecision(1) << total_elapsed << "s"
                          << std::flush;

                frame_count++;
                last_frame_time = current_time;
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double>(end_time - start_time).count();

        std::cerr << "\n\nRendering complete!\n";
        std::cerr << "Total time: " << std::fixed << std::setprecision(2) << total_time << " seconds\n";
        std::cerr << "Total frames saved: " << frame_count << "\n";

        // Copy final framebuffer
        checkCudaErrors(cudaMemcpy(h_fb, d_fb, fb_size, cudaMemcpyDeviceToHost));

        // Output final image (need to normalize for progressive mode)
        std::cout << "P3\n" << nx << " " << ny << "\n255\n";
        for (int j = ny-1; j >= 0; j--) {
            for (int i = 0; i < nx; i++) {
                size_t pixel_index = j * nx + i;
                vec3 col = h_fb[pixel_index] / float(ns);
                col = vec3(sqrt(fmax(0.0, col[0])), sqrt(fmax(0.0, col[1])), sqrt(fmax(0.0, col[2])));

                int ir = int(255.99 * fmin(1.0, col.x()));
                int ig = int(255.99 * fmin(1.0, col.y()));
                int ib = int(255.99 * fmin(1.0, col.z()));
                std::cout << ir << " " << ig << " " << ib << "\n";
            }
        }

        std::cerr << "\nTo create a GIF showing render progress, run:\n";
        std::cerr << "  python make_gif.py render_progress 1\n";

    } else if (cfg.use_tiled_persistent) {
        // Tiled persistent threads rendering - blocks grab tiles dynamically
        int tile_size_x = cfg.tile_size_x;
        int tile_size_y = cfg.tile_size_y;
        std::cerr << "Starting render with tiled persistent threads (" << tile_size_x << "x" << tile_size_y << " tiles)...\n";

        // Allocate tile counter on device
        int *d_tile_counter;
        checkCudaErrors(cudaMalloc((void **)&d_tile_counter, sizeof(int)));
        checkCudaErrors(cudaMemset(d_tile_counter, 0, sizeof(int)));

        // Calculate number of tiles
        int tiles_x = (nx + tile_size_x - 1) / tile_size_x;
        int tiles_y = (ny + tile_size_y - 1) / tile_size_y;
        int total_tiles = tiles_x * tiles_y;

        std::cerr << "Tiles: " << tiles_x << "x" << tiles_y << " = " << total_tiles << " total\n";

        // Launch enough blocks to saturate GPU
        // Use 2D thread blocks matching tile dimensions for better coalescing
        dim3 block_dim(tile_size_x, tile_size_y);  // e.g., 32x8 = 256 threads

        render_tiled_persistent<<<cfg.num_blocks, block_dim>>>(
            d_fb, nx, ny, ns, d_world, d_rand_state,
            d_tile_counter, tiles_x, tiles_y, tile_size_x, tile_size_y);

        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        auto end_time = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double>(end_time - start_time).count();

        std::cerr << "Rendering complete!\n";
        std::cerr << "Total time: " << std::fixed << std::setprecision(2) << total_time << " seconds\n";

        // Cleanup tile counter
        checkCudaErrors(cudaFree(d_tile_counter));

        // Copy framebuffer to host
        if (cfg.use_pinned_memory) {
            checkCudaErrors(cudaMemcpyAsync(h_fb, d_fb, fb_size, cudaMemcpyDeviceToHost, streams[0]));
            checkCudaErrors(cudaStreamSynchronize(streams[0]));
        } else {
            checkCudaErrors(cudaMemcpy(h_fb, d_fb, fb_size, cudaMemcpyDeviceToHost));
        }

        // Output final image
        std::cout << "P3\n" << nx << " " << ny << "\n255\n";
        for (int j = ny-1; j >= 0; j--) {
            for (int i = 0; i < nx; i++) {
                size_t pixel_index = j * nx + i;
                int ir = int(255.99 * h_fb[pixel_index].x());
                int ig = int(255.99 * h_fb[pixel_index].y());
                int ib = int(255.99 * h_fb[pixel_index].z());
                std::cout << ir << " " << ig << " " << ib << "\n";
            }
        }

    } else if (cfg.use_persistent_threads) {
        // Persistent threads rendering - dynamic load balancing
        std::cerr << "Starting render with persistent threads...\n";

        // Allocate work counter on device
        int *d_work_counter;
        checkCudaErrors(cudaMalloc((void **)&d_work_counter, sizeof(int)));
        checkCudaErrors(cudaMemset(d_work_counter, 0, sizeof(int)));

        // Calculate grid size: enough blocks to fill the GPU
        // For H100/A100: ~80-100 SMs, 2048 threads per SM max
        // We want enough threads to saturate but not too many
        int threads_per_block = tx * ty;  // 256

        dim3 blocks(cfg.num_blocks, 1);
        dim3 thread_dim(threads_per_block, 1);

        render_persistent<<<blocks, thread_dim>>>(d_fb, nx, ny, ns, d_world, d_rand_state,
                                                   d_work_counter, num_pixels);

        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        auto end_time = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double>(end_time - start_time).count();

        std::cerr << "Rendering complete!\n";
        std::cerr << "Total time: " << std::fixed << std::setprecision(2) << total_time << " seconds\n";

        // Cleanup work counter
        checkCudaErrors(cudaFree(d_work_counter));

        // Copy framebuffer to host
        if (cfg.use_pinned_memory) {
            checkCudaErrors(cudaMemcpyAsync(h_fb, d_fb, fb_size, cudaMemcpyDeviceToHost, streams[0]));
            checkCudaErrors(cudaStreamSynchronize(streams[0]));
        } else {
            checkCudaErrors(cudaMemcpy(h_fb, d_fb, fb_size, cudaMemcpyDeviceToHost));
        }

        // Output final image
        std::cout << "P3\n" << nx << " " << ny << "\n255\n";
        for (int j = ny-1; j >= 0; j--) {
            for (int i = 0; i < nx; i++) {
                size_t pixel_index = j * nx + i;
                int ir = int(255.99 * h_fb[pixel_index].x());
                int ig = int(255.99 * h_fb[pixel_index].y());
                int ib = int(255.99 * h_fb[pixel_index].z());
                std::cout << ir << " " << ig << " " << ib << "\n";
            }
        }

    } else {
        // Fast rendering with streams - divide image into horizontal strips
        std::cerr << "Starting fast render with " << num_streams << " streams...\n";

        for (int s = 0; s < num_streams; s++) {
            int y_offset = s * rows_per_stream;
            int y_size = std::min(rows_per_stream, ny - y_offset);
            if (y_size <= 0) continue;

            dim3 blocks((nx + tx - 1) / tx, (y_size + ty - 1) / ty);

            if (cfg.use_constant_memory) {
                render<<<blocks, threads, 0, streams[s]>>>(d_fb, nx, ny, ns, d_world, d_rand_state, y_offset, y_size);
            } else {
                render_param<<<blocks, threads, 0, streams[s]>>>(d_fb, nx, ny, ns, cam_host.data, d_world, d_rand_state, y_offset, y_size);
            }
        }

        // Synchronize all streams
        for (int s = 0; s < num_streams; s++) {
            checkCudaErrors(cudaStreamSynchronize(streams[s]));
        }
        checkCudaErrors(cudaGetLastError());

        auto end_time = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double>(end_time - start_time).count();

        std::cerr << "Rendering complete!\n";
        std::cerr << "Total time: " << std::fixed << std::setprecision(2) << total_time << " seconds\n";

        // Copy framebuffer to host (async with pinned memory)
        if (cfg.use_pinned_memory) {
            checkCudaErrors(cudaMemcpyAsync(h_fb, d_fb, fb_size, cudaMemcpyDeviceToHost, streams[0]));
            checkCudaErrors(cudaStreamSynchronize(streams[0]));
        } else {
            checkCudaErrors(cudaMemcpy(h_fb, d_fb, fb_size, cudaMemcpyDeviceToHost));
        }

        // Output final image (already normalized in kernel)
        std::cout << "P3\n" << nx << " " << ny << "\n255\n";
        for (int j = ny-1; j >= 0; j--) {
            for (int i = 0; i < nx; i++) {
                size_t pixel_index = j * nx + i;
                int ir = int(255.99 * h_fb[pixel_index].x());
                int ig = int(255.99 * h_fb[pixel_index].y());
                int ib = int(255.99 * h_fb[pixel_index].z());
                std::cout << ir << " " << ig << " " << ib << "\n";
            }
        }
    }

    // Destroy streams
    for (int s = 0; s < num_streams; s++) {
        checkCudaErrors(cudaStreamDestroy(streams[s]));
    }

    // Freeing memory
    checkCudaErrors(cudaFree(d_fb));
    if (cfg.use_pinned_memory) {
        checkCudaErrors(cudaFreeHost(h_fb));
    } else {
        delete[] h_fb;
    }
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));

    return 0;
}
