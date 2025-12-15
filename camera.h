#ifndef CAMERA_H
#define CAMERA_H
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

#include <curand_kernel.h>
#include "ray.h"

struct CameraData {
    vec3 origin;
    vec3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;
    float time0, time1;
    float lens_radius;
    int image_width;
    int image_height;

    __host__ __device__ CameraData() : time0(0), time1(0), lens_radius(0), image_width(0), image_height(0) {}

    // Explicit copy constructor for CUDA compatibility
    __host__ __device__ CameraData(const CameraData& other)
        : origin(other.origin), lower_left_corner(other.lower_left_corner),
          horizontal(other.horizontal), vertical(other.vertical),
          u(other.u), v(other.v), w(other.w),
          time0(other.time0), time1(other.time1),
          lens_radius(other.lens_radius),
          image_width(other.image_width), image_height(other.image_height) {}

    // Explicit assignment operator for CUDA compatibility
    __host__ __device__ CameraData& operator=(const CameraData& other) {
        origin = other.origin;
        lower_left_corner = other.lower_left_corner;
        horizontal = other.horizontal;
        vertical = other.vertical;
        u = other.u;
        v = other.v;
        w = other.w;
        time0 = other.time0;
        time1 = other.time1;
        lens_radius = other.lens_radius;
        image_width = other.image_width;
        image_height = other.image_height;
        return *this;
    }
};

class camera_host {
public:
    CameraData data;

    camera_host(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect, 
                float aperture, float focus_dist, float t0, float t1, int w, int h) {
        data.time0 = t0;
        data.time1 = t1;
        data.image_width = w;
        data.image_height = h;
        data.lens_radius = aperture / 2;

        float theta = vfov * M_PI / 180;
        float half_height = tan(theta / 2);
        float half_width = aspect * half_height;

        data.origin = lookfrom;
        data.w = unit_vector(lookfrom - lookat);
        data.u = unit_vector(cross(vup, data.w));
        data.v = cross(data.w, data.u);

        data.lower_left_corner = data.origin - half_width * focus_dist * data.u 
                               - half_height * focus_dist * data.v - focus_dist * data.w;
        data.horizontal = 2 * half_width * focus_dist * data.u;
        data.vertical = 2 * half_height * focus_dist * data.v;
    }
};

// Generates rays on the GPU
__device__ ray get_ray(const CameraData& cam, float s, float t, curandState *local_rand_state) {
    vec3 rd = cam.lens_radius * random_in_unit_disk(local_rand_state);
    vec3 offset = cam.u * rd.x() + cam.v * rd.y();
    float time = cam.time0 + curand_uniform(local_rand_state) * (cam.time1 - cam.time0);
    
    return ray(cam.origin + offset, 
               cam.lower_left_corner + s * cam.horizontal + t * cam.vertical - cam.origin - offset, 
               time);
}

#endif

