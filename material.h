#ifndef MATERIAL_H
#define MATERIAL_H
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

#include "hittable.h"
#include "texture.h"
#include <curand_kernel.h>

class material {
  public:
    __device__ virtual ~material() = default;

    __device__ virtual color emitted(real_t u, real_t v, const point3& p) const {
        return color(0,0,0);
    }

    __device__ virtual bool scatter(
        const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState *local_rand_state
    ) const {
        return false;
    }
};

// Helper class for constant color textures used inline in materials
class constant_texture : public texture {
  public:
    color albedo;

    __device__ constant_texture(const color& c) : albedo(c) {}

    __device__ color value(real_t u, real_t v, const point3& p) const override {
        return albedo;
    }
};

class lambertian : public material {
  public:
    texture *tex;

    __device__ lambertian(texture *t) : tex(t) {}

    __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState *local_rand_state)
    const override {
        auto scatter_direction = rec.normal + random_unit_vector(local_rand_state);

        // Catch degenerate scatter direction
        if (scatter_direction.near_zero())
            scatter_direction = rec.normal;

        scattered = ray(rec.p, scatter_direction, r_in.time());
        attenuation = tex->value(rec.u, rec.v, rec.p);
        return true;
    }
};

class diffuse_light : public material {
  public:
    texture *tex;

    __device__ diffuse_light(texture *t) : tex(t) {}

    __device__ color emitted(real_t u, real_t v, const point3& p) const override {
        return tex->value(u, v, p);
    }
};

#endif
