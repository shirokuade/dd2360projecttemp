#ifndef VEC3_H
#define VEC3_H
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
#include "precision.h"

class vec3 {
  public:
    real_t e[3];

    __host__ __device__ vec3() : e{0,0,0} {}
    __host__ __device__ vec3(real_t e0, real_t e1, real_t e2) : e{e0, e1, e2} {}

    // Explicit copy constructor for CUDA compatibility
    __host__ __device__ vec3(const vec3& other) : e{other.e[0], other.e[1], other.e[2]} {}

    // Explicit assignment operator for CUDA compatibility
    __host__ __device__ vec3& operator=(const vec3& other) {
        e[0] = other.e[0];
        e[1] = other.e[1];
        e[2] = other.e[2];
        return *this;
    }

    __host__ __device__ real_t x() const { return e[0]; }
    __host__ __device__ real_t y() const { return e[1]; }
    __host__ __device__ real_t z() const { return e[2]; }

    __host__ __device__ vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    __host__ __device__ real_t operator[](int i) const { return e[i]; }
    __host__ __device__ real_t& operator[](int i) { return e[i]; }

    __host__ __device__ vec3& operator+=(const vec3& v) {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    __host__ __device__ vec3& operator*=(real_t t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    __host__ __device__ vec3& operator*=(const vec3& v) {
        e[0] *= v.e[0];
        e[1] *= v.e[1];
        e[2] *= v.e[2];
        return *this;
    }

    __host__ __device__ vec3& operator/=(real_t t) {
        return *this *= REAL_CONST(1.0)/t;
    }

    __host__ __device__ real_t length() const {
        return REAL_SQRT(length_squared());
    }

    __host__ __device__ real_t length_squared() const {
        return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
    }

    __host__ __device__ bool near_zero() const {
        // Return true if the vector is close to zero in all dimensions.
        return (REAL_FABS(e[0]) < REAL_EPSILON) && (REAL_FABS(e[1]) < REAL_EPSILON) && (REAL_FABS(e[2]) < REAL_EPSILON);
    }
};

// point3 is just an alias for vec3, but useful for geometric clarity in the code.
using point3 = vec3;


// Vector Utility Functions

__host__ __device__ inline vec3 operator+(const vec3& u, const vec3& v) {
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3& u, const vec3& v) {
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& u, const vec3& v) {
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__host__ __device__ inline vec3 operator*(real_t t, const vec3& v) {
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& v, real_t t) {
    return t * v;
}

__host__ __device__ inline vec3 operator/(const vec3& v, real_t t) {
    return (REAL_CONST(1.0)/t) * v;
}

__host__ __device__ inline real_t dot(const vec3& u, const vec3& v) {
    return u.e[0] * v.e[0]
         + u.e[1] * v.e[1]
         + u.e[2] * v.e[2];
}

__host__ __device__ inline vec3 cross(const vec3& u, const vec3& v) {
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__host__ __device__ inline vec3 unit_vector(const vec3& v) {
    return v / v.length();
}

// GPU random functions using curand
__device__ inline vec3 random_in_unit_disk(curandState *local_rand_state) {
    while (true) {
        auto p = vec3(curand_uniform(local_rand_state) * REAL_CONST(2.0) - REAL_CONST(1.0),
                      curand_uniform(local_rand_state) * REAL_CONST(2.0) - REAL_CONST(1.0),
                      0);
        if (p.length_squared() < REAL_CONST(1.0))
            return p;
    }
}

__device__ inline vec3 random_unit_vector(curandState *local_rand_state) {
    while (true) {
        auto p = vec3(curand_uniform(local_rand_state) * REAL_CONST(2.0) - REAL_CONST(1.0),
                      curand_uniform(local_rand_state) * REAL_CONST(2.0) - REAL_CONST(1.0),
                      curand_uniform(local_rand_state) * REAL_CONST(2.0) - REAL_CONST(1.0));
        auto lensq = p.length_squared();
        if (REAL_CONST(1e-160) < lensq && lensq <= REAL_CONST(1.0))
            return p / REAL_SQRT(lensq);
    }
}

__device__ inline vec3 random_on_hemisphere(const vec3& normal, curandState *local_rand_state) {
    vec3 on_unit_sphere = random_unit_vector(local_rand_state);
    if (dot(on_unit_sphere, normal) > REAL_CONST(0.0))
        return on_unit_sphere;
    else
        return -on_unit_sphere;
}

__host__ __device__ inline vec3 reflect(const vec3& v, const vec3& n) {
    return v - REAL_CONST(2.0)*dot(v,n)*n;
}

__host__ __device__ inline vec3 refract(const vec3& uv, const vec3& n, real_t etai_over_etat) {
    auto cos_theta = REAL_FMIN(dot(-uv, n), REAL_CONST(1.0));
    vec3 r_out_perp =  etai_over_etat * (uv + cos_theta*n);
    vec3 r_out_parallel = -REAL_SQRT(REAL_FABS(REAL_CONST(1.0) - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}


#endif
