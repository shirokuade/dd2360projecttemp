#ifndef RAY_H
#define RAY_H
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

#include "vec3.h"

class ray {
  public:
    __host__ __device__ ray() : tm(0) {}

    __host__ __device__ ray(const point3& origin, const vec3& direction, double time)
      : orig(origin), dir(direction), tm(time) {}

    __host__ __device__ ray(const point3& origin, const vec3& direction)
      : ray(origin, direction, 0) {}

    // Explicit copy constructor for CUDA compatibility
    __host__ __device__ ray(const ray& other)
      : orig(other.orig), dir(other.dir), tm(other.tm) {}

    // Explicit assignment operator for CUDA compatibility
    __host__ __device__ ray& operator=(const ray& other) {
        orig = other.orig;
        dir = other.dir;
        tm = other.tm;
        return *this;
    }

    __host__ __device__ const point3& origin() const  { return orig; }
    __host__ __device__ const vec3& direction() const { return dir; }

    __host__ __device__ double time() const { return tm; }

    __host__ __device__ point3 at(double t) const {
        return orig + t*dir;
    }

  private:
    point3 orig;
    vec3 dir;
    double tm;
};


#endif
