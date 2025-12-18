#ifndef INTERVAL_H
#define INTERVAL_H
//==============================================================================================
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================

#include "precision.h"

class interval {
  public:
    real_t min, max;

    __host__ __device__
    interval() : min(+REAL_INF), max(-REAL_INF) {} // Default interval is empty

    __host__ __device__
    interval(real_t min, real_t max) : min(min), max(max) {}

    // Explicit copy constructor for CUDA compatibility
    __host__ __device__
    interval(const interval& other) : min(other.min), max(other.max) {}

    // Explicit assignment operator for CUDA compatibility
    __host__ __device__
    interval& operator=(const interval& other) {
        min = other.min;
        max = other.max;
        return *this;
    }

    __host__ __device__
    interval(const interval& a, const interval& b) {
        // Create the interval tightly enclosing the two input intervals.
        min = a.min <= b.min ? a.min : b.min;
        max = a.max >= b.max ? a.max : b.max;
    }

    __host__ __device__
    real_t size() const {
        return max - min;
    }

    __host__ __device__
    bool contains(real_t x) const {
        return min <= x && x <= max;
    }

    __host__ __device__
    bool surrounds(real_t x) const {
        return min < x && x < max;
    }

    __host__ __device__
    real_t clamp(real_t x) const {
        if (x < min) return min;
        if (x > max) return max;
        return x;
    }

    __host__ __device__
    interval expand(real_t delta) const {
        auto padding = delta / REAL_CONST(2.0);
        return interval(min - padding, max + padding);
    }

};

__host__ __device__
inline interval operator+(const interval& ival, real_t displacement) {
    return interval(ival.min + displacement, ival.max + displacement);
}

__host__ __device__
inline interval operator+(real_t displacement, const interval& ival) {
    return ival + displacement;
}

#endif
