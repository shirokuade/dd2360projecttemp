#ifndef RTWEEKEND_H
#define RTWEEKEND_H
//==============================================================================================
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "precision.h"

// Constants
#define infinity REAL_INF
#define pi REAL_CONST(3.1415926535897932385)

// Utility Functions
__host__ __device__ inline real_t degrees_to_radians(real_t degrees) {
    return degrees * pi / REAL_CONST(180.0);
}

// Common Headers
#include "color.h"
#include "interval.h"
#include "ray.h"
#include "vec3.h"


#endif
