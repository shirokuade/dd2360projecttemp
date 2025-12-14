#ifndef TEXTURE_H
#define TEXTURE_H
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

class texture {
  public:
    __device__ virtual ~texture() = default;

    __device__ virtual color value(double u, double v, const point3& p) const = 0;
};

class solid_color : public texture {
  public:
    color albedo;

    __device__ solid_color(const color& c) : albedo(c) {}

    __device__ solid_color(double red, double green, double blue) : solid_color(color(red,green,blue)) {}

    __device__ color value(double u, double v, const point3& p) const override {
        return albedo;
    }
};

#endif
