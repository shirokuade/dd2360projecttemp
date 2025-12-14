#ifndef QUAD_H
#define QUAD_H
//==============================================================================================
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================

#include "hittable.h"
#include "hittable_list.h"

// Axis-aligned rectangle in XY plane
class xy_rect : public hittable {
  public:
    material *mp;
    double x0, x1, y0, y1, k;

    __device__ xy_rect() {}
    __device__ xy_rect(double _x0, double _x1, double _y0, double _y1, double _k, material *mat)
        : x0(_x0), x1(_x1), y0(_y0), y1(_y1), k(_k), mp(mat) {}

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        auto t = (k - r.origin().z()) / r.direction().z();
        if (t < ray_t.min || t > ray_t.max)
            return false;

        auto x = r.origin().x() + t * r.direction().x();
        auto y = r.origin().y() + t * r.direction().y();
        if (x < x0 || x > x1 || y < y0 || y > y1)
            return false;

        rec.u = (x - x0) / (x1 - x0);
        rec.v = (y - y0) / (y1 - y0);
        rec.t = t;
        vec3 outward_normal = vec3(0, 0, 1);
        rec.set_face_normal(r, outward_normal);
        rec.mat_ptr = mp;
        rec.p = r.at(t);
        return true;
    }

    __device__ aabb bounding_box() const override {
        return aabb(point3(x0, y0, k - 0.0001), point3(x1, y1, k + 0.0001));
    }
};

// Axis-aligned rectangle in XZ plane
class xz_rect : public hittable {
  public:
    material *mp;
    double x0, x1, z0, z1, k;

    __device__ xz_rect() {}
    __device__ xz_rect(double _x0, double _x1, double _z0, double _z1, double _k, material *mat)
        : x0(_x0), x1(_x1), z0(_z0), z1(_z1), k(_k), mp(mat) {}

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        auto t = (k - r.origin().y()) / r.direction().y();
        if (t < ray_t.min || t > ray_t.max)
            return false;

        auto x = r.origin().x() + t * r.direction().x();
        auto z = r.origin().z() + t * r.direction().z();
        if (x < x0 || x > x1 || z < z0 || z > z1)
            return false;

        rec.u = (x - x0) / (x1 - x0);
        rec.v = (z - z0) / (z1 - z0);
        rec.t = t;
        vec3 outward_normal = vec3(0, 1, 0);
        rec.set_face_normal(r, outward_normal);
        rec.mat_ptr = mp;
        rec.p = r.at(t);
        return true;
    }

    __device__ aabb bounding_box() const override {
        return aabb(point3(x0, k - 0.0001, z0), point3(x1, k + 0.0001, z1));
    }
};

// Axis-aligned rectangle in YZ plane
class yz_rect : public hittable {
  public:
    material *mp;
    double y0, y1, z0, z1, k;

    __device__ yz_rect() {}
    __device__ yz_rect(double _y0, double _y1, double _z0, double _z1, double _k, material *mat)
        : y0(_y0), y1(_y1), z0(_z0), z1(_z1), k(_k), mp(mat) {}

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        auto t = (k - r.origin().x()) / r.direction().x();
        if (t < ray_t.min || t > ray_t.max)
            return false;

        auto y = r.origin().y() + t * r.direction().y();
        auto z = r.origin().z() + t * r.direction().z();
        if (y < y0 || y > y1 || z < z0 || z > z1)
            return false;

        rec.u = (y - y0) / (y1 - y0);
        rec.v = (z - z0) / (z1 - z0);
        rec.t = t;
        vec3 outward_normal = vec3(1, 0, 0);
        rec.set_face_normal(r, outward_normal);
        rec.mat_ptr = mp;
        rec.p = r.at(t);
        return true;
    }

    __device__ aabb bounding_box() const override {
        return aabb(point3(k - 0.0001, y0, z0), point3(k + 0.0001, y1, z1));
    }
};

// Wrapper to flip normals (for inside-facing surfaces)
class flip_normals : public hittable {
  public:
    hittable *ptr;

    __device__ flip_normals(hittable *p) : ptr(p) {}

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        if (!ptr->hit(r, ray_t, rec))
            return false;

        rec.front_face = !rec.front_face;
        rec.normal = -rec.normal;
        return true;
    }

    __device__ aabb bounding_box() const override {
        return ptr->bounding_box();
    }
};

// Box (6 sides) - returns a hittable_list containing 6 rectangles
class box : public hittable {
  public:
    hittable **sides;
    int num_sides;
    aabb bbox;

    __device__ box() {}
    __device__ box(const point3& p0, const point3& p1, material *mat) {
        num_sides = 6;
        sides = new hittable*[6];

        auto min_pt = point3(fmin(p0.x(), p1.x()), fmin(p0.y(), p1.y()), fmin(p0.z(), p1.z()));
        auto max_pt = point3(fmax(p0.x(), p1.x()), fmax(p0.y(), p1.y()), fmax(p0.z(), p1.z()));

        // Front and back (XY planes)
        sides[0] = new xy_rect(min_pt.x(), max_pt.x(), min_pt.y(), max_pt.y(), max_pt.z(), mat);
        sides[1] = new flip_normals(new xy_rect(min_pt.x(), max_pt.x(), min_pt.y(), max_pt.y(), min_pt.z(), mat));

        // Top and bottom (XZ planes)
        sides[2] = new xz_rect(min_pt.x(), max_pt.x(), min_pt.z(), max_pt.z(), max_pt.y(), mat);
        sides[3] = new flip_normals(new xz_rect(min_pt.x(), max_pt.x(), min_pt.z(), max_pt.z(), min_pt.y(), mat));

        // Left and right (YZ planes)
        sides[4] = new yz_rect(min_pt.y(), max_pt.y(), min_pt.z(), max_pt.z(), max_pt.x(), mat);
        sides[5] = new flip_normals(new yz_rect(min_pt.y(), max_pt.y(), min_pt.z(), max_pt.z(), min_pt.x(), mat));

        bbox = aabb(min_pt, max_pt);
    }

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        hit_record temp_rec;
        bool hit_anything = false;
        auto closest_so_far = ray_t.max;

        for (int i = 0; i < num_sides; i++) {
            if (sides[i]->hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }
        return hit_anything;
    }

    __device__ aabb bounding_box() const override { return bbox; }
};

#endif
