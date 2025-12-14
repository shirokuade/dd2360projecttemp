#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H
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

#include "aabb.h"
#include "hittable.h"

#include <vector>


class hittable_list : public hittable {
  public:
    hittable **objects;
    int list_size;
    aabb bbox;

    __device__ hittable_list() { objects = nullptr; list_size = 0; }

    __device__ hittable_list(hittable **l, int n) {
        objects = l;
        list_size = n;
        
        // Calculate the bounding box for the entire list immediately
        if (list_size > 0) {
            bbox = objects[0]->bounding_box();
            for (int i = 1; i < list_size; i++) {
                bbox = aabb(bbox, objects[i]->bounding_box());
            }
        }
    }

    // Iterates through the raw array
    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        hit_record temp_rec;
        bool hit_anything = false;
        auto closest_so_far = ray_t.max;

        for (int i = 0; i < list_size; i++) {
            if (objects[i]->hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
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
