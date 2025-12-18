#ifndef BVH_H
#define BVH_H
//==============================================================================================
// BVH (Bounding Volume Hierarchy) acceleration structure for CUDA ray tracing
//==============================================================================================

#include "hittable.h"
#include "aabb.h"

// BVH Node - binary tree node containing either two children or a leaf object
class bvh_node : public hittable {
  public:
    hittable* left;
    hittable* right;
    aabb bbox;

    __device__ bvh_node() : left(nullptr), right(nullptr) {}

    // Build BVH from array of objects (device-side construction)
    __device__ bvh_node(hittable** objects, int start, int end, unsigned int* rand_seed) {
        // Calculate bounding box for this node's objects
        bbox = objects[start]->bounding_box();
        for (int i = start + 1; i < end; i++) {
            bbox = aabb(bbox, objects[i]->bounding_box());
        }

        int object_span = end - start;

        if (object_span == 1) {
            // Leaf node with single object
            left = right = objects[start];
        } else if (object_span == 2) {
            // Two objects - assign to left and right
            left = objects[start];
            right = objects[start + 1];
        } else {
            // More than 2 objects - split and recurse

            // Choose axis to split on (longest axis of bounding box)
            int axis = bbox.longest_axis();

            // Simple bubble sort by axis (fine for small n)
            for (int i = start; i < end - 1; i++) {
                for (int j = start; j < end - 1 - (i - start); j++) {
                    aabb box_a = objects[j]->bounding_box();
                    aabb box_b = objects[j + 1]->bounding_box();

                    real_t a_val = box_a.axis_interval(axis).min;
                    real_t b_val = box_b.axis_interval(axis).min;

                    if (a_val > b_val) {
                        // Swap
                        hittable* temp = objects[j];
                        objects[j] = objects[j + 1];
                        objects[j + 1] = temp;
                    }
                }
            }

            // Split in the middle
            int mid = start + object_span / 2;

            left = new bvh_node(objects, start, mid, rand_seed);
            right = new bvh_node(objects, mid, end, rand_seed);
        }
    }

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        // First test against this node's bounding box
        if (!bbox.hit(r, ray_t))
            return false;

        // If we hit the box, test children
        bool hit_left = left->hit(r, ray_t, rec);
        bool hit_right = right->hit(r, interval(ray_t.min, hit_left ? rec.t : ray_t.max), rec);

        return hit_left || hit_right;
    }

    __device__ aabb bounding_box() const override {
        return bbox;
    }
};

// Wrapper class that builds BVH from a list of objects
class bvh_tree : public hittable {
  public:
    bvh_node* root;
    aabb bbox;

    __device__ bvh_tree() : root(nullptr) {}

    __device__ bvh_tree(hittable** objects, int n) {
        if (n == 0) {
            root = nullptr;
            return;
        }

        unsigned int rand_seed = 42;
        root = new bvh_node(objects, 0, n, &rand_seed);
        bbox = root->bounding_box();
    }

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        if (root == nullptr)
            return false;
        return root->hit(r, ray_t, rec);
    }

    __device__ aabb bounding_box() const override {
        return bbox;
    }
};

#endif
