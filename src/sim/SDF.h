#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <functional>

/**
 * Signed Distance Field utilities
 * Provides analytic SDFs for common primitives and operations
 */
namespace SDF {

// Primitive SDFs
inline float plane(const glm::vec3& p, const glm::vec3& n, float d) {
    return glm::dot(p, n) + d;
}

inline float floor(const glm::vec3& p, float height = 0.0f) {
    return p.y - height;
}

inline float box(const glm::vec3& p, const glm::vec3& center, const glm::vec3& halfExtent) {
    glm::vec3 q = glm::abs(p - center) - halfExtent;
    return glm::length(glm::max(q, glm::vec3(0.0f))) + 
           glm::min(glm::max(q.x, glm::max(q.y, q.z)), 0.0f);
}

inline float sphere(const glm::vec3& p, const glm::vec3& center, float radius) {
    return glm::length(p - center) - radius;
}

inline float cylinder(const glm::vec3& p, const glm::vec3& center, float radius, float height) {
    glm::vec2 d = glm::abs(glm::vec2(glm::length(glm::vec2(p.x - center.x, p.z - center.z)), 
                                      p.y - center.y)) - glm::vec2(radius, height);
    return glm::min(glm::max(d.x, d.y), 0.0f) + glm::length(glm::max(d, glm::vec2(0.0f)));
}

// CSG operations
inline float opUnion(float d1, float d2) {
    return glm::min(d1, d2);
}

inline float opIntersect(float d1, float d2) {
    return glm::max(d1, d2);
}

inline float opSubtract(float d1, float d2) {
    return glm::max(d1, -d2);
}

inline float opSmoothUnion(float d1, float d2, float k) {
    float h = glm::clamp(0.5f + 0.5f * (d2 - d1) / k, 0.0f, 1.0f);
    return glm::mix(d2, d1, h) - k * h * (1.0f - h);
}

// Gradient computation via central differences
inline glm::vec3 gradient(std::function<float(const glm::vec3&)> sdf, 
                          const glm::vec3& p, float eps = 0.001f) {
    return glm::vec3(
        sdf(p + glm::vec3(eps, 0, 0)) - sdf(p - glm::vec3(eps, 0, 0)),
        sdf(p + glm::vec3(0, eps, 0)) - sdf(p - glm::vec3(0, eps, 0)),
        sdf(p + glm::vec3(0, 0, eps)) - sdf(p - glm::vec3(0, 0, eps))
    ) / (2.0f * eps);
}

/**
 * Scene SDF: combine multiple primitives
 */
class Scene {
public:
    Scene() = default;
    
    void addFloor(float height = 0.0f);
    void addBox(const glm::vec3& center, const glm::vec3& halfExtent);
    void addSphere(const glm::vec3& center, float radius);
    void addCylinder(const glm::vec3& center, float radius, float height);
    
    float evaluate(const glm::vec3& p) const;
    glm::vec3 gradient(const glm::vec3& p, float eps = 0.001f) const;
    
    void clear();
    
private:
    struct Primitive {
        enum class Type { Floor, Box, Sphere, Cylinder };
        Type type;
        glm::vec3 center;
        glm::vec3 params;  // Type-specific parameters
    };
    
    std::vector<Primitive> primitives;
};

} // namespace SDF

