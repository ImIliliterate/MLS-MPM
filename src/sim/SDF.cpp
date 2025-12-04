#include "SDF.h"

namespace SDF {

void Scene::addFloor(float height) {
    Primitive p;
    p.type = Primitive::Type::Floor;
    p.center = glm::vec3(0.0f);
    p.params = glm::vec3(height, 0.0f, 0.0f);
    primitives.push_back(p);
}

void Scene::addBox(const glm::vec3& center, const glm::vec3& halfExtent) {
    Primitive p;
    p.type = Primitive::Type::Box;
    p.center = center;
    p.params = halfExtent;
    primitives.push_back(p);
}

void Scene::addSphere(const glm::vec3& center, float radius) {
    Primitive p;
    p.type = Primitive::Type::Sphere;
    p.center = center;
    p.params = glm::vec3(radius, 0.0f, 0.0f);
    primitives.push_back(p);
}

void Scene::addCylinder(const glm::vec3& center, float radius, float height) {
    Primitive p;
    p.type = Primitive::Type::Cylinder;
    p.center = center;
    p.params = glm::vec3(radius, height, 0.0f);
    primitives.push_back(p);
}

float Scene::evaluate(const glm::vec3& pos) const {
    float minDist = 1e10f;
    
    for (const auto& prim : primitives) {
        float d = 0.0f;
        
        switch (prim.type) {
            case Primitive::Type::Floor:
                d = floor(pos, prim.params.x);
                break;
            case Primitive::Type::Box:
                d = box(pos, prim.center, prim.params);
                break;
            case Primitive::Type::Sphere:
                d = sphere(pos, prim.center, prim.params.x);
                break;
            case Primitive::Type::Cylinder:
                d = cylinder(pos, prim.center, prim.params.x, prim.params.y);
                break;
        }
        
        minDist = opUnion(minDist, d);
    }
    
    return minDist;
}

glm::vec3 Scene::gradient(const glm::vec3& p, float eps) const {
    return glm::vec3(
        evaluate(p + glm::vec3(eps, 0, 0)) - evaluate(p - glm::vec3(eps, 0, 0)),
        evaluate(p + glm::vec3(0, eps, 0)) - evaluate(p - glm::vec3(0, eps, 0)),
        evaluate(p + glm::vec3(0, 0, eps)) - evaluate(p - glm::vec3(0, 0, eps))
    ) / (2.0f * eps);
}

void Scene::clear() {
    primitives.clear();
}

} // namespace SDF

