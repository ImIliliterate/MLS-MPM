#include "Camera.h"
#include <algorithm>
#include <cmath>

Camera::Camera(glm::vec3 target, float distance)
    : target(target)
    , up(0.0f, 1.0f, 0.0f)
    , distance(distance)
    , yaw(0.0f)
    , pitch(0.4f)  // Start looking slightly down
{
    updateVectors();
}

void Camera::processMouseMovement(float deltaX, float deltaY, bool rotating, bool panning) {
    if (rotating) {
        yaw += deltaX * rotationSensitivity;
        pitch += deltaY * rotationSensitivity;
        
        // Clamp pitch to avoid flipping
        pitch = std::clamp(pitch, minPitch, maxPitch);
        
        updateVectors();
    }
    else if (panning) {
        // Calculate right and up vectors for panning
        glm::vec3 pos = getPosition();
        glm::vec3 forward = glm::normalize(target - pos);
        glm::vec3 right = glm::normalize(glm::cross(forward, up));
        glm::vec3 camUp = glm::normalize(glm::cross(right, forward));
        
        // Pan in screen space
        float panScale = distance * panSensitivity;
        target -= right * deltaX * panScale;
        target += camUp * deltaY * panScale;
    }
}

void Camera::processScroll(float delta) {
    distance -= delta * zoomSensitivity * distance;
    distance = std::clamp(distance, minDistance, maxDistance);
}

glm::mat4 Camera::getViewMatrix() const {
    return glm::lookAt(getPosition(), target, up);
}

glm::mat4 Camera::getProjectionMatrix(float aspectRatio) const {
    return glm::perspective(glm::radians(fov), aspectRatio, nearPlane, farPlane);
}

glm::vec3 Camera::getPosition() const {
    // Spherical coordinates to Cartesian
    float x = distance * std::cos(pitch) * std::sin(yaw);
    float y = distance * std::sin(pitch);
    float z = distance * std::cos(pitch) * std::cos(yaw);
    
    return target + glm::vec3(x, y, z);
}

void Camera::updateVectors() {
    // Recalculate up vector based on camera orientation
    // Keep global up for now
    up = glm::vec3(0.0f, 1.0f, 0.0f);
}

