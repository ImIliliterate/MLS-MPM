#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

/**
 * Orbit camera for viewing the simulation
 * Allows rotation around a target point with mouse controls
 */
class Camera {
public:
    Camera(glm::vec3 target = glm::vec3(0.5f), float distance = 2.0f);
    
    // Update camera based on mouse input
    void processMouseMovement(float deltaX, float deltaY, bool rotating, bool panning);
    void processScroll(float delta);
    
    // Get matrices
    glm::mat4 getViewMatrix() const;
    glm::mat4 getProjectionMatrix(float aspectRatio) const;
    glm::vec3 getPosition() const;
    glm::vec3 getTarget() const { return target; }
    glm::vec3 getUp() const { return up; }
    
    // Camera parameters
    float fov = 45.0f;
    float nearPlane = 0.01f;
    float farPlane = 100.0f;
    
    // Sensitivity
    float rotationSensitivity = 0.005f;
    float panSensitivity = 0.002f;
    float zoomSensitivity = 0.1f;
    
private:
    void updateVectors();
    
    glm::vec3 target;       // Look-at target
    glm::vec3 up;           // Up vector
    
    float distance;         // Distance from target
    float yaw;              // Horizontal angle (radians)
    float pitch;            // Vertical angle (radians)
    
    // Clamp values
    float minPitch = -1.5f; // ~-85 degrees
    float maxPitch = 1.5f;  // ~85 degrees
    float minDistance = 0.1f;
    float maxDistance = 20.0f;
};

