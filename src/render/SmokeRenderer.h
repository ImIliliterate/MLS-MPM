#pragma once

#ifndef GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_NONE
#endif
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <vector>
#include "Shader.h"
#include "../sim/SmokeSim.h"

/**
 * Volume renderer for smoke using ray-marching
 * Renders a fullscreen quad and ray-marches through the 3D density texture
 */
class SmokeRenderer {
public:
    // Rendering parameters
    int raySteps = 12;                      // Ray-marching steps (minimal for perf)
    float densityScale = 25.0f;             // Density multiplier - boosted for visibility
    float absorptionCoeff = 0.8f;           // Absorption coefficient - more opaque
    glm::vec3 smokeColor{0.85f, 0.88f, 0.95f}; // Slightly blue-tinted smoke
    glm::vec3 lightColor{1.0f, 0.95f, 0.9f}; // Light color
    float shadowSteps = 4;                   // Shadow ray steps (reduced for perf)
    float shadowDensity = 0.4f;             // Shadow density multiplier
    
    SmokeRenderer();
    ~SmokeRenderer();
    
    bool init();
    void cleanup();
    
    // Update density texture from simulation
    void updateDensity(const SmokeSim& sim);
    
    // Render the smoke volume
    void render(const glm::mat4& view, const glm::mat4& projection,
                const glm::vec3& cameraPos, const glm::vec3& lightDir,
                const glm::vec3& volumeMin, const glm::vec3& volumeMax);
    
private:
    Shader shader;
    
    // Fullscreen quad
    GLuint quadVAO = 0;
    GLuint quadVBO = 0;
    
    // 3D density texture
    GLuint densityTexture = 0;
    int texWidth = 0, texHeight = 0, texDepth = 0;
    
    bool initQuad();
    bool initShader();
};

