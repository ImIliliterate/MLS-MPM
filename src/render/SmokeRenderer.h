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
    // Rendering parameters - very visible fog
    int raySteps = 48;                      // Good quality
    float densityScale = 300.0f;            // VERY high for visibility
    float absorptionCoeff = 0.5f;           // Balanced absorption
    glm::vec3 smokeColor{0.9f, 0.93f, 1.0f}; // Slightly blue fog
    glm::vec3 lightColor{1.0f, 0.95f, 0.9f}; // Warm light
    float shadowSteps = 4;                   // Perf
    float shadowDensity = 0.15f;            // Soft shadows
    float scatteringG = 0.5f;               // Forward scattering
    float ambientStrength = 0.5f;           // Strong ambient
    
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

