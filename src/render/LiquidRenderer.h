#pragma once

#ifndef GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_NONE
#endif
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <vector>
#include "Shader.h"
#include "../sim/MpmSim.h"

/**
 * Renders liquid from MPM simulation
 * - Point sprites for particles
 * - Marching cubes mesh for surface
 */
class LiquidRenderer {
public:
    enum class RenderMode {
        Points,
        Mesh
    };
    
    RenderMode mode = RenderMode::Points;
    
    // Point rendering parameters
    float pointSize = 4.0f;
    glm::vec3 pointColor{0.15f, 0.45f, 0.85f};  // Nice water blue
    
    // Mesh rendering parameters
    float isoLevel = 0.5f;  // Adjust based on particle density
    glm::vec3 meshColor{0.1f, 0.35f, 0.7f};  // Deeper water blue
    float shininess = 128.0f;  // Higher for sharper highlights
    float fresnel = 0.7f;  // More rim lighting for watery look
    
    LiquidRenderer();
    ~LiquidRenderer();
    
    // Initialize rendering resources
    bool init();
    void cleanup();
    
    // Update particle data from simulation
    void updateParticles(const std::vector<Particle>& particles);
    
    // Generate mesh from particles using marching cubes
    void generateMesh(const MpmSim& sim, float isoLevel = 0.5f);
    
    // Render the liquid
    void render(const glm::mat4& view, const glm::mat4& projection, 
                const glm::vec3& cameraPos, const glm::vec3& lightDir);
    
    // Get particle count
    size_t getParticleCount() const { return particleCount; }
    size_t getTriangleCount() const { return triangleCount; }
    
private:
    // Point rendering
    GLuint pointVAO = 0;
    GLuint pointVBO = 0;
    size_t particleCount = 0;
    Shader pointShader;
    
    // Mesh rendering
    GLuint meshVAO = 0;
    GLuint meshVBO = 0;
    GLuint meshEBO = 0;
    size_t triangleCount = 0;
    size_t indexCount = 0;
    Shader meshShader;
    
    // Scalar field for marching cubes
    std::vector<float> scalarField;
    int fieldNx = 0, fieldNy = 0, fieldNz = 0;
    float fieldDx = 0.0f;
    glm::vec3 fieldMin{0.0f};
    
    // Marching cubes helpers
    void buildScalarField(const MpmSim& sim);
    void marchingCubes();
    
    // Mesh data
    std::vector<float> meshVertices;  // pos.xyz, normal.xyz
    std::vector<unsigned int> meshIndices;
    
    bool initPointRendering();
    bool initMeshRendering();
};

