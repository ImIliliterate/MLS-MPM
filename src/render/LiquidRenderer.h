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
 * - Screen-Space Fluid Rendering (SSFR) for smooth surfaces
 * - Point sprites fallback
 * - Marching cubes mesh (CPU)
 */
class LiquidRenderer {
public:
    enum class RenderMode {
        Points,      // Basic point sprites
        SSFR,        // Screen-space fluid rendering (smooth!)
        Mesh         // Marching cubes mesh
    };
    
    RenderMode mode = RenderMode::Mesh;  // Default to marching cubes mesh (Zhu & Bridson)
    
    // Point rendering parameters
    float pointSize = 4.0f;
    glm::vec3 pointColor{0.15f, 0.45f, 0.85f};
    
    // SSFR parameters - particles must overlap to look like fluid
    float ssfrPointScale = 3000.0f;  // Large enough to overlap
    float ssfrParticleRadius = 0.07f; // Must be > particle spacing for overlap
    int ssfrSmoothIterations = 20;   // Smooth out seams
    float ssfrSmoothDt = 0.004f;     // Fill gaps between particles
    glm::vec3 ssfrWaterColor{0.05f, 0.2f, 0.4f};      // Deep water color
    glm::vec3 ssfrSurfaceColor{0.3f, 0.6f, 0.85f};    // Surface/shallow color (brighter)
    glm::vec3 ssfrSpecular{1.0f, 1.0f, 1.0f};         // Specular highlight
    float ssfrShininess = 400.0f;                     // Very sharp water specular
    float ssfrFresnelBias = 0.04f;                    // Slight base reflectivity
    float ssfrFresnelScale = 0.96f;                   // Strong fresnel
    float ssfrFresnelPower = 4.0f;                    // Fresnel curve
    float ssfrThicknessScale = 2.5f;                  // Depth-based absorption
    
    // Mesh rendering parameters
    float isoLevel = 0.5f;
    glm::vec3 meshColor{0.05f, 0.25f, 0.6f};  // Water color as specified
    float shininess = 64.0f;
    float fresnel = 1.0f;
    
    LiquidRenderer();
    ~LiquidRenderer();
    
    bool init();
    void cleanup();
    
    void updateParticles(const std::vector<Particle>& particles);
    void updatePositions(const std::vector<glm::vec3>& positions);
    void generateMesh(const MpmSim& sim, float isoLevel = 0.5f);
    
    void render(const glm::mat4& view, const glm::mat4& projection, 
                const glm::vec3& cameraPos, const glm::vec3& lightDir,
                int viewportWidth = 1280, int viewportHeight = 720);
    
    // Resize SSFR framebuffers when window size changes
    void resizeSSFR(int width, int height);
    
    size_t getParticleCount() const { return particleCount; }
    size_t getTriangleCount() const { return triangleCount; }
    
private:
    // Point rendering
    GLuint pointVAO = 0;
    GLuint pointVBO = 0;
    size_t particleCount = 0;
    Shader pointShader;
    
    // SSFR rendering
    int ssfrWidth = 0, ssfrHeight = 0;
    GLuint ssfrDepthFBO = 0;      // Framebuffer for depth pass
    GLuint ssfrDepthTex = 0;      // Depth texture
    GLuint ssfrBlurFBO = 0;       // Framebuffer for blur pass
    GLuint ssfrBlurTex = 0;       // Blurred depth texture
    GLuint ssfrThickFBO = 0;      // Thickness accumulation
    GLuint ssfrThickTex = 0;      // Thickness texture
    GLuint ssfrQuadVAO = 0;       // Fullscreen quad
    GLuint ssfrQuadVBO = 0;
    Shader ssfrDepthShader;       // Render particle depths
    Shader ssfrThickShader;       // Render thickness (additive)
    Shader ssfrBlurShader;        // Bilateral blur
    Shader ssfrCompositeShader;   // Final composite with normals & shading
    
    bool initSSFR();
    void cleanupSSFR();
    void renderSSFR(const glm::mat4& view, const glm::mat4& projection,
                    const glm::vec3& cameraPos, const glm::vec3& lightDir,
                    int viewportWidth, int viewportHeight);
    
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
    
    void buildScalarField(const MpmSim& sim);
    void marchingCubes();
    
    std::vector<float> meshVertices;
    std::vector<unsigned int> meshIndices;
    
    bool initPointRendering();
    bool initMeshRendering();
};

