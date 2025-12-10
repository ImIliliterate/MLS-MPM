#include "LiquidRenderer.h"
#include "../sim/SimulationParams.h"
#include <glad/glad.h>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <algorithm>
#include <cmath>
#include <unordered_map>

// Marching cubes lookup tables
namespace MarchingCubes {
    // Edge table - which edges are intersected for each cube configuration
    extern const int edgeTable[256];
    extern const int triTable[256][16];
}

LiquidRenderer::LiquidRenderer() = default;

LiquidRenderer::~LiquidRenderer() {
    cleanup();
}

bool LiquidRenderer::init() {
    if (!initPointRendering()) return false;
    if (!initMeshRendering()) return false;
    if (!initSSFR()) return false;  // GPU-accelerated fluid rendering
    return true;
}

void LiquidRenderer::cleanup() {
    if (pointVAO) { glDeleteVertexArrays(1, &pointVAO); pointVAO = 0; }
    if (pointVBO) { glDeleteBuffers(1, &pointVBO); pointVBO = 0; }
    if (meshVAO) { glDeleteVertexArrays(1, &meshVAO); meshVAO = 0; }
    if (meshVBO) { glDeleteBuffers(1, &meshVBO); meshVBO = 0; }
    if (meshEBO) { glDeleteBuffers(1, &meshEBO); meshEBO = 0; }
    // SSFR cleanup disabled
}

// SSFR cleanup and resize are defined later in the file (in #if 1 block)

bool LiquidRenderer::initPointRendering() {
    // DEBUG: Simple 2-pixel dots colored by velocity
    const char* vertSrc = R"(
        #version 330 core
        layout(location = 0) in vec3 aPos;
        
        uniform mat4 uView;
        uniform mat4 uProj;
        
        void main() {
            gl_Position = uProj * uView * vec4(aPos, 1.0);
            gl_PointSize = 2.0;  // 2 pixel dots
        }
    )";
    
    const char* fragSrc = R"(
        #version 330 core
        out vec4 FragColor;
        
        void main() {
            // Simple blue dot
            FragColor = vec4(0.2, 0.5, 1.0, 1.0);
        }
    )";
    
    if (!pointShader.loadFromSource(vertSrc, fragSrc)) {
        return false;
    }
    
    glGenVertexArrays(1, &pointVAO);
    glGenBuffers(1, &pointVBO);
    
    glBindVertexArray(pointVAO);
    glBindBuffer(GL_ARRAY_BUFFER, pointVBO);
    
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    glBindVertexArray(0);
    
    return true;
}

bool LiquidRenderer::initMeshRendering() {
    // Water-like mesh shader with improved lighting
    const char* vertSrc = R"(
        #version 330 core
        layout(location = 0) in vec3 aPos;
        layout(location = 1) in vec3 aNormal;
        
        uniform mat4 uModel;
        uniform mat4 uView;
        uniform mat4 uProj;
        
        out vec3 vWorldPos;
        out vec3 vNormal;
        out vec3 vViewPos;
        
        void main() {
            vec4 worldPos = uModel * vec4(aPos, 1.0);
            vWorldPos = worldPos.xyz;
            vNormal = mat3(transpose(inverse(uModel))) * aNormal;
            vec4 viewPos = uView * worldPos;
            vViewPos = viewPos.xyz;
            gl_Position = uProj * viewPos;
        }
    )";
    
    // Exact shader from ChatGPT specification
    const char* fragSrc = R"(
        #version 330 core
        in vec3 vWorldPos;
        in vec3 vNormal;
        in vec3 vViewPos;
        
        out vec4 FragColor;
        
        uniform vec3 uCameraPos;
        uniform vec3 uLightDir;
        uniform vec3 uColor;  // uWaterColor: e.g. vec3(0.05, 0.25, 0.6)
        uniform float uShininess;
        uniform float uFresnel;
        
        void main() {
            vec3 N = normalize(vNormal);
            vec3 V = normalize(uCameraPos - vWorldPos);
            vec3 L = normalize(uLightDir);
            
            float NdotL = max(dot(N, L), 0.0);
            float fresnel = pow(1.0 - max(dot(N, V), 0.0), 3.0);
            
            vec3 diffuse = uColor * (0.2 + 0.8 * NdotL);
            vec3 spec = vec3(1.0) * pow(max(dot(reflect(-L, N), V), 0.0), 64.0) * 0.4;
            vec3 reflectionTint = vec3(0.5, 0.6, 0.7);
            
            vec3 baseColor = diffuse + spec;
            vec3 color = mix(baseColor, reflectionTint, fresnel);
            
            float alpha = 0.9;  // mostly opaque
            FragColor = vec4(color, alpha);
        }
    )";
    
    if (!meshShader.loadFromSource(vertSrc, fragSrc)) {
        return false;
    }
    
    glGenVertexArrays(1, &meshVAO);
    glGenBuffers(1, &meshVBO);
    glGenBuffers(1, &meshEBO);
    
    return true;
}

void LiquidRenderer::updateParticles(const std::vector<Particle>& particles) {
    particleCount = particles.size();
    
    if (particleCount == 0) return;
    
    std::vector<float> positions(particleCount * 3);
    for (size_t i = 0; i < particleCount; i++) {
        positions[i * 3 + 0] = particles[i].x.x;
        positions[i * 3 + 1] = particles[i].x.y;
        positions[i * 3 + 2] = particles[i].x.z;
    }
    
    glBindBuffer(GL_ARRAY_BUFFER, pointVBO);
    glBufferData(GL_ARRAY_BUFFER, positions.size() * sizeof(float), positions.data(), GL_DYNAMIC_DRAW);
}

void LiquidRenderer::updatePositions(const std::vector<glm::vec3>& positions) {
    particleCount = positions.size();
    
    if (particleCount == 0) return;
    
    // glm::vec3 is already tightly packed as 3 floats
    glBindBuffer(GL_ARRAY_BUFFER, pointVBO);
    glBufferData(GL_ARRAY_BUFFER, particleCount * sizeof(glm::vec3), positions.data(), GL_DYNAMIC_DRAW);
}

void LiquidRenderer::generateMesh(const MpmSim& sim, float iso) {
    isoLevel = iso;
    buildScalarField(sim);
    marchingCubes();
    
    if (meshVertices.empty()) {
        triangleCount = 0;
        return;
    }
    
    // Upload mesh data
    glBindVertexArray(meshVAO);
    
    glBindBuffer(GL_ARRAY_BUFFER, meshVBO);
    glBufferData(GL_ARRAY_BUFFER, meshVertices.size() * sizeof(float), meshVertices.data(), GL_DYNAMIC_DRAW);
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, meshEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, meshIndices.size() * sizeof(unsigned int), meshIndices.data(), GL_DYNAMIC_DRAW);
    
    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    // Normal attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    glBindVertexArray(0);
    
    triangleCount = meshIndices.size() / 3;
    indexCount = meshIndices.size();
}

void LiquidRenderer::render(const glm::mat4& view, const glm::mat4& projection,
                            const glm::vec3& cameraPos, const glm::vec3& lightDir,
                            int viewportWidth, int viewportHeight) {
    // SSFR - GPU-accelerated smooth fluid rendering
    if (mode == RenderMode::SSFR) {
        if (particleCount == 0) return;
        renderSSFR(view, projection, cameraPos, lightDir, viewportWidth, viewportHeight);
        return;
    }
    
    // Points - fast debug rendering
    if (mode == RenderMode::Points) {
        if (particleCount == 0) return;
        
        glEnable(GL_PROGRAM_POINT_SIZE);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_DEPTH_TEST);
        
        pointShader.use();
        pointShader.setMat4("uView", view);
        pointShader.setMat4("uProj", projection);
        pointShader.setFloat("uPointSize", pointSize);
        pointShader.setVec3("uColor", pointColor);
        pointShader.setVec3("uLightDir", lightDir);
        
        glBindVertexArray(pointVAO);
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(particleCount));
        glBindVertexArray(0);
        
        glDisable(GL_BLEND);
        return;
    }
    
    // Mesh - CPU marching cubes (slow but detailed)
    if (mode == RenderMode::Mesh) {
        if (indexCount == 0) return;
        
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        
        meshShader.use();
        meshShader.setMat4("uModel", glm::mat4(1.0f));
        meshShader.setMat4("uView", view);
        meshShader.setMat4("uProj", projection);
        meshShader.setVec3("uCameraPos", cameraPos);
        meshShader.setVec3("uLightDir", lightDir);
        meshShader.setVec3("uColor", meshColor);
        meshShader.setFloat("uShininess", shininess);
        meshShader.setFloat("uFresnel", fresnel);
        
        glBindVertexArray(meshVAO);
        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(indexCount), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
        
        glDisable(GL_BLEND);
    }
}

// ============== Screen-Space Fluid Rendering ==============
// Fast GPU-based fluid surface rendering
// Much faster than CPU marching cubes with better quality

#if 1  // SSFR ENABLED
bool LiquidRenderer::initSSFR() {
    // Depth pass shader - renders particles as spheres to depth buffer
    const char* depthVertSrc = R"(
        #version 330 core
        layout(location = 0) in vec3 aPos;
        
        uniform mat4 uView;
        uniform mat4 uProj;
        uniform float uPointScale;
        uniform float uPointRadius;
        
        out vec3 vViewPos;
        out float vRadius;
        
        void main() {
            vec4 viewPos = uView * vec4(aPos, 1.0);
            vViewPos = viewPos.xyz;
            vRadius = uPointRadius;
            gl_Position = uProj * viewPos;
            // Scale point size by distance for perspective-correct spheres
            gl_PointSize = uPointScale * uPointRadius / max(-viewPos.z, 0.1);
        }
    )";
    
    const char* depthFragSrc = R"(
        #version 330 core
        in vec3 vViewPos;
        in float vRadius;
        
        uniform mat4 uProj;
        
        out vec4 FragColor;
        
        void main() {
            // Compute sphere normal from point coord
            vec2 coord = gl_PointCoord * 2.0 - 1.0;
            float r2 = dot(coord, coord);
            if (r2 > 1.0) discard;
            
            // Sphere depth offset
            float z = sqrt(1.0 - r2);
            vec3 spherePos = vViewPos + vec3(coord.x * vRadius, -coord.y * vRadius, z * vRadius);
            
            // Output linear depth
            FragColor = vec4(-spherePos.z, 0.0, 0.0, 1.0);
            
            // Also write to depth buffer for proper occlusion
            vec4 clipPos = uProj * vec4(spherePos, 1.0);
            float ndcDepth = clipPos.z / clipPos.w;
            gl_FragDepth = ndcDepth * 0.5 + 0.5;
        }
    )";
    
    if (!ssfrDepthShader.loadFromSource(depthVertSrc, depthFragSrc)) {
        return false;
    }
    
    // Thickness pass shader - outputs sphere thickness for additive blending
    const char* thickFragSrc = R"(
        #version 330 core
        in vec3 vViewPos;
        in float vRadius;
        
        out vec4 FragColor;
        
        void main() {
            // Compute sphere from point coord
            vec2 coord = gl_PointCoord * 2.0 - 1.0;
            float r2 = dot(coord, coord);
            if (r2 > 1.0) discard;
            
            // Thickness = full diameter at this slice of the sphere
            float z = sqrt(1.0 - r2);
            float thickness = 2.0 * vRadius * z;  // Diameter at this height
            
            // Output thickness (will be accumulated additively)
            FragColor = vec4(thickness, 0.0, 0.0, 1.0);
        }
    )";
    
    if (!ssfrThickShader.loadFromSource(depthVertSrc, thickFragSrc)) {
        return false;
    }
    
    // Fullscreen quad vertex shader (shared)
    const char* blurVertSrc = R"(
        #version 330 core
        layout(location = 0) in vec2 aPos;
        layout(location = 1) in vec2 aTexCoord;
        out vec2 vTexCoord;
        void main() {
            vTexCoord = aTexCoord;
            gl_Position = vec4(aPos, 0.0, 1.0);
        }
    )";
    
    // Curvature flow smoothing shader (van der Laan et al.)
    // Smooths depth field based on mean curvature - much better than bilateral blur
    const char* blurFragSrc = R"(
        #version 330 core
        in vec2 vTexCoord;
        out vec4 FragColor;
        
        uniform sampler2D uDepthTex;
        uniform vec2 uTexelSize;
        uniform float uDt;  // Smoothing timestep
        
        void main() {
            float depth = texture(uDepthTex, vTexCoord).r;
            if (depth <= 0.0) {
                FragColor = vec4(0.0);
                return;
            }
            
            // Sample neighbors
            float dxp = texture(uDepthTex, vTexCoord + vec2(uTexelSize.x, 0.0)).r;
            float dxn = texture(uDepthTex, vTexCoord - vec2(uTexelSize.x, 0.0)).r;
            float dyp = texture(uDepthTex, vTexCoord + vec2(0.0, uTexelSize.y)).r;
            float dyn = texture(uDepthTex, vTexCoord - vec2(0.0, uTexelSize.y)).r;
            
            // Handle boundaries - use current depth if neighbor is empty
            if (dxp <= 0.0) dxp = depth;
            if (dxn <= 0.0) dxn = depth;
            if (dyp <= 0.0) dyp = depth;
            if (dyn <= 0.0) dyn = depth;
            
            // Compute curvature using Laplacian
            // Mean curvature flow: d/dt(z) = H where H is mean curvature
            float laplacian = (dxp + dxn + dyp + dyn) - 4.0 * depth;
            
            // Update depth based on curvature (explicit Euler)
            float newDepth = depth + uDt * laplacian;
            
            // Keep depth positive
            newDepth = max(newDepth, 0.001);
            
            FragColor = vec4(newDepth, 0.0, 0.0, 1.0);
        }
    )";
    
    if (!ssfrBlurShader.loadFromSource(blurVertSrc, blurFragSrc)) {
        return false;
    }
    
    // Composite shader - realistic water shading with Fresnel and thickness
    const char* compositeFragSrc = R"(
        #version 330 core
        in vec2 vTexCoord;
        out vec4 FragColor;
        
        uniform sampler2D uDepthTex;
        uniform sampler2D uThickTex;   // Accumulated thickness
        uniform mat4 uProjInv;
        uniform vec2 uTexelSize;
        uniform vec3 uLightDir;
        uniform vec3 uWaterColor;      // Deep water color
        uniform vec3 uSurfaceColor;    // Shallow/surface color
        uniform vec3 uSpecularColor;
        uniform float uShininess;
        uniform float uFresnelBias;
        uniform float uFresnelScale;
        uniform float uFresnelPower;
        uniform float uThicknessScale;
        
        vec3 uvToView(vec2 uv, float depth) {
            vec4 clip = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
            vec4 view = uProjInv * clip;
            view.xyz /= view.w;
            return vec3(view.xy, -depth);
        }
        
        void main() {
            float depth = texture(uDepthTex, vTexCoord).r;
            float thickness = texture(uThickTex, vTexCoord).r;
            if (depth <= 0.0) discard;
            
            // Sample neighbors for normal reconstruction (central differences)
            float dxp = texture(uDepthTex, vTexCoord + vec2(uTexelSize.x, 0.0)).r;
            float dxn = texture(uDepthTex, vTexCoord - vec2(uTexelSize.x, 0.0)).r;
            float dyp = texture(uDepthTex, vTexCoord + vec2(0.0, uTexelSize.y)).r;
            float dyn = texture(uDepthTex, vTexCoord - vec2(0.0, uTexelSize.y)).r;
            
            // Reconstruct view-space position
            vec3 viewPos = uvToView(vTexCoord, depth);
            
            // Compute normal from depth gradient using central differences
            vec3 ddx = uvToView(vTexCoord + vec2(uTexelSize.x, 0.0), dxp > 0.0 ? dxp : depth) -
                       uvToView(vTexCoord - vec2(uTexelSize.x, 0.0), dxn > 0.0 ? dxn : depth);
            vec3 ddy = uvToView(vTexCoord + vec2(0.0, uTexelSize.y), dyp > 0.0 ? dyp : depth) -
                       uvToView(vTexCoord - vec2(0.0, uTexelSize.y), dyn > 0.0 ? dyn : depth);
            vec3 crossN = cross(ddy, ddx);
            float len = length(crossN);
            
            // Fallback to camera-facing normal if gradient is too small
            vec3 normal = len > 0.0001 ? crossN / len : vec3(0.0, 0.0, 1.0);
            
            // Ensure normal points toward camera
            if (normal.z < 0.0) normal = -normal;
            
            // View and light directions
            vec3 viewDir = normalize(-viewPos);
            vec3 lightDir = normalize(-uLightDir);
            vec3 halfDir = normalize(lightDir + viewDir);
            vec3 reflectDir = reflect(-viewDir, normal);
            
            // === FRESNEL (Schlick approximation) ===
            float NdotV = max(dot(normal, viewDir), 0.0);
            float fresnel = uFresnelBias + uFresnelScale * pow(1.0 - NdotV, uFresnelPower);
            fresnel = clamp(fresnel, 0.0, 1.0);
            
            // === WATER COLOR with thickness-based absorption (Beer's Law) ===
            float thicknessScaled = thickness * uThicknessScale;
            vec3 absorption = exp(-thicknessScaled * vec3(0.4, 0.1, 0.05)); // Water absorbs red first
            vec3 waterColor = uSurfaceColor * absorption;
            
            // === KEY LIGHT (main sun/directional) ===
            float NdotL = max(dot(normal, lightDir), 0.0);
            float wrapLight = NdotL * 0.5 + 0.5; // Wrap lighting for softer look
            vec3 keyLight = vec3(1.0, 0.95, 0.9) * wrapLight * 0.7;
            
            // === FILL LIGHT (from below/side for depth) ===
            vec3 fillDir = normalize(vec3(0.5, -0.3, 0.5));
            float fillNdotL = max(dot(normal, fillDir), 0.0);
            vec3 fillLight = vec3(0.4, 0.5, 0.7) * fillNdotL * 0.3;
            
            // === RIM LIGHT (backlight for silhouette) ===
            vec3 rimDir = normalize(vec3(0.0, 0.5, -1.0));
            float rimNdotL = max(dot(normal, rimDir), 0.0);
            float rim = pow(1.0 - NdotV, 3.0) * rimNdotL;
            vec3 rimLight = vec3(0.6, 0.8, 1.0) * rim * 0.5;
            
            // === SPECULAR (sharp water highlights) ===
            float NdotH = max(dot(normal, halfDir), 0.0);
            float specular = pow(NdotH, uShininess);
            // Secondary specular for fill light
            vec3 halfFill = normalize(fillDir + viewDir);
            float specFill = pow(max(dot(normal, halfFill), 0.0), uShininess * 0.5) * 0.3;
            
            // === ENVIRONMENT REFLECTION (gradient sky dome) ===
            float skyBlend = reflectDir.y * 0.5 + 0.5;
            vec3 skyColor = mix(
                vec3(0.7, 0.8, 0.95),  // Horizon
                vec3(0.3, 0.5, 0.9),   // Zenith
                pow(skyBlend, 0.8)
            );
            // Add subtle sun reflection
            float sunReflect = pow(max(dot(reflectDir, lightDir), 0.0), 64.0);
            skyColor += vec3(1.0, 0.9, 0.7) * sunReflect * 0.8;
            
            // === COMBINE LIGHTING ===
            vec3 diffuseLight = keyLight + fillLight;
            vec3 color = waterColor * diffuseLight * (1.0 - fresnel * 0.5);
            color += skyColor * fresnel * 0.7;  // Environment reflection
            color += rimLight;                   // Rim/backlight
            color += uSpecularColor * (specular + specFill) * 1.5;  // Specular highlights
            
            // Subtle caustics approximation (fake)
            float caustic = pow(max(dot(normal, vec3(0, 1, 0)), 0.0), 2.0) * 0.1;
            color += vec3(0.5, 0.7, 1.0) * caustic;
            
            // Alpha based on thickness + fresnel (thicker = more opaque)
            float thickAlpha = 1.0 - exp(-thicknessScaled * 2.0);  // Exponential falloff
            float alpha = mix(thickAlpha * 0.85, 0.98, fresnel);
            
            FragColor = vec4(color, alpha);
        }
    )";
    
    if (!ssfrCompositeShader.loadFromSource(blurVertSrc, compositeFragSrc)) {
        return false;
    }
    
    // Create fullscreen quad
    float quadVerts[] = {
        // pos        // texcoord
        -1.0f, -1.0f, 0.0f, 0.0f,
         1.0f, -1.0f, 1.0f, 0.0f,
         1.0f,  1.0f, 1.0f, 1.0f,
        -1.0f, -1.0f, 0.0f, 0.0f,
         1.0f,  1.0f, 1.0f, 1.0f,
        -1.0f,  1.0f, 0.0f, 1.0f
    };
    
    glGenVertexArrays(1, &ssfrQuadVAO);
    glGenBuffers(1, &ssfrQuadVBO);
    glBindVertexArray(ssfrQuadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, ssfrQuadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVerts), quadVerts, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);
    
    return true;
}

void LiquidRenderer::cleanupSSFR() {
    if (ssfrDepthFBO) { glDeleteFramebuffers(1, &ssfrDepthFBO); ssfrDepthFBO = 0; }
    if (ssfrDepthTex) { glDeleteTextures(1, &ssfrDepthTex); ssfrDepthTex = 0; }
    if (ssfrBlurFBO) { glDeleteFramebuffers(1, &ssfrBlurFBO); ssfrBlurFBO = 0; }
    if (ssfrBlurTex) { glDeleteTextures(1, &ssfrBlurTex); ssfrBlurTex = 0; }
    if (ssfrThickFBO) { glDeleteFramebuffers(1, &ssfrThickFBO); ssfrThickFBO = 0; }
    if (ssfrThickTex) { glDeleteTextures(1, &ssfrThickTex); ssfrThickTex = 0; }
    if (ssfrQuadVAO) { glDeleteVertexArrays(1, &ssfrQuadVAO); ssfrQuadVAO = 0; }
    if (ssfrQuadVBO) { glDeleteBuffers(1, &ssfrQuadVBO); ssfrQuadVBO = 0; }
}

void LiquidRenderer::resizeSSFR(int width, int height) {
    if (width == ssfrWidth && height == ssfrHeight) return;
    ssfrWidth = width;
    ssfrHeight = height;
    
    // Cleanup old textures
    if (ssfrDepthTex) glDeleteTextures(1, &ssfrDepthTex);
    if (ssfrBlurTex) glDeleteTextures(1, &ssfrBlurTex);
    if (ssfrDepthFBO) glDeleteFramebuffers(1, &ssfrDepthFBO);
    if (ssfrBlurFBO) glDeleteFramebuffers(1, &ssfrBlurFBO);
    
    // Create depth texture and FBO
    glGenTextures(1, &ssfrDepthTex);
    glBindTexture(GL_TEXTURE_2D, ssfrDepthTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    
    glGenFramebuffers(1, &ssfrDepthFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, ssfrDepthFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ssfrDepthTex, 0);
    
    // Add depth texture for proper depth testing
    GLuint depthTex;
    glGenTextures(1, &depthTex);
    glBindTexture(GL_TEXTURE_2D, depthTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTex, 0);
    
    // Create blur texture and FBO
    glGenTextures(1, &ssfrBlurTex);
    glBindTexture(GL_TEXTURE_2D, ssfrBlurTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    
    glGenFramebuffers(1, &ssfrBlurFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, ssfrBlurFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ssfrBlurTex, 0);
    
    // Create thickness texture and FBO (for additive accumulation)
    if (ssfrThickTex) glDeleteTextures(1, &ssfrThickTex);
    if (ssfrThickFBO) glDeleteFramebuffers(1, &ssfrThickFBO);
    
    glGenTextures(1, &ssfrThickTex);
    glBindTexture(GL_TEXTURE_2D, ssfrThickTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);  // Linear for smooth sampling
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    
    glGenFramebuffers(1, &ssfrThickFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, ssfrThickFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ssfrThickTex, 0);
    
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void LiquidRenderer::renderSSFR(const glm::mat4& view, const glm::mat4& projection,
                                 const glm::vec3& cameraPos, const glm::vec3& lightDir,
                                 int viewportWidth, int viewportHeight) {
    if (particleCount == 0) return;
    
    int width = viewportWidth;
    int height = viewportHeight;
    
    // Ensure FBOs are the right size
    resizeSSFR(width, height);
    
    // === Pass 1a: Render particle depths (closest surface) ===
    glBindFramebuffer(GL_FRAMEBUFFER, ssfrDepthFBO);
    glViewport(0, 0, width, height);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glDisable(GL_BLEND);
    
    ssfrDepthShader.use();
    ssfrDepthShader.setMat4("uView", view);
    ssfrDepthShader.setMat4("uProj", projection);
    ssfrDepthShader.setFloat("uPointScale", ssfrPointScale);
    ssfrDepthShader.setFloat("uPointRadius", ssfrParticleRadius);
    
    glBindVertexArray(pointVAO);
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(particleCount));
    
    // === Pass 1b: Render thickness (ADDITIVE - this makes particles merge!) ===
    glBindFramebuffer(GL_FRAMEBUFFER, ssfrThickFBO);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    
    glDisable(GL_DEPTH_TEST);  // No depth test - accumulate ALL particles
    glEnable(GL_BLEND);
    glBlendFunc(1, 1);  // GL_ONE, GL_ONE - ADDITIVE blending
    
    // Use thickness shader - outputs sphere thickness at each pixel
    ssfrThickShader.use();
    ssfrThickShader.setMat4("uView", view);
    ssfrThickShader.setMat4("uProj", projection);
    ssfrThickShader.setFloat("uPointScale", ssfrPointScale);
    ssfrThickShader.setFloat("uPointRadius", ssfrParticleRadius);
    
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(particleCount));
    
    // === Pass 2: Curvature flow smoothing ===
    // Iteratively smooth the depth field based on mean curvature
    glDisable(GL_DEPTH_TEST);
    
    glm::vec2 texelSize(1.0f / width, 1.0f / height);
    
    for (int iter = 0; iter < ssfrSmoothIterations; iter++) {
        // Smooth: depth -> blur
        glBindFramebuffer(GL_FRAMEBUFFER, ssfrBlurFBO);
        glClear(GL_COLOR_BUFFER_BIT);
        
        ssfrBlurShader.use();
        ssfrBlurShader.setInt("uDepthTex", 0);
        ssfrBlurShader.setVec2("uTexelSize", texelSize);
        ssfrBlurShader.setFloat("uDt", ssfrSmoothDt);
        
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, ssfrDepthTex);
        
        glBindVertexArray(ssfrQuadVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        
        // Copy back: blur -> depth
        glBindFramebuffer(GL_FRAMEBUFFER, ssfrDepthFBO);
        glClear(GL_COLOR_BUFFER_BIT);
        
        glBindTexture(GL_TEXTURE_2D, ssfrBlurTex);
        glDrawArrays(GL_TRIANGLES, 0, 6);
    }
    
    // === Pass 3: Composite - reconstruct normals and shade ===
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, width, height);
    
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    // Compute inverse projection
    glm::mat4 projInv = glm::inverse(projection);
    
    ssfrCompositeShader.use();
    ssfrCompositeShader.setInt("uDepthTex", 0);
    ssfrCompositeShader.setInt("uThickTex", 1);
    ssfrCompositeShader.setMat4("uProjInv", projInv);
    ssfrCompositeShader.setVec2("uTexelSize", texelSize);
    ssfrCompositeShader.setVec3("uLightDir", lightDir);
    ssfrCompositeShader.setVec3("uWaterColor", ssfrWaterColor);
    ssfrCompositeShader.setVec3("uSurfaceColor", ssfrSurfaceColor);
    ssfrCompositeShader.setVec3("uSpecularColor", ssfrSpecular);
    ssfrCompositeShader.setFloat("uShininess", ssfrShininess);
    ssfrCompositeShader.setFloat("uFresnelBias", ssfrFresnelBias);
    ssfrCompositeShader.setFloat("uFresnelScale", ssfrFresnelScale);
    ssfrCompositeShader.setFloat("uFresnelPower", ssfrFresnelPower);
    ssfrCompositeShader.setFloat("uThicknessScale", ssfrThicknessScale);
    
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, ssfrDepthTex);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, ssfrThickTex);
    
    glBindVertexArray(ssfrQuadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    
    glBindVertexArray(0);
    glDisable(GL_BLEND);
}
#endif  // SSFR ENABLED

// ============== Marching Cubes ==============

void LiquidRenderer::buildScalarField(const MpmSim& sim) {
    /**
     * Round 6 OPTIMIZED: Build scalar field with early bounds clamping
     * 
     * Optimizations:
     * - Pre-clamp loop bounds to avoid inner loop bounds checking
     * - Skip anisotropy computation when disabled
     * - Flatten array index computation
     */
    fieldNx = sim.Nx;
    fieldNy = sim.Ny;
    fieldNz = sim.Nz;
    fieldDx = sim.dx;
    fieldMin = sim.worldMin;
    
    // Zero the scalar field
    scalarField.assign(fieldNx * fieldNy * fieldNz, 0.0f);
    
    // Zhu & Bridson style: kernel radius = 2 * dx (EXACTLY as specified)
    float kernelRadius = 2.0f * fieldDx;
    float invKernelRadius = 1.0f / kernelRadius;
    int range = 2;  // 2 cells in each direction
    int strideY = fieldNx;
    int strideZ = fieldNx * fieldNy;
    
    // For each particle, distribute influence using simple (1-q)^2 kernel
    for (const auto& p : sim.particles) {
        // Convert particle position to grid indices
        glm::vec3 gridPos = (p.x - fieldMin) / fieldDx;
        int baseI = static_cast<int>(gridPos.x);
        int baseJ = static_cast<int>(gridPos.y);
        int baseK = static_cast<int>(gridPos.z);
        
        // Clamp loop bounds
        int iMin = std::max(0, baseI - range);
        int iMax = std::min(fieldNx - 1, baseI + range);
        int jMin = std::max(0, baseJ - range);
        int jMax = std::min(fieldNy - 1, baseJ + range);
        int kMin = std::max(0, baseK - range);
        int kMax = std::min(fieldNz - 1, baseK + range);
        
        for (int k = kMin; k <= kMax; k++) {
            float cz = fieldMin.z + (k + 0.5f) * fieldDx;
            float dz = p.x.z - cz;
            int idxK = k * strideZ;
            
            for (int j = jMin; j <= jMax; j++) {
                float cy = fieldMin.y + (j + 0.5f) * fieldDx;
                float dy = p.x.y - cy;
                int idxJK = j * strideY + idxK;
                
                for (int i = iMin; i <= iMax; i++) {
                    float cx = fieldMin.x + (i + 0.5f) * fieldDx;
                    float dx_diff = p.x.x - cx;
                    
                    // Distance from particle to cell center
                    float r = std::sqrt(dx_diff * dx_diff + dy * dy + dz * dz);
                    
                    if (r < kernelRadius) {
                        // Simple (1-q)^2 kernel as specified by Zhu & Bridson
                        float q = r * invKernelRadius;  // q in [0,1]
                        float w = (1.0f - q) * (1.0f - q);
                        scalarField[i + idxJK] += w;
                    }
                }
            }
        }
    }
    
    /**
     * Phase 3.2: Laplacian smoothing on scalar field
     * This reduces "bag of marbles" / "metaball blobby" artifacts
     * by smoothing the density field before meshing.
     * 
     * True Laplacian: phi_new[i] = (phi[i] + sum(phi[neighbors])) / (1 + neighborCount)
     */
    int numIterations = g_params.smoothingIterations;
    if (numIterations > 0) {
        std::vector<float> smoothed(scalarField.size());
        
        for (int iter = 0; iter < numIterations; iter++) {
            // Copy boundary cells
            for (size_t idx = 0; idx < scalarField.size(); idx++) {
                smoothed[idx] = scalarField[idx];
            }
            
            // Apply Laplacian smoothing to interior cells
            for (int k = 1; k < fieldNz - 1; k++) {
                for (int j = 1; j < fieldNy - 1; j++) {
                    for (int i = 1; i < fieldNx - 1; i++) {
                        int idx = i + fieldNx * (j + fieldNy * k);
                        
                        // 6-connected neighbors
                        float sum = scalarField[idx];  // Include self
                        sum += scalarField[idx - 1];   // -x
                        sum += scalarField[idx + 1];   // +x
                        sum += scalarField[idx - fieldNx];  // -y
                        sum += scalarField[idx + fieldNx];  // +y
                        sum += scalarField[idx - fieldNx * fieldNy];  // -z
                        sum += scalarField[idx + fieldNx * fieldNy];  // +z
                        
                        // Laplacian average: (self + 6 neighbors) / 7
                        smoothed[idx] = sum / 7.0f;
                    }
                }
            }
            
            // Swap buffers
            std::swap(scalarField, smoothed);
        }
    }
}

void LiquidRenderer::marchingCubes() {
    meshVertices.clear();
    meshIndices.clear();
    
    if (scalarField.empty()) return;
    
    // Vertex interpolation on edges
    auto vertexInterp = [&](const glm::vec3& p1, const glm::vec3& p2, float v1, float v2) -> glm::vec3 {
        if (std::abs(isoLevel - v1) < 1e-6f) return p1;
        if (std::abs(isoLevel - v2) < 1e-6f) return p2;
        if (std::abs(v1 - v2) < 1e-6f) return p1;
        float t = (isoLevel - v1) / (v2 - v1);
        return p1 + t * (p2 - p1);
    };
    
    // Compute gradient for normals
    auto computeNormal = [&](int i, int j, int k) -> glm::vec3 {
        auto sample = [&](int x, int y, int z) -> float {
            x = glm::clamp(x, 0, fieldNx - 1);
            y = glm::clamp(y, 0, fieldNy - 1);
            z = glm::clamp(z, 0, fieldNz - 1);
            return scalarField[x + fieldNx * (y + fieldNy * z)];
        };
        
        glm::vec3 n;
        n.x = sample(i - 1, j, k) - sample(i + 1, j, k);
        n.y = sample(i, j - 1, k) - sample(i, j + 1, k);
        n.z = sample(i, j, k - 1) - sample(i, j, k + 1);
        
        float len = glm::length(n);
        return len > 1e-6f ? n / len : glm::vec3(0, 1, 0);
    };
    
    // Map for vertex deduplication
    std::unordered_map<uint64_t, unsigned int> vertexMap;
    
    auto addVertex = [&](const glm::vec3& pos, const glm::vec3& normal) -> unsigned int {
        // Simple approach: just add vertex without deduplication for now
        unsigned int idx = static_cast<unsigned int>(meshVertices.size() / 6);
        meshVertices.push_back(pos.x);
        meshVertices.push_back(pos.y);
        meshVertices.push_back(pos.z);
        meshVertices.push_back(normal.x);
        meshVertices.push_back(normal.y);
        meshVertices.push_back(normal.z);
        return idx;
    };
    
    // Process each cell
    for (int k = 0; k < fieldNz - 1; k++) {
        for (int j = 0; j < fieldNy - 1; j++) {
            for (int i = 0; i < fieldNx - 1; i++) {
                // Get corner positions and values
                glm::vec3 corners[8];
                float values[8];
                
                corners[0] = fieldMin + glm::vec3(i, j, k) * fieldDx;
                corners[1] = fieldMin + glm::vec3(i + 1, j, k) * fieldDx;
                corners[2] = fieldMin + glm::vec3(i + 1, j + 1, k) * fieldDx;
                corners[3] = fieldMin + glm::vec3(i, j + 1, k) * fieldDx;
                corners[4] = fieldMin + glm::vec3(i, j, k + 1) * fieldDx;
                corners[5] = fieldMin + glm::vec3(i + 1, j, k + 1) * fieldDx;
                corners[6] = fieldMin + glm::vec3(i + 1, j + 1, k + 1) * fieldDx;
                corners[7] = fieldMin + glm::vec3(i, j + 1, k + 1) * fieldDx;
                
                values[0] = scalarField[i + fieldNx * (j + fieldNy * k)];
                values[1] = scalarField[(i + 1) + fieldNx * (j + fieldNy * k)];
                values[2] = scalarField[(i + 1) + fieldNx * ((j + 1) + fieldNy * k)];
                values[3] = scalarField[i + fieldNx * ((j + 1) + fieldNy * k)];
                values[4] = scalarField[i + fieldNx * (j + fieldNy * (k + 1))];
                values[5] = scalarField[(i + 1) + fieldNx * (j + fieldNy * (k + 1))];
                values[6] = scalarField[(i + 1) + fieldNx * ((j + 1) + fieldNy * (k + 1))];
                values[7] = scalarField[i + fieldNx * ((j + 1) + fieldNy * (k + 1))];
                
                // Determine cube index
                int cubeIndex = 0;
                if (values[0] < isoLevel) cubeIndex |= 1;
                if (values[1] < isoLevel) cubeIndex |= 2;
                if (values[2] < isoLevel) cubeIndex |= 4;
                if (values[3] < isoLevel) cubeIndex |= 8;
                if (values[4] < isoLevel) cubeIndex |= 16;
                if (values[5] < isoLevel) cubeIndex |= 32;
                if (values[6] < isoLevel) cubeIndex |= 64;
                if (values[7] < isoLevel) cubeIndex |= 128;
                
                if (MarchingCubes::edgeTable[cubeIndex] == 0) continue;
                
                // Interpolate vertices on edges
                glm::vec3 vertList[12];
                
                if (MarchingCubes::edgeTable[cubeIndex] & 1)
                    vertList[0] = vertexInterp(corners[0], corners[1], values[0], values[1]);
                if (MarchingCubes::edgeTable[cubeIndex] & 2)
                    vertList[1] = vertexInterp(corners[1], corners[2], values[1], values[2]);
                if (MarchingCubes::edgeTable[cubeIndex] & 4)
                    vertList[2] = vertexInterp(corners[2], corners[3], values[2], values[3]);
                if (MarchingCubes::edgeTable[cubeIndex] & 8)
                    vertList[3] = vertexInterp(corners[3], corners[0], values[3], values[0]);
                if (MarchingCubes::edgeTable[cubeIndex] & 16)
                    vertList[4] = vertexInterp(corners[4], corners[5], values[4], values[5]);
                if (MarchingCubes::edgeTable[cubeIndex] & 32)
                    vertList[5] = vertexInterp(corners[5], corners[6], values[5], values[6]);
                if (MarchingCubes::edgeTable[cubeIndex] & 64)
                    vertList[6] = vertexInterp(corners[6], corners[7], values[6], values[7]);
                if (MarchingCubes::edgeTable[cubeIndex] & 128)
                    vertList[7] = vertexInterp(corners[7], corners[4], values[7], values[4]);
                if (MarchingCubes::edgeTable[cubeIndex] & 256)
                    vertList[8] = vertexInterp(corners[0], corners[4], values[0], values[4]);
                if (MarchingCubes::edgeTable[cubeIndex] & 512)
                    vertList[9] = vertexInterp(corners[1], corners[5], values[1], values[5]);
                if (MarchingCubes::edgeTable[cubeIndex] & 1024)
                    vertList[10] = vertexInterp(corners[2], corners[6], values[2], values[6]);
                if (MarchingCubes::edgeTable[cubeIndex] & 2048)
                    vertList[11] = vertexInterp(corners[3], corners[7], values[3], values[7]);
                
                // Create triangles
                for (int n = 0; MarchingCubes::triTable[cubeIndex][n] != -1; n += 3) {
                    glm::vec3 v0 = vertList[MarchingCubes::triTable[cubeIndex][n]];
                    glm::vec3 v1 = vertList[MarchingCubes::triTable[cubeIndex][n + 1]];
                    glm::vec3 v2 = vertList[MarchingCubes::triTable[cubeIndex][n + 2]];
                    
                    // Compute face normal
                    glm::vec3 normal = glm::normalize(glm::cross(v1 - v0, v2 - v0));
                    
                    unsigned int i0 = addVertex(v0, normal);
                    unsigned int i1 = addVertex(v1, normal);
                    unsigned int i2 = addVertex(v2, normal);
                    
                    meshIndices.push_back(i0);
                    meshIndices.push_back(i1);
                    meshIndices.push_back(i2);
                }
            }
        }
    }
}

// Marching cubes lookup tables
namespace MarchingCubes {

const int edgeTable[256] = {
    0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0
};

const int triTable[256][16] = {
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
    {3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
    {3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
    {3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
    {9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
    {2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
    {8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
    {9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
    {4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
    {3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
    {1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
    {4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
    {4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
    {5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
    {2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
    {9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
    {0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
    {2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
    {10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
    {5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
    {5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
    {9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
    {1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
    {10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
    {8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
    {2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
    {7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
    {2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
    {11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
    {5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
    {11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
    {11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
    {9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
    {5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
    {2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
    {5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
    {6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
    {3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
    {6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
    {5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
    {10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
    {6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
    {8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
    {7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
    {3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
    {5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
    {0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
    {9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
    {8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
    {5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
    {0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
    {6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
    {10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
    {10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
    {8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
    {1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
    {0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
    {10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
    {3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
    {6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
    {9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
    {8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
    {3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
    {6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
    {0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
    {10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
    {10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
    {2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
    {7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
    {7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
    {2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
    {1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
    {11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
    {8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
    {0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
    {7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
    {10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
    {2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
    {6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
    {7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
    {2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
    {1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
    {10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
    {10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
    {0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
    {7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
    {6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
    {8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
    {9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
    {6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
    {4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
    {10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
    {8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
    {0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
    {1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
    {8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
    {10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
    {4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
    {10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
    {5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
    {11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
    {9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
    {6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
    {7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
    {3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
    {7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
    {3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
    {6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
    {9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
    {1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
    {4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
    {7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
    {6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
    {3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
    {0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
    {6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
    {0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
    {11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
    {6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
    {5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
    {9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
    {1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
    {1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
    {10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
    {0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
    {5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
    {10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
    {11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
    {9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
    {7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
    {2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
    {8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
    {9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
    {9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
    {1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
    {9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
    {9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
    {5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
    {0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
    {10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
    {2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
    {0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
    {0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
    {9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
    {5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
    {3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
    {5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
    {8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
    {0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
    {9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
    {1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
    {3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
    {4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
    {9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
    {11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
    {11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
    {2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
    {9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
    {3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
    {1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
    {4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
    {4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
    {3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
    {3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
    {0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
    {1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
};

} // namespace MarchingCubes

