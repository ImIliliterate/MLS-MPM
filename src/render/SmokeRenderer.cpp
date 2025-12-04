#include "SmokeRenderer.h"
#include <glm/gtc/type_ptr.hpp>

SmokeRenderer::SmokeRenderer() = default;

SmokeRenderer::~SmokeRenderer() {
    cleanup();
}

bool SmokeRenderer::init() {
    if (!initQuad()) return false;
    if (!initShader()) return false;
    return true;
}

void SmokeRenderer::cleanup() {
    if (quadVAO) { glDeleteVertexArrays(1, &quadVAO); quadVAO = 0; }
    if (quadVBO) { glDeleteBuffers(1, &quadVBO); quadVBO = 0; }
    if (densityTexture) { glDeleteTextures(1, &densityTexture); densityTexture = 0; }
}

bool SmokeRenderer::initQuad() {
    // Fullscreen quad vertices (position + UV)
    float quadVertices[] = {
        -1.0f,  1.0f, 0.0f, 1.0f,
        -1.0f, -1.0f, 0.0f, 0.0f,
         1.0f, -1.0f, 1.0f, 0.0f,
         
        -1.0f,  1.0f, 0.0f, 1.0f,
         1.0f, -1.0f, 1.0f, 0.0f,
         1.0f,  1.0f, 1.0f, 1.0f
    };
    
    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);
    
    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
    
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    glBindVertexArray(0);
    
    return true;
}

bool SmokeRenderer::initShader() {
    const char* vertSrc = R"(
        #version 330 core
        layout(location = 0) in vec2 aPos;
        layout(location = 1) in vec2 aUV;
        
        out vec2 vUV;
        
        void main() {
            vUV = aUV;
            gl_Position = vec4(aPos, 0.0, 1.0);
        }
    )";
    
    const char* fragSrc = R"(
        #version 330 core
        in vec2 vUV;
        out vec4 FragColor;
        
        uniform sampler3D uDensity;
        uniform mat4 uInvView;
        uniform mat4 uInvProj;
        uniform vec3 uCameraPos;
        uniform vec3 uLightDir;
        uniform vec3 uVolumeMin;
        uniform vec3 uVolumeMax;
        uniform vec3 uSmokeColor;
        uniform vec3 uLightColor;
        uniform int uRaySteps;
        uniform float uDensityScale;
        uniform float uAbsorption;
        uniform float uShadowSteps;
        uniform float uShadowDensity;
        
        // Ray-box intersection
        vec2 intersectBox(vec3 ro, vec3 rd, vec3 boxMin, vec3 boxMax) {
            vec3 t1 = (boxMin - ro) / rd;
            vec3 t2 = (boxMax - ro) / rd;
            vec3 tmin = min(t1, t2);
            vec3 tmax = max(t1, t2);
            float tNear = max(max(tmin.x, tmin.y), tmin.z);
            float tFar = min(min(tmax.x, tmax.y), tmax.z);
            return vec2(tNear, tFar);
        }
        
        // Sample density at world position
        float sampleDensity(vec3 worldPos) {
            vec3 uvw = (worldPos - uVolumeMin) / (uVolumeMax - uVolumeMin);
            if (any(lessThan(uvw, vec3(0.0))) || any(greaterThan(uvw, vec3(1.0)))) {
                return 0.0;
            }
            return texture(uDensity, uvw).r;
        }
        
        // Simple shadow ray
        float shadowRay(vec3 pos, vec3 lightDir) {
            float shadow = 0.0;
            float stepSize = length(uVolumeMax - uVolumeMin) / uShadowSteps;
            
            for (int i = 0; i < int(uShadowSteps); i++) {
                pos += lightDir * stepSize;
                shadow += sampleDensity(pos) * uShadowDensity;
            }
            
            return exp(-shadow);
        }
        
        void main() {
            // Reconstruct ray from screen position
            vec4 clipPos = vec4(vUV * 2.0 - 1.0, -1.0, 1.0);
            vec4 viewPos = uInvProj * clipPos;
            viewPos.xyz /= viewPos.w;
            vec3 rayDir = normalize((uInvView * vec4(viewPos.xyz, 0.0)).xyz);
            vec3 rayOrigin = uCameraPos;
            
            // Intersect with volume bounds
            vec2 t = intersectBox(rayOrigin, rayDir, uVolumeMin, uVolumeMax);
            
            if (t.x > t.y || t.y < 0.0) {
                FragColor = vec4(0.0);
                return;
            }
            
            t.x = max(t.x, 0.0);
            
            // Ray march through volume
            float stepSize = (t.y - t.x) / float(uRaySteps);
            vec3 pos = rayOrigin + rayDir * t.x;
            
            vec3 accumulatedColor = vec3(0.0);
            float accumulatedAlpha = 0.0;
            
            vec3 lightDir = normalize(-uLightDir);
            
            for (int i = 0; i < uRaySteps && accumulatedAlpha < 0.99; i++) {
                float density = sampleDensity(pos) * uDensityScale;
                
                if (density > 0.001) {
                    // Beer-Lambert absorption
                    float alpha = 1.0 - exp(-density * uAbsorption * stepSize);
                    
                    // Lighting with shadow
                    float shadow = shadowRay(pos, lightDir);
                    vec3 color = uSmokeColor * uLightColor * (0.2 + 0.8 * shadow);
                    
                    // Accumulate
                    accumulatedColor += (1.0 - accumulatedAlpha) * alpha * color;
                    accumulatedAlpha += (1.0 - accumulatedAlpha) * alpha;
                }
                
                pos += rayDir * stepSize;
            }
            
            FragColor = vec4(accumulatedColor, accumulatedAlpha);
        }
    )";
    
    return shader.loadFromSource(vertSrc, fragSrc);
}

void SmokeRenderer::updateDensity(const SmokeSim& sim) {
    // Create or resize texture if needed
    if (texWidth != sim.Nx || texHeight != sim.Ny || texDepth != sim.Nz) {
        if (densityTexture) {
            glDeleteTextures(1, &densityTexture);
        }
        
        glGenTextures(1, &densityTexture);
        glBindTexture(GL_TEXTURE_3D, densityTexture);
        
        texWidth = sim.Nx;
        texHeight = sim.Ny;
        texDepth = sim.Nz;
        
        glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, texWidth, texHeight, texDepth, 
                     0, GL_RED, GL_FLOAT, nullptr);
        
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    }
    
    // Upload density data
    glBindTexture(GL_TEXTURE_3D, densityTexture);
    glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, texWidth, texHeight, texDepth,
                    GL_RED, GL_FLOAT, sim.getDensityData().data());
}

void SmokeRenderer::render(const glm::mat4& view, const glm::mat4& projection,
                           const glm::vec3& cameraPos, const glm::vec3& lightDir,
                           const glm::vec3& volumeMin, const glm::vec3& volumeMax) {
    if (!densityTexture) return;
    
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_DEPTH_TEST);
    
    shader.use();
    
    glm::mat4 invView = glm::inverse(view);
    glm::mat4 invProj = glm::inverse(projection);
    
    shader.setMat4("uInvView", invView);
    shader.setMat4("uInvProj", invProj);
    shader.setVec3("uCameraPos", cameraPos);
    shader.setVec3("uLightDir", lightDir);
    shader.setVec3("uVolumeMin", volumeMin);
    shader.setVec3("uVolumeMax", volumeMax);
    shader.setVec3("uSmokeColor", smokeColor);
    shader.setVec3("uLightColor", lightColor);
    shader.setInt("uRaySteps", raySteps);
    shader.setFloat("uDensityScale", densityScale);
    shader.setFloat("uAbsorption", absorptionCoeff);
    shader.setFloat("uShadowSteps", shadowSteps);
    shader.setFloat("uShadowDensity", shadowDensity);
    
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, densityTexture);
    shader.setInt("uDensity", 0);
    
    glBindVertexArray(quadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
    
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
}

