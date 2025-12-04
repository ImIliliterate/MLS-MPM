#include "Coupling.h"
#include <algorithm>
#include <cmath>

Coupling::Coupling(MpmSim* mpm, SmokeSim* smoke)
    : mpm(mpm), smoke(smoke) {
}

void Coupling::init(MpmSim* mpmSim, SmokeSim* smokeSim) {
    mpm = mpmSim;
    smoke = smokeSim;
    prevVelocities.clear();
}

void Coupling::apply(float dt) {
    if (!enabled || !mpm || !smoke) return;
    
    if (liquidToSmoke) {
        applyLiquidToSmoke(dt);
    }
    
    if (smokeToLiquid) {
        applySmokeToLiquid(dt);
    }
    
    // Update previous velocities for next frame
    prevVelocities.resize(mpm->particles.size());
    for (size_t i = 0; i < mpm->particles.size(); i++) {
        prevVelocities[i] = mpm->particles[i].v;
    }
}

void Coupling::applyLiquidToSmoke(float dt) {
    // 1. Update liquid occupancy mask for smoke
    std::vector<glm::vec3> positions;
    positions.reserve(mpm->particles.size());
    for (const auto& p : mpm->particles) {
        positions.push_back(p.x);
    }
    smoke->setLiquidMask(positions, particleRadius);
    
    // 2. Inject smoke at impact locations
    if (prevVelocities.size() == mpm->particles.size()) {
        for (size_t i = 0; i < mpm->particles.size(); i++) {
            const Particle& p = mpm->particles[i];
            glm::vec3 prevVel = prevVelocities[i];
            
            // Check for high-speed impacts
            float speedChange = glm::length(p.v - prevVel);
            
            // Detect impact with ground or objects
            bool nearGround = p.x.y < mpm->worldMin.y + surfaceLayerThickness;
            bool velocityReversed = prevVel.y < -1.0f && p.v.y >= 0.0f;
            
            if (speedChange > impactVelocityThreshold || (nearGround && velocityReversed)) {
                // Inject splash
                float strength = glm::min(speedChange / 10.0f, 1.0f);
                smoke->addDensity(p.x, splashDensityAmount * strength, splashRadius);
                smoke->addTemperature(p.x, splashTemperatureAmount * strength, splashRadius);
                
                // Add some velocity to make smoke move away from impact
                glm::vec3 impactNormal(0, 1, 0);  // Assume ground impact
                smoke->addVelocity(p.x, impactNormal * speedChange * 0.1f, splashRadius);
            }
        }
    }
    
    // 3. Set smoke velocity at liquid interface
    // For cells marked as liquid, we could enforce the liquid velocity
    // This is handled in the smoke pressure solve by treating liquid as solid
}

void Coupling::applySmokeToLiquid(float dt) {
    // Apply drag and buoyancy from smoke to surface particles
    
    std::vector<size_t> surfaceParticles = getSurfaceParticles();
    
    for (size_t idx : surfaceParticles) {
        Particle& p = mpm->particles[idx];
        
        // Sample smoke at particle position
        glm::vec3 smokeVel = smoke->sampleVelocity(p.x);
        float smokeDensity = smoke->sampleDensity(p.x);
        float smokeTemp = smoke->sampleTemperature(p.x);
        
        // Apply drag force
        // F_drag = -c_drag * (v_particle - v_smoke)
        glm::vec3 relVel = p.v - smokeVel;
        glm::vec3 dragForce = -dragCoefficient * smokeDensity * relVel;
        
        // Apply buoyancy from hot air
        float dT = glm::max(0.0f, smokeTemp - smoke->ambientTemperature);
        glm::vec3 buoyancyForce = buoyancyCoefficient * dT * glm::vec3(0, 1, 0);
        
        // Update particle velocity
        glm::vec3 totalForce = dragForce + buoyancyForce;
        p.v += (dt / p.mass) * totalForce;
    }
}

std::vector<size_t> Coupling::getSurfaceParticles() const {
    std::vector<size_t> surface;
    if (!mpm) return surface;
    
    // Simple heuristic: particles near the top of the liquid
    // A better approach would use neighbor counting
    
    // Find approximate liquid surface height in local regions
    // For simplicity, just use particles in the upper portion
    
    surface.reserve(mpm->particles.size() / 4);
    
    for (size_t i = 0; i < mpm->particles.size(); i++) {
        const Particle& p = mpm->particles[i];
        
        if (isNearSurface(p)) {
            surface.push_back(i);
        }
    }
    
    return surface;
}

bool Coupling::isNearSurface(const Particle& p) const {
    if (!smoke) return false;
    
    // Check if any neighboring smoke cell is not liquid
    glm::vec3 gridPos = smoke->worldToGrid(p.x);
    int i = static_cast<int>(gridPos.x);
    int j = static_cast<int>(gridPos.y);
    int k = static_cast<int>(gridPos.z);
    
    // Check 3x3x3 neighborhood
    for (int di = -1; di <= 1; di++) {
        for (int dj = -1; dj <= 1; dj++) {
            for (int dk = -1; dk <= 1; dk++) {
                int ni = i + di;
                int nj = j + dj;
                int nk = k + dk;
                
                if (ni < 0 || ni >= smoke->Nx ||
                    nj < 0 || nj >= smoke->Ny ||
                    nk < 0 || nk >= smoke->Nz) {
                    // Near domain boundary = near surface
                    return true;
                }
                
                if (!smoke->liquidMask(ni, nj, nk)) {
                    // Neighboring air cell = near surface
                    return true;
                }
            }
        }
    }
    
    return false;
}

