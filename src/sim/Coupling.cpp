#include "Coupling.h"
#include "SimulationParams.h"
#include <algorithm>
#include <cmath>
#include <unordered_map>

Coupling::Coupling(MpmSim* mpm, SmokeSim* smoke)
    : mpm(mpm), smoke(smoke) {
}

void Coupling::init(MpmSim* mpmSim, SmokeSim* smokeSim) {
    mpm = mpmSim;
    smoke = smokeSim;
    prevVelocities.clear();
    neighborCounts.clear();
}

void Coupling::apply(float dt) {
    if (!enabled || !mpm || !smoke) return;
    
    // Build neighbor counts for surface detection (Phase 1.2)
    // Skip if using fast mode (just use smoke grid for surface detection)
    if (!g_params.skipCouplingNeighborCount) {
        buildNeighborCounts();
    }
    
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

void Coupling::buildNeighborCounts() {
    // Build spatial hash for efficient neighbor counting
    neighborCounts.resize(mpm->particles.size(), 0);
    
    float h = mpm->dx * 2.0f;  // Neighbor search radius
    float h2 = h * h;
    
    // Spatial hash
    std::unordered_map<int64_t, std::vector<int>> spatialHash;
    auto hashPos = [&](const glm::vec3& pos) -> int64_t {
        int cx = static_cast<int>(std::floor(pos.x / h));
        int cy = static_cast<int>(std::floor(pos.y / h));
        int cz = static_cast<int>(std::floor(pos.z / h));
        return (static_cast<int64_t>(cx) * 73856093) ^
               (static_cast<int64_t>(cy) * 19349663) ^
               (static_cast<int64_t>(cz) * 83492791);
    };
    
    for (int i = 0; i < static_cast<int>(mpm->particles.size()); i++) {
        spatialHash[hashPos(mpm->particles[i].x)].push_back(i);
    }
    
    // Count neighbors for each particle
    for (int i = 0; i < static_cast<int>(mpm->particles.size()); i++) {
        const auto& pi = mpm->particles[i];
        int cx = static_cast<int>(std::floor(pi.x.x / h));
        int cy = static_cast<int>(std::floor(pi.x.y / h));
        int cz = static_cast<int>(std::floor(pi.x.z / h));
        
        int count = 0;
        
        for (int di = -1; di <= 1; di++) {
            for (int dj = -1; dj <= 1; dj++) {
                for (int dk = -1; dk <= 1; dk++) {
                    int64_t hash = (static_cast<int64_t>(cx + di) * 73856093) ^
                                   (static_cast<int64_t>(cy + dj) * 19349663) ^
                                   (static_cast<int64_t>(cz + dk) * 83492791);
                    
                    auto it = spatialHash.find(hash);
                    if (it == spatialHash.end()) continue;
                    
                    for (int j : it->second) {
                        if (i == j) continue;
                        glm::vec3 diff = pi.x - mpm->particles[j].x;
                        float d2 = glm::dot(diff, diff);
                        if (d2 < h2) {
                            count++;
                        }
                    }
                }
            }
        }
        
        neighborCounts[i] = count;
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
            
            float speedChange = glm::length(p.v - prevVel);
            bool nearGround = p.x.y < mpm->worldMin.y + surfaceLayerThickness;
            bool velocityReversed = prevVel.y < -1.0f && p.v.y >= 0.0f;
            
            float threshold = g_params.splashVelocityThreshold;
            if (speedChange > threshold || (nearGround && velocityReversed)) {
                float strength = glm::min(speedChange / 10.0f, 1.0f);
                smoke->addDensity(p.x, g_params.sprayDensityGain * strength, splashRadius);
                smoke->addTemperature(p.x, splashTemperatureAmount * strength, splashRadius);
                
                glm::vec3 impactNormal(0, 1, 0);
                smoke->addVelocity(p.x, impactNormal * speedChange * 0.1f, splashRadius);
            }
        }
    }
}

void Coupling::applySmokeToLiquid(float dt) {
    /**
     * Phase 1 Coupling Improvements:
     * 
     * 1.1 - IMPLICIT DRAG (numerically stable even when C*dt > 1)
     *       v_new = (v_old + k * v_smoke) / (1 + k)  where k = C * dt
     * 
     * 1.2 - SURFACE BAND: Only apply to particles near surface (low neighbor count)
     * 
     * 1.3 - CLAMPED BUOYANCY: Prevent entire bulk from levitating
     */
    
    int surfaceCount = 0;
    
    for (size_t idx = 0; idx < mpm->particles.size(); idx++) {
        // Phase 1.2: Check if particle is near surface
        if (!isNearSurface(idx)) {
            continue;  // Skip interior particles
        }
        surfaceCount++;
        
        Particle& p = mpm->particles[idx];
        
        // Sample smoke at particle position
        glm::vec3 v_smoke = smoke->sampleVelocity(p.x);
        float smokeTemp = smoke->sampleTemperature(p.x);
        
        // ==================== PHASE 1.1: IMPLICIT DRAG ====================
        // Mathematically: v_new = (v_old + k * v_smoke) / (1 + k)
        // This is unconditionally stable (no oscillation even with large C*dt)
        float C = g_params.dragCoeff;
        float k = C * dt;
        
        glm::vec3 v_new = (p.v + k * v_smoke) / (1.0f + k);
        
        // Optional: clamp max velocity change from drag
        glm::vec3 dv = v_new - p.v;
        float dvLen = glm::length(dv);
        float maxDv = g_params.maxDragDelta;
        if (dvLen > maxDv) {
            dv *= maxDv / (dvLen + 1e-6f);
            v_new = p.v + dv;
        }
        
        p.v = v_new;
        
        // ==================== PHASE 1.3: CLAMPED BUOYANCY ====================
        // Only apply gentle, clamped buoyancy
        float dT = glm::max(0.0f, smokeTemp - g_params.ambientTemperature);
        glm::vec3 a_buoy = g_params.buoyancyCoeff * dT * glm::vec3(0.0f, 1.0f, 0.0f);
        
        // Clamp buoyancy acceleration magnitude
        float aLen = glm::length(a_buoy);
        if (aLen > g_params.buoyancyMaxAccel) {
            a_buoy *= g_params.buoyancyMaxAccel / (aLen + 1e-6f);
        }
        
        p.v += a_buoy * dt;
    }
    
    // Update diagnostic
    g_params.surfaceParticleCount = surfaceCount;
}

std::vector<size_t> Coupling::getSurfaceParticles() const {
    std::vector<size_t> surface;
    if (!mpm) return surface;
    
    surface.reserve(mpm->particles.size() / 4);
    
    for (size_t i = 0; i < mpm->particles.size(); i++) {
        if (isNearSurface(i)) {
            surface.push_back(i);
        }
    }
    
    return surface;
}

bool Coupling::isNearSurface(size_t particleIdx) const {
    if (!mpm || particleIdx >= mpm->particles.size()) return false;
    
    // Phase 1.2: Use neighbor count for surface detection
    // Particles with few neighbors are near the surface
    if (particleIdx < neighborCounts.size()) {
        int neighbors = neighborCounts[particleIdx];
        if (neighbors < g_params.surfaceNeighborThreshold) {
            return true;
        }
    }
    
    // Also check smoke grid for air cells nearby
    if (smoke) {
        const Particle& p = mpm->particles[particleIdx];
        glm::vec3 gridPos = smoke->worldToGrid(p.x);
        int i = static_cast<int>(gridPos.x);
        int j = static_cast<int>(gridPos.y);
        int k = static_cast<int>(gridPos.z);
        
        // Check immediate neighborhood for air cells
        for (int di = -1; di <= 1; di++) {
            for (int dj = -1; dj <= 1; dj++) {
                for (int dk = -1; dk <= 1; dk++) {
                    int ni = i + di;
                    int nj = j + dj;
                    int nk = k + dk;
                    
                    if (ni < 0 || ni >= smoke->Nx ||
                        nj < 0 || nj >= smoke->Ny ||
                        nk < 0 || nk >= smoke->Nz) {
                        return true;  // Near domain boundary
                    }
                    
                    if (!smoke->liquidMask(ni, nj, nk)) {
                        return true;  // Neighboring air cell
                    }
                }
            }
        }
    }
    
    return false;
}

// Keep old signature for compatibility
bool Coupling::isNearSurface(const Particle& p) const {
    // Find particle index (slow, prefer using index version)
    for (size_t i = 0; i < mpm->particles.size(); i++) {
        if (&mpm->particles[i] == &p) {
            return isNearSurface(i);
        }
    }
    return false;
}
