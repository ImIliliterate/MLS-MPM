#pragma once

#include "MpmSim.h"
#include "SmokeSim.h"
#include <glm/glm.hpp>

/**
 * Two-way coupling between liquid (MPM) and smoke (Eulerian) simulations
 * 
 * Liquid → Smoke:
 *   - Liquid acts as moving obstacle for smoke
 *   - Splashes/impacts inject smoke density and temperature
 * 
 * Smoke → Liquid:
 *   - Wind drag affects surface particles
 *   - Hot air provides buoyancy
 */
class Coupling {
public:
    // References to simulations
    MpmSim* mpm = nullptr;
    SmokeSim* smoke = nullptr;
    
    // Coupling toggles
    bool enabled = true;
    bool liquidToSmoke = true;
    bool smokeToLiquid = true;
    
    // Liquid → Smoke parameters
    float splashDensityAmount = 5.0f;    // Density added per splash
    float splashTemperatureAmount = 2.0f; // Temperature added per splash  
    float splashRadius = 0.05f;           // Radius of splash injection
    float impactVelocityThreshold = 3.0f; // Minimum velocity change for splash
    
    // Smoke → Liquid parameters
    float dragCoefficient = 0.5f;         // Drag force coefficient
    float buoyancyCoefficient = 0.2f;     // Buoyancy force coefficient
    float surfaceLayerThickness = 0.1f;   // Only couple particles near surface
    
    // Particle radius for liquid mask
    float particleRadius = 0.01f;
    
    Coupling() = default;
    Coupling(MpmSim* mpm, SmokeSim* smoke);
    
    // Initialize coupling between simulations
    void init(MpmSim* mpm, SmokeSim* smoke);
    
    // Main coupling step - call before or after sim steps
    void apply(float dt);
    
    // Individual coupling operations
    void applyLiquidToSmoke(float dt);
    void applySmokeToLiquid(float dt);
    
    // Detect which particles are near the liquid surface
    std::vector<size_t> getSurfaceParticles() const;
    
private:
    // Track previous particle velocities for impact detection
    std::vector<glm::vec3> prevVelocities;
    
    // Helper to estimate if a particle is near the surface
    bool isNearSurface(const Particle& p) const;
};

