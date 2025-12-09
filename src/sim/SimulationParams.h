#pragma once

/**
 * Central configuration for all simulation parameters
 * 
 * Iteration 3: Resolution fix (Gemini feedback)
 *   - Particle spacing = dx/2 for 8 PPC in filled regions
 *   - Grid at 64³
 *   - FLIP = 98%
 *   - stiffnessK doubled
 *   - All damping/viscosity = 0
 */

// Resolution presets for quick PPC tuning
enum class FluidPreset {
    Coarse = 0,   // res=48, fewer cells
    Default = 1,  // res=64, balanced (Iteration 3 default)
    Fine = 2      // res=80, more detail
};

// Quality presets for performance vs fidelity tradeoff
enum class QualityPreset {
    Fast = 0,      // 2 substeps, 1 smooth iter
    Balanced = 1,  // 3 substeps, 2 smooth iters
    Quality = 2    // 4 substeps, 3 smooth iters
};

struct SimulationParams {
    // ==================== BALANCED FLUID ====================
    int liquidGridRes = 64;
    float particleSpacingFactor = 0.4f;
    float dt = 0.001f;                 // NUCLEAR: Larger timestep works with low stiffness
    float maxDt = 0.001f;
    float cflNumber = 0.3f;
    float worldSize = 1.0f;
    
    FluidPreset fluidPreset = FluidPreset::Coarse;
    QualityPreset qualityPreset = QualityPreset::Balanced;
    
    // ==================== Iteration 3: FLIP/PIC (98% FLIP) ====================
    float gravity = -9.8f;
    float flipRatio = 0.98f;           // Iteration 3: 98% FLIP, only 2% PIC
    float apicBlend = 1.0f;            // Full APIC - preserves angular momentum
    
    // ==================== BALANCED: Cohesive but stiff ====================
    bool enablePressure = true;
    float restDensity = 1000.0f;
    float stiffnessK = 2000.0f;        // Very low - guarantees CFL stability
    
    // ==================== Iteration 3: Viscosity = ZERO ====================
    float viscosityBlend = 0.0f;       // Iteration 3: ZERO
    float damping = 0.0f;              // Iteration 3: ZERO (no velocity decay)
    bool enableGridSmoothing = false;  // Iteration 3: OFF
    int gridSmoothingIterations = 0;
    
    // ==================== Iteration 5: Meshing (sharper with more particles) ====================
    // With higher particle density, use tighter kernel for sharper surface
    float isoThreshold = 0.6f;         // Higher = tighter/sharper surface
    float smoothingRadius = 1.5f;      // Tighter kernel = 1.5*dx (was 2.0)
    int smoothingIterations = 2;       // 2 passes
    float meshAnisotropy = 0.2f;       // Less anisotropy for finer detail
    bool fastMeshing = true;
    float particleRenderRadius = 0.4f; // Smaller render radius with more particles
    
    // ==================== Coupling (Iteration 4: Reduced buoyancy) ====================
    float dragCoeff = 0.3f;           // Moderate - visible interaction but water stays heavy
    float maxDragDelta = 8.0f;
    float buoyancyCoeff = 0.001f;     // Iteration 4: 5x reduction (was 0.005)
    float buoyancyMaxAccel = 0.3f;    // Iteration 4: Tighter clamp
    float ambientTemperature = 0.0f;
    float couplingBandWidth = 0.05f;
    int surfaceNeighborThreshold = 20;
    
    // ==================== Liquid → Smoke ====================
    float liquidVelocityBlend = 0.5f;
    float sprayDensityGain = 5.0f;
    float splashVelocityThreshold = 2.0f;
    
    // ==================== Smoke Grid ====================
    int smokeGridRes = 16;             // Cheaper smoke grid default
    float smokeViscosity = 0.0001f;
    float smokeBuoyancy = 1.0f;
    float smokeDissipation = 0.995f;
    
    // ==================== Performance ====================
    int substeps = 8;                  // Balanced for moderate stiffness
    bool enableCohesion = false;
    bool enableSmokeSimulation = true;   // ON - GPU smoke solver
    bool enableCoupling = true;          // ON - GPU coupling
    bool adaptiveDt = true;            // ON - CFL-adaptive prevents explosions
    bool skipCouplingNeighborCount = true;
    bool skipDiagnosticsEveryFrame = true;
    int diagnosticsInterval = 10;
    
    // Stiffness used by particles - HIGH for incompressible water
    // CFL-adaptive stepping prevents explosions automatically
    float bulkModulus = 50000.0f;  // High stiffness = incompressible water
    float poissonRatio = 0.3f;
    
    // ==================== Diagnostics ====================
    bool enableDiagnostics = false;    // Skip extra CPU work by default
    
    mutable float maxLiquidSpeed = 0.0f;
    mutable float maxSmokeSpeed = 0.0f;
    mutable int numLiquidParticles = 0;
    mutable float avgParticlesPerCell = 0.0f;
    mutable float activePPC = 0.0f;
    mutable int activeCellCount = 0;
    mutable float actualDt = 0.0f;
    mutable int surfaceParticleCount = 0;
    mutable float maxDensity = 0.0f;
    mutable float maxPressure = 0.0f;
    mutable int frameCount = 0;
    mutable float dx = 0.0f;           // Iteration 3: Track dx for diagnostics
};

extern SimulationParams g_params;
