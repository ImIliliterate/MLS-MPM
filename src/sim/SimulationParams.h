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
    // ==================== Iteration 3: Resolution & PPC ====================
    // Target: 150k+ particles with 8+ PPC in filled regions
    int liquidGridRes = 64;            // Iteration 3: 64³ grid
    float particleSpacingFactor = 0.5f; // Iteration 3: spacing = dx * 0.5 (gives ~8 PPC)
    float dt = 0.003f;                 // Iteration 3: Reduced for stability with stiffer fluid
    float maxDt = 0.003f;
    float cflNumber = 0.4f;
    float worldSize = 1.0f;
    
    FluidPreset fluidPreset = FluidPreset::Default;  // Iteration 3: 64³ default
    QualityPreset qualityPreset = QualityPreset::Balanced;
    
    // ==================== Iteration 3: FLIP/PIC (98% FLIP) ====================
    float gravity = -9.8f;
    float flipRatio = 0.98f;           // Iteration 3: 98% FLIP, only 2% PIC
    float apicBlend = 0.5f;
    
    // ==================== Iteration 3: EOS Stiffness (2x increase) ====================
    bool enablePressure = true;
    float restDensity = 1000.0f;
    float stiffnessK = 200.0f;         // Iteration 3: DOUBLED from 100 to 200
    
    // ==================== Iteration 3: Viscosity = ZERO ====================
    float viscosityBlend = 0.0f;       // Iteration 3: ZERO
    float damping = 0.0f;              // Iteration 3: ZERO (no velocity decay)
    bool enableGridSmoothing = false;  // Iteration 3: OFF
    int gridSmoothingIterations = 0;
    
    // ==================== Iteration 3: Meshing ====================
    float isoThreshold = 0.4f;         // Iteration 3: Slightly lower for detail
    float smoothingRadius = 2.0f;      // Iteration 3: 2.0 * dx kernel radius
    int smoothingIterations = 2;
    float meshAnisotropy = 0.3f;
    bool fastMeshing = true;
    float particleRenderRadius = 0.5f; // Iteration 3: 0.5 * dx (down from 1.0)
    
    // ==================== Coupling (Iteration 3 adjustments) ====================
    float dragCoeff = 2.0f;           // Keep as-is for first high-res test
    float maxDragDelta = 8.0f;
    float buoyancyCoeff = 0.005f;     // Iteration 3: HALVED from 0.01
    float buoyancyMaxAccel = 0.5f;    // Iteration 3: Tighter clamp
    float ambientTemperature = 0.0f;
    float couplingBandWidth = 0.05f;
    int surfaceNeighborThreshold = 20;
    
    // ==================== Liquid → Smoke ====================
    float liquidVelocityBlend = 0.5f;
    float sprayDensityGain = 5.0f;
    float splashVelocityThreshold = 2.0f;
    
    // ==================== Smoke Grid ====================
    int smokeGridRes = 32;
    float smokeViscosity = 0.0001f;
    float smokeBuoyancy = 1.0f;
    float smokeDissipation = 0.995f;
    
    // ==================== Performance ====================
    int substeps = 3;
    bool enableCohesion = false;
    bool enableSmokeSimulation = false;
    bool enableCoupling = false;
    bool adaptiveDt = true;
    bool skipCouplingNeighborCount = true;
    bool skipDiagnosticsEveryFrame = true;
    int diagnosticsInterval = 10;
    
    // Legacy
    float bulkModulus = 1500.0f;
    float poissonRatio = 0.3f;
    
    // ==================== Diagnostics ====================
    bool enableDiagnostics = true;
    
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
