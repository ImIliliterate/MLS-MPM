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
    // ==================== FLUID PARAMETERS ====================
    int liquidGridRes = 64;
    float particleSpacingFactor = 0.6f; // dx*0.6 = target Active PPC 3-5
    float dt = 0.001f;
    float maxDt = 0.002f;
    float cflNumber = 0.4f;
    float worldSize = 1.0f;
    
    FluidPreset fluidPreset = FluidPreset::Default;  // 64^3 grid
    QualityPreset qualityPreset = QualityPreset::Balanced;
    
    // ==================== FLIP/PIC ====================
    float gravity = -9.8f;
    float flipRatio = 0.95f;           // 95% FLIP - less bouncy, more damped
    float apicBlend = 1.0f;            // Full APIC - preserves angular momentum
    
    // ==================== INCOMPRESSIBILITY ====================
    bool enablePressure = true;
    float restDensity = 1000.0f;
    float stiffnessK = 500.0f;         // Lower = flows more freely
    
    // ==================== VISCOSITY ====================
    float viscosityBlend = 0.0f;       // No viscosity - water is inviscid
    float damping = 0.0f;              // No global damping
    bool enableGridSmoothing = false;  // Off for water
    int gridSmoothingIterations = 0;
    
    // ==================== Iteration 5: Meshing (sharper with more particles) ====================
    // Meshing: Zhu & Bridson style surface reconstruction
    float isoThreshold = 0.5f;         // Middle value for smooth surface
    float smoothingRadius = 2.0f;      // 2*dx (now hardcoded in buildScalarField)
    int smoothingIterations = 2;       // 2 passes as specified
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
    float liquidVelocityBlend = 1.0f;
    float sprayDensityGain = 100000.0f;     // INSANE - 100x
    float splashVelocityThreshold = 0.01f;  // Hyper sensitive
    
    // ==================== Smoke Grid ====================
    int smokeGridRes = 48;             // Good detail
    float smokeViscosity = 0.0f;       // ZERO = maximum turbulence
    float smokeBuoyancy = 50.0f;       // EXTREME buoyancy = rockets up fast
    float smokeDissipation = 0.995f;   // Slower dissipation so we see it
    
    // ==================== Performance ====================
    int substeps = 3;                  // Reduced for perf - still stable with CFL
    bool enableCohesion = false;       // OFF - expensive
    bool enableSmokeSimulation = true;   // ON - GPU smoke solver
    bool enableCoupling = true;          // ON - GPU coupling
    bool adaptiveDt = true;            // ON - CFL-adaptive prevents explosions
    bool skipCouplingNeighborCount = true;
    bool skipDiagnosticsEveryFrame = true;
    int diagnosticsInterval = 10;
    
    // Cohesion / Surface Tension parameters
    float cohesionStrength = 500.0f;   // How strongly particles attract at surface
    float surfaceTensionThreshold = 0.7f; // Density threshold for surface detection
    
    // Stiffness used by particles - lower = more splashy, higher = more rigid
    float bulkModulus = 5000.0f;   // Low - allows spreading and splashing
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
