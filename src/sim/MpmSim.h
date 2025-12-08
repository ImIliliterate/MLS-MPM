#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <functional>

/**
 * Grid node for MPM simulation
 * Stores mass, velocity, momentum, and fluid pressure for each grid cell
 * 
 * Round 3: Added density and pressure for continuum fluid behavior
 */
struct GridNode {
    float mass = 0.0f;
    glm::vec3 vel{0.0f};
    glm::vec3 velNew{0.0f};  // For FLIP/PIC blending
    
    // Phase 1: Fluid pressure fields
    float density = 0.0f;    // Computed from mass / cell volume
    float pressure = 0.0f;   // Computed from equation of state
};

/**
 * Particle for MPM simulation
 * Stores position, velocity, deformation gradient, and material properties
 */
struct Particle {
    glm::vec3 x{0.0f};      // Position
    glm::vec3 v{0.0f};      // Velocity
    float mass = 1.0f;
    float volume0 = 1.0f;   // Initial volume
    glm::mat3 F{1.0f};      // Deformation gradient
    glm::mat3 C{0.0f};      // Affine momentum (APIC)
    
    // Material properties
    float E = 1000.0f;      // Young's modulus
    float nu = 0.2f;        // Poisson's ratio
    
    // For visualization
    glm::vec3 color{0.2f, 0.5f, 0.9f};
};

/**
 * MLS-MPM (Moving Least Squares Material Point Method) simulator
 * Based on Hu et al. "A Moving Least Squares Material Point Method with Displacement Discontinuity"
 */
class MpmSim {
public:
    // World bounds and grid resolution
    glm::vec3 worldMin{0.0f};
    glm::vec3 worldMax{1.0f};
    int Nx = 64, Ny = 64, Nz = 64;
    
    // Grid cell size (computed from bounds and resolution)
    float dx;
    
    // Simulation parameters
    float gravity = -9.8f;
    float flipRatio = 0.95f;  // Higher = more fluid, splashy motion
    bool enableCohesion = true;  // Surface tension (expensive, disable for performance)
    
    // Material presets - softer for cohesive fluid behavior
    static constexpr float WATER_E = 1500.0f;
    static constexpr float WATER_NU = 0.3f;
    
    // Storage
    std::vector<GridNode> grid;
    std::vector<Particle> particles;
    
    // SDF function for collisions (returns signed distance)
    std::function<float(const glm::vec3&)> sdfFunc;
    std::function<glm::vec3(const glm::vec3&)> sdfGradFunc;
    
    MpmSim();
    
    // Initialize simulation
    void init(int nx, int ny, int nz);
    void reset();
    
    // Add particles in various shapes
    void addBox(const glm::vec3& min, const glm::vec3& max, const glm::vec3& velocity, 
                float particleSpacing, float density = 1000.0f);
    void addSphere(const glm::vec3& center, float radius, const glm::vec3& velocity,
                   float particleSpacing, float density = 1000.0f);
    
    // Main simulation step
    void step(float dt);
    
    // Compute adaptive timestep based on CFL condition
    float computeAdaptiveDt() const;
    
    // Update diagnostic values (max speed, particle counts, etc.)
    void computeDiagnostics();
    
    // Phase 2.3: Jacobi smoothing on grid velocities to reduce clumping
    void smoothGridVelocities();
    
    // Round 3 Phase 1: Pressure-based fluid forces
    void computeGridDensity();      // 1.2: Compute density from mass
    void computeGridPressure();     // 1.3: Equation of state
    void applyPressureForces(float dt);  // 1.4: Pressure gradient forces
    
    // Access particles for rendering
    const std::vector<Particle>& getParticles() const { return particles; }
    const std::vector<GridNode>& getGrid() const { return grid; }
    
    // Get world-space grid positions
    glm::vec3 gridToWorld(int i, int j, int k) const;
    glm::ivec3 worldToGrid(const glm::vec3& pos) const;
    
    // Detect high-speed impacts for coupling (positions of splashes)
    std::vector<glm::vec3> getImpactPositions() const;
    
    // For coupling: apply external force to particles near a position
    void applyExternalForce(const glm::vec3& pos, const glm::vec3& force, float radius);
    
private:
    // Core MLS-MPM steps
    void clearGrid();
    void particleToGrid(float dt);
    void applyGridForcesAndBCs(float dt);
    void gridToParticle(float dt);
    
    // Surface tension / cohesion
    void applyCohesionForces(float dt);
    
    // Grid indexing
    int gridIndex(int i, int j, int k) const;
    bool isValidCell(int i, int j, int k) const;
    
    // B-spline weight functions
    float N(float x) const;  // Quadratic B-spline
    float dN(float x) const; // Derivative of quadratic B-spline
    
    // Compute stress from deformation gradient (Neo-Hookean)
    glm::mat3 computeStress(const Particle& p) const;
    
    // Track impacts for coupling
    std::vector<glm::vec3> impactPositions;
};

