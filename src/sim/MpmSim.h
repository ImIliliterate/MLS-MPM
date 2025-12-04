#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <functional>

/**
 * Grid node for MPM simulation
 * Stores mass, velocity, and momentum for each grid cell
 */
struct GridNode {
    float mass = 0.0f;
    glm::vec3 vel{0.0f};
    glm::vec3 velNew{0.0f};  // For FLIP/PIC blending
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
    float flipRatio = 0.0f;  // Blend between FLIP (1.0) and PIC (0.0) - using pure PIC for stability
    
    // Material presets
    static constexpr float WATER_E = 1000.0f;
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

