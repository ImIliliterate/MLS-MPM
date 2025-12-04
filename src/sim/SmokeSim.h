#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <functional>

/**
 * 3D Array helper for smoke grids
 */
template<typename T>
class Array3D {
public:
    int Nx = 0, Ny = 0, Nz = 0;
    std::vector<T> data;
    
    Array3D() = default;
    
    void resize(int nx, int ny, int nz, T defaultVal = T()) {
        Nx = nx;
        Ny = ny;
        Nz = nz;
        data.assign(nx * ny * nz, defaultVal);
    }
    
    T& operator()(int i, int j, int k) {
        return data[i + Nx * (j + Ny * k)];
    }
    
    const T& operator()(int i, int j, int k) const {
        return data[i + Nx * (j + Ny * k)];
    }
    
    T sample(float fx, float fy, float fz) const {
        // Trilinear interpolation
        int i0 = static_cast<int>(fx);
        int j0 = static_cast<int>(fy);
        int k0 = static_cast<int>(fz);
        
        float tx = fx - i0;
        float ty = fy - j0;
        float tz = fz - k0;
        
        // Clamp to valid range
        i0 = glm::clamp(i0, 0, Nx - 1);
        j0 = glm::clamp(j0, 0, Ny - 1);
        k0 = glm::clamp(k0, 0, Nz - 1);
        int i1 = glm::min(i0 + 1, Nx - 1);
        int j1 = glm::min(j0 + 1, Ny - 1);
        int k1 = glm::min(k0 + 1, Nz - 1);
        
        // Trilinear interpolation
        T c000 = (*this)(i0, j0, k0);
        T c100 = (*this)(i1, j0, k0);
        T c010 = (*this)(i0, j1, k0);
        T c110 = (*this)(i1, j1, k0);
        T c001 = (*this)(i0, j0, k1);
        T c101 = (*this)(i1, j0, k1);
        T c011 = (*this)(i0, j1, k1);
        T c111 = (*this)(i1, j1, k1);
        
        T c00 = c000 * (1 - tx) + c100 * tx;
        T c10 = c010 * (1 - tx) + c110 * tx;
        T c01 = c001 * (1 - tx) + c101 * tx;
        T c11 = c011 * (1 - tx) + c111 * tx;
        
        T c0 = c00 * (1 - ty) + c10 * ty;
        T c1 = c01 * (1 - ty) + c11 * ty;
        
        return c0 * (1 - tz) + c1 * tz;
    }
    
    void fill(T value) {
        std::fill(data.begin(), data.end(), value);
    }
};

/**
 * Stable Fluids smoke simulation
 * Based on Stam's "Stable Fluids" paper
 * Uses staggered MAC grid for velocity
 */
class SmokeSim {
public:
    // Grid dimensions
    int Nx = 32, Ny = 32, Nz = 32;
    
    // World bounds (same as MPM for easy coupling)
    glm::vec3 worldMin{0.0f};
    glm::vec3 worldMax{1.0f};
    float dx;
    
    // Staggered velocity grids (MAC grid)
    Array3D<float> u;  // x-velocity at (i+0.5, j, k)
    Array3D<float> v;  // y-velocity at (i, j+0.5, k)  
    Array3D<float> w;  // z-velocity at (i, j, k+0.5)
    
    // Scalar fields
    Array3D<float> density;
    Array3D<float> temperature;
    
    // Temporary storage for advection
    Array3D<float> uTemp, vTemp, wTemp;
    Array3D<float> densityTemp, temperatureTemp;
    
    // Pressure solver
    Array3D<float> pressure;
    Array3D<float> divergence;
    
    // Liquid occupancy mask (for coupling) - using char instead of bool to avoid vector<bool> issues
    Array3D<char> liquidMask;
    
    // Simulation parameters
    float ambientTemperature = 0.0f;
    float buoyancyAlpha = 0.1f;    // Density coefficient (smoke falls)
    float buoyancyBeta = 1.0f;     // Temperature coefficient (hot rises)
    float diffusion = 0.0001f;      // Velocity diffusion
    float densityDiffusion = 0.0001f;
    float dissipation = 0.99f;      // Density decay
    float tempDissipation = 0.995f; // Temperature decay
    
    // Pressure solver iterations
    int pressureIterations = 50;
    
    SmokeSim();
    
    void init(int nx, int ny, int nz);
    void reset();
    
    // Main simulation step
    void step(float dt);
    
    // Add sources
    void addDensity(const glm::vec3& worldPos, float amount, float radius);
    void addTemperature(const glm::vec3& worldPos, float amount, float radius);
    void addVelocity(const glm::vec3& worldPos, const glm::vec3& vel, float radius);
    
    // Set liquid occupancy for coupling
    void setLiquidMask(const std::vector<glm::vec3>& particlePositions, float particleRadius);
    void clearLiquidMask();
    
    // Sample velocity at world position
    glm::vec3 sampleVelocity(const glm::vec3& worldPos) const;
    float sampleDensity(const glm::vec3& worldPos) const;
    float sampleTemperature(const glm::vec3& worldPos) const;
    
    // Grid coordinate conversions
    glm::vec3 worldToGrid(const glm::vec3& worldPos) const;
    glm::vec3 gridToWorld(int i, int j, int k) const;
    
    // Get density data for rendering
    const std::vector<float>& getDensityData() const { return density.data; }
    
private:
    // Stable Fluids steps
    void addForces(float dt);
    void advectVelocity(float dt);
    void diffuseVelocity(float dt);
    void project(float dt);
    void advectScalars(float dt);
    
    // Boundary conditions
    void setBoundaryConditions();
    void applyLiquidBoundary(const glm::vec3& liquidVelocity);
    
    // Helper for sampling velocities with MAC offset
    float sampleU(float x, float y, float z) const;
    float sampleV(float x, float y, float z) const;
    float sampleW(float x, float y, float z) const;
};

