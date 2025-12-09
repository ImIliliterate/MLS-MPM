#include "MpmSim.h"
#include "SimulationParams.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <unordered_map>

#ifdef USE_OPENMP
#include <omp.h>
#endif

MpmSim::MpmSim() {
    dx = (worldMax.x - worldMin.x) / static_cast<float>(Nx);
}

void MpmSim::init(int nx, int ny, int nz) {
    Nx = nx;
    Ny = ny;
    Nz = nz;
    
    dx = (worldMax.x - worldMin.x) / static_cast<float>(Nx);
    
    grid.resize(Nx * Ny * Nz);
    
#ifdef USE_CUDA
    std::cout << "[CUDA] GPU simulation enabled" << std::endl;
#endif
    
    reset();
}

void MpmSim::logResolutionStats() const {
    int numCells = Nx * Ny * Nz;
    int numParticles = static_cast<int>(particles.size());
    float avgPPC = static_cast<float>(numParticles) / static_cast<float>(numCells);
    
    // Count active cells (cells with mass)
    int activeCells = 0;
    for (const auto& node : grid) {
        if (node.mass > 1e-6f) activeCells++;
    }
    float activePPC = activeCells > 0 ? static_cast<float>(numParticles) / static_cast<float>(activeCells) : 0.0f;
    
    std::cout << "\n===== RESOLUTION STATS =====" << std::endl;
    std::cout << "Grid: " << Nx << "x" << Ny << "x" << Nz << " = " << numCells << " cells" << std::endl;
    std::cout << "Cell size (dx): " << dx << std::endl;
    std::cout << "Particles: " << numParticles << std::endl;
    std::cout << "Avg PPC (all cells): " << avgPPC << std::endl;
    std::cout << "Avg PPC (active cells): " << activePPC << std::endl;
    std::cout << "Target: >= 8 PPC in filled regions" << std::endl;
    std::cout << "============================\n" << std::endl;
}

void MpmSim::reset() {
    particles.clear();
    clearGrid();
    impactPositions.clear();
}

void MpmSim::addBox(const glm::vec3& min, const glm::vec3& max, const glm::vec3& velocity,
                    float particleSpacing, float density) {
    float volume = particleSpacing * particleSpacing * particleSpacing;
    float mass = density * volume;
    
    // Jitter amplitude: ±10% of spacing to break grid aliasing
    float jitterAmp = particleSpacing * 0.1f;
    
    for (float x = min.x; x <= max.x; x += particleSpacing) {
        for (float y = min.y; y <= max.y; y += particleSpacing) {
            for (float z = min.z; z <= max.z; z += particleSpacing) {
                Particle p;
                // Add random jitter to break lattice symmetry
                float jx = jitterAmp * (2.0f * (rand() / (float)RAND_MAX) - 1.0f);
                float jy = jitterAmp * (2.0f * (rand() / (float)RAND_MAX) - 1.0f);
                float jz = jitterAmp * (2.0f * (rand() / (float)RAND_MAX) - 1.0f);
                p.x = glm::vec3(x + jx, y + jy, z + jz);
                p.v = velocity;
                p.mass = mass;
                p.volume0 = volume;
                p.F = glm::mat3(1.0f);
                p.C = glm::mat3(0.0f);
                p.E = g_params.bulkModulus;
                p.nu = g_params.poissonRatio;
                particles.push_back(p);
            }
        }
    }
}

void MpmSim::addSphere(const glm::vec3& center, float radius, const glm::vec3& velocity,
                       float particleSpacing, float density) {
    float volume = particleSpacing * particleSpacing * particleSpacing;
    float mass = density * volume;
    float r2 = radius * radius;
    
    // Jitter amplitude: ±10% of spacing to break grid aliasing
    float jitterAmp = particleSpacing * 0.1f;
    
    for (float x = center.x - radius; x <= center.x + radius; x += particleSpacing) {
        for (float y = center.y - radius; y <= center.y + radius; y += particleSpacing) {
            for (float z = center.z - radius; z <= center.z + radius; z += particleSpacing) {
                glm::vec3 pos(x, y, z);
                glm::vec3 diff = pos - center;
                if (glm::dot(diff, diff) <= r2) {
                    Particle p;
                    // Add random jitter to break lattice symmetry
                    float jx = jitterAmp * (2.0f * (rand() / (float)RAND_MAX) - 1.0f);
                    float jy = jitterAmp * (2.0f * (rand() / (float)RAND_MAX) - 1.0f);
                    float jz = jitterAmp * (2.0f * (rand() / (float)RAND_MAX) - 1.0f);
                    p.x = pos + glm::vec3(jx, jy, jz);
                    p.v = velocity;
                    p.mass = mass;
                    p.volume0 = volume;
                    p.F = glm::mat3(1.0f);
                    p.C = glm::mat3(0.0f);
                    p.E = g_params.bulkModulus;
                    p.nu = g_params.poissonRatio;
                    particles.push_back(p);
                }
            }
        }
    }
}

float MpmSim::computeAdaptiveDt() const {
    if (!g_params.adaptiveDt) {
        return g_params.dt;
    }
    
#ifdef USE_CUDA
    // GPU path: use CUDA-computed CFL-safe dt
    extern float mpmCudaComputeSafeDt(int numParticles, float dx, float bulkModulus, float density);
    float safeDt = mpmCudaComputeSafeDt(
        static_cast<int>(particles.size()),
        dx,
        g_params.bulkModulus,
        g_params.restDensity
    );
    return safeDt;
#else
    // CPU path: compute CFL from particle velocities
    float h = (worldMax.x - worldMin.x) / static_cast<float>(Nx);  // cell size
    float maxV = std::max(g_params.maxLiquidSpeed, 0.1f);  // minimum to avoid div by zero
    
    // CFL for advection: dt < C * h / v_max
    float dtAdvect = g_params.cflNumber * h / maxV;
    
    // CFL for stiffness (sound speed): c = sqrt(K / rho), dt < C * h / c
    // This prevents pressure waves from traveling more than one cell per step
    float soundSpeed = std::sqrt(g_params.stiffnessK / g_params.restDensity);
    float dtSound = g_params.cflNumber * h / soundSpeed;
    
    // Take the minimum of both constraints
    float dtCfl = std::min(dtAdvect, dtSound);
    
    // Clamp to user-specified range
    return std::clamp(dtCfl, g_params.dt * 0.1f, g_params.maxDt);
#endif
}

void MpmSim::computeDiagnostics() {
    // Compute max liquid speed
    float maxSpeed = 0.0f;
    for (const auto& p : particles) {
        float speed = glm::length(p.v);
        maxSpeed = std::max(maxSpeed, speed);
    }
    g_params.maxLiquidSpeed = maxSpeed;
    
    // Particle stats
    g_params.numLiquidParticles = static_cast<int>(particles.size());
    int totalCells = Nx * Ny * Nz;
    g_params.avgParticlesPerCell = static_cast<float>(particles.size()) / static_cast<float>(totalCells);
}

void MpmSim::step(float dt) {
    impactPositions.clear();
    g_params.frameCount++;
    
    // Round 6: Only compute full diagnostics every N frames
    bool fullDiagnostics = !g_params.skipDiagnosticsEveryFrame || 
                           (g_params.frameCount % g_params.diagnosticsInterval == 0);
    if (fullDiagnostics) {
        computeDiagnostics();
    }
    
    // Compute adaptive timestep
    float adaptDt = computeAdaptiveDt();
    g_params.actualDt = adaptDt;
    
    // Apply cohesion/surface tension forces (expensive, can be disabled)
    // Note: Cohesion runs on CPU and requires particle download, so skip on GPU
#ifndef USE_CUDA
    if (enableCohesion) {
        applyCohesionForces(adaptDt);
    }
#endif

#ifdef USE_CUDA
    extern void mpmCudaStepSimple(
        std::vector<Particle>& particles,
        int Nx, int Ny, int Nz,
        float dx, float dt,
        float apicBlend, float flipRatio, float gravity,
        const glm::vec3& worldMin, const glm::vec3& worldMax);
    
    mpmCudaStepSimple(particles, Nx, Ny, Nz, dx, adaptDt,
                      g_params.apicBlend, flipRatio, g_params.gravity,
                      worldMin, worldMax);
    return;
#endif

    clearGrid();
    particleToGrid(adaptDt);
    
    // Round 3: Compute fluid pressure and apply pressure gradient forces
    if (g_params.enablePressure) {
        computeGridDensity();
        computeGridPressure();
        applyPressureForces(adaptDt);
    }
    
    // Optional grid velocity smoothing (disabled for water)
    if (g_params.enableGridSmoothing) {
        smoothGridVelocities();
    }
    
    applyGridForcesAndBCs(adaptDt);
    gridToParticle(adaptDt);
}

void MpmSim::smoothGridVelocities() {
    /**
     * Phase 1.5 / 2.3: Jacobi smoothing on grid velocities (viscosity)
     * 
     * A tiny amount of grid-space viscosity can kill "grape clumps"
     * without making it soup. Use viscosityBlend to control amount.
     */
    
    std::vector<glm::vec3> smoothed(grid.size());
    float blend = g_params.viscosityBlend;  // How much to blend toward smoothed
    
    for (int iter = 0; iter < g_params.gridSmoothingIterations; iter++) {
        // Compute smoothed velocities
        for (int k = 0; k < Nz; k++) {
            for (int j = 0; j < Ny; j++) {
                for (int i = 0; i < Nx; i++) {
                    int idx = gridIndex(i, j, k);
                    
                    // Skip empty cells
                    if (grid[idx].mass < 1e-6f) {
                        smoothed[idx] = grid[idx].vel;
                        continue;
                    }
                    
                    // Average with neighbors (only non-empty)
                    glm::vec3 sum = grid[idx].vel;
                    int count = 1;
                    
                    // 6-connected neighbors
                    const int neighbors[6][3] = {
                        {-1, 0, 0}, {1, 0, 0},
                        {0, -1, 0}, {0, 1, 0},
                        {0, 0, -1}, {0, 0, 1}
                    };
                    
                    for (int n = 0; n < 6; n++) {
                        int ni = i + neighbors[n][0];
                        int nj = j + neighbors[n][1];
                        int nk = k + neighbors[n][2];
                        
                        if (isValidCell(ni, nj, nk)) {
                            int nidx = gridIndex(ni, nj, nk);
                            if (grid[nidx].mass > 1e-6f) {
                                sum += grid[nidx].vel;
                                count++;
                            }
                        }
                    }
                    
                    glm::vec3 avgVel = sum / static_cast<float>(count);
                    // Blend: grid.vel = mix(original, smoothed, viscosityBlend)
                    smoothed[idx] = glm::mix(grid[idx].vel, avgVel, blend);
                }
            }
        }
        
        // Copy back to grid
        for (int k = 0; k < Nz; k++) {
            for (int j = 0; j < Ny; j++) {
                for (int i = 0; i < Nx; i++) {
                    int idx = gridIndex(i, j, k);
                    grid[idx].vel = smoothed[idx];
                }
            }
        }
    }
}

glm::vec3 MpmSim::gridToWorld(int i, int j, int k) const {
    return worldMin + glm::vec3(i, j, k) * dx;
}

glm::ivec3 MpmSim::worldToGrid(const glm::vec3& pos) const {
    glm::vec3 rel = (pos - worldMin) / dx;
    return glm::ivec3(
        static_cast<int>(std::floor(rel.x)),
        static_cast<int>(std::floor(rel.y)),
        static_cast<int>(std::floor(rel.z))
    );
}

std::vector<glm::vec3> MpmSim::getImpactPositions() const {
    return impactPositions;
}

void MpmSim::applyExternalForce(const glm::vec3& pos, const glm::vec3& force, float radius) {
    float r2 = radius * radius;
    for (auto& p : particles) {
        glm::vec3 diff = p.x - pos;
        float d2 = glm::dot(diff, diff);
        if (d2 < r2) {
            float w = 1.0f - std::sqrt(d2) / radius;
            p.v += (force / p.mass) * w;
        }
    }
}

void MpmSim::clearGrid() {
    // Round 6: Use memset for faster zeroing (GridNode is POD-like)
    std::memset(grid.data(), 0, grid.size() * sizeof(GridNode));
}

// ==================== Round 3: Pressure-based Fluid Forces ====================
// Round 6: Combined density + pressure computation for better cache efficiency

void MpmSim::computeGridDensity() {
    /**
     * Round 6 OPTIMIZED: Compute density per cell from mass
     * Combined with pressure computation in computeGridPressure for efficiency
     */
    float cellVolume = dx * dx * dx;
    float invCellVolume = 1.0f / cellVolume;
    int activeCells = 0;
    float maxDens = 0.0f;
    
    // Only compute detailed diagnostics every N frames
    bool computeStats = !g_params.skipDiagnosticsEveryFrame || 
                        (g_params.frameCount % g_params.diagnosticsInterval == 0);
    
    int gridSize = Nx * Ny * Nz;
    #ifdef USE_OPENMP
    #pragma omp parallel for reduction(+:activeCells) reduction(max:maxDens)
    #endif
    for (int idx = 0; idx < gridSize; idx++) {
        float mass = grid[idx].mass;
        if (mass > 0.0f) {
            grid[idx].density = mass * invCellVolume;
            if (computeStats) {
                maxDens = std::max(maxDens, grid[idx].density);
                activeCells++;
            }
        } else {
            grid[idx].density = 0.0f;
        }
    }
    
    // Update diagnostics only when computed
    if (computeStats) {
        g_params.activeCellCount = activeCells;
        g_params.maxDensity = maxDens;
        if (activeCells > 0) {
            g_params.activePPC = static_cast<float>(particles.size()) / static_cast<float>(activeCells);
        }
    }
}

void MpmSim::computeGridPressure() {
    /**
     * Round 6 OPTIMIZED: Compute pressure from equation of state
     * Uses flat array iteration for better cache performance
     */
    float rho0 = g_params.restDensity;
    float K = g_params.stiffnessK;
    float invRho0 = 1.0f / rho0;
    float maxPress = 0.0f;
    
    bool computeStats = !g_params.skipDiagnosticsEveryFrame || 
                        (g_params.frameCount % g_params.diagnosticsInterval == 0);
    
    int gridSize = Nx * Ny * Nz;
    #ifdef USE_OPENMP
    #pragma omp parallel for reduction(max:maxPress)
    #endif
    for (int idx = 0; idx < gridSize; idx++) {
        float rho = grid[idx].density;
        
        if (rho > 0.0f) {
            float compression = (rho - rho0) * invRho0;
            float p = K * std::max(0.0f, compression);
            grid[idx].pressure = p;
            if (computeStats) {
                maxPress = std::max(maxPress, p);
            }
        } else {
            grid[idx].pressure = 0.0f;
        }
    }
    
    if (computeStats) {
        g_params.maxPressure = maxPress;
    }
}

void MpmSim::applyPressureForces(float dt) {
    /**
     * Phase 1.4: Apply pressure forces as grid velocity changes
     * 
     * Acceleration: a = -∇p / ρ
     * 
     * Uses central difference for pressure gradient
     */
    float invTwoDx = 1.0f / (2.0f * dx);
    
    #ifdef USE_OPENMP
    // MSVC OpenMP on Windows does not support collapse for these nested loops reliably
    #pragma omp parallel for
    #endif
    for (int k = 1; k < Nz - 1; k++) {
        for (int j = 1; j < Ny - 1; j++) {
            for (int i = 1; i < Nx - 1; i++) {
                int idx = gridIndex(i, j, k);
                
                // Skip cells with no density
                if (grid[idx].density <= 0.0f) continue;
                
                // Get neighbor pressures
                float pL = grid[gridIndex(i - 1, j, k)].pressure;
                float pR = grid[gridIndex(i + 1, j, k)].pressure;
                float pD = grid[gridIndex(i, j - 1, k)].pressure;
                float pU = grid[gridIndex(i, j + 1, k)].pressure;
                float pB = grid[gridIndex(i, j, k - 1)].pressure;
                float pF = grid[gridIndex(i, j, k + 1)].pressure;
                
                // Central difference gradient
                glm::vec3 gradP;
                gradP.x = (pR - pL) * invTwoDx;
                gradP.y = (pU - pD) * invTwoDx;
                gradP.z = (pF - pB) * invTwoDx;
                
                // Acceleration: a = -∇p / ρ
                float rho = grid[idx].density;
                glm::vec3 accel = -gradP / (rho + 1e-6f);
                
                // Apply to grid velocity
                grid[idx].vel += accel * dt;
            }
        }
    }
}

void MpmSim::particleToGrid(float dt) {
    float invDx = 1.0f / dx;
    
    // Store old grid velocities for FLIP
    std::vector<glm::vec3> oldGridVel(grid.size());
    for (size_t i = 0; i < grid.size(); i++) {
        oldGridVel[i] = grid[i].vel;
    }
    
    for (auto& p : particles) {
        // Find base cell
        glm::vec3 cellPos = (p.x - worldMin) * invDx;
        glm::ivec3 base = glm::ivec3(cellPos - 0.5f);
        glm::vec3 fx = cellPos - glm::vec3(base);
        
        // Quadratic B-spline weights
        glm::vec3 w[3];
        w[0] = 0.5f * glm::pow(1.5f - fx, glm::vec3(2.0f));
        w[1] = 0.75f - glm::pow(fx - 1.0f, glm::vec3(2.0f));
        w[2] = 0.5f * glm::pow(fx - 0.5f, glm::vec3(2.0f));
        
        // Compute stress contribution
        glm::mat3 stress = computeStress(p);
        
        // MLS-MPM: equation (29) from Hu et al.
        // APIC contribution controlled by params
        float apicBlend = g_params.apicBlend;
        glm::mat3 affine = -dt * 4.0f * invDx * invDx * p.volume0 * stress + p.mass * p.C * apicBlend;
        
        // Scatter to 3x3x3 neighborhood
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    int gi = base.x + i;
                    int gj = base.y + j;
                    int gk = base.z + k;
                    
                    if (!isValidCell(gi, gj, gk)) continue;
                    
                    float weight = w[i].x * w[j].y * w[k].z;
                    glm::vec3 dpos = (glm::vec3(i, j, k) - fx) * dx;
                    
                    int idx = gridIndex(gi, gj, gk);
                    grid[idx].mass += weight * p.mass;
                    grid[idx].vel += weight * (p.mass * p.v + affine * dpos);
                }
            }
        }
    }
    
    // Normalize velocities by mass and store old velocity
    for (size_t i = 0; i < grid.size(); i++) {
        if (grid[i].mass > 1e-6f) {
            grid[i].vel /= grid[i].mass;
        }
        // Store pre-force velocity for FLIP
        oldGridVel[i] = grid[i].vel;
    }
}

void MpmSim::applyGridForcesAndBCs(float dt) {
    // Apply gravity and boundary conditions
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            for (int k = 0; k < Nz; k++) {
                int idx = gridIndex(i, j, k);
                GridNode& node = grid[idx];
                
                if (node.mass <= 1e-6f) continue;
                
                // Apply gravity (use params)
                node.vel.y += dt * g_params.gravity;
                
                // Boundary conditions (walls)
                glm::vec3 worldPos = gridToWorld(i, j, k);
                int boundary = 3;  // Number of cells from edge
                float boundaryDist = boundary * dx;
                
                // Floor
                if (worldPos.y < worldMin.y + boundaryDist) {
                    node.vel.y = std::max(node.vel.y, 0.0f);
                }
                // Ceiling
                if (worldPos.y > worldMax.y - boundaryDist) {
                    node.vel.y = std::min(node.vel.y, 0.0f);
                }
                // Left/Right walls
                if (worldPos.x < worldMin.x + boundaryDist) {
                    node.vel.x = std::max(node.vel.x, 0.0f);
                }
                if (worldPos.x > worldMax.x - boundaryDist) {
                    node.vel.x = std::min(node.vel.x, 0.0f);
                }
                // Front/Back walls
                if (worldPos.z < worldMin.z + boundaryDist) {
                    node.vel.z = std::max(node.vel.z, 0.0f);
                }
                if (worldPos.z > worldMax.z - boundaryDist) {
                    node.vel.z = std::min(node.vel.z, 0.0f);
                }
                
                // SDF-based collision
                if (sdfFunc) {
                    float sdf = sdfFunc(worldPos);
                    if (sdf < 0) {
                        if (sdfGradFunc) {
                            glm::vec3 normal = sdfGradFunc(worldPos);
                            float len = glm::length(normal);
                            if (len > 1e-6f) {
                                normal /= len;
                                float vn = glm::dot(node.vel, normal);
                                if (vn < 0) {
                                    // Remove inward velocity component
                                    node.vel -= vn * normal;
                                    // Add friction
                                    node.vel *= 0.9f;
                                }
                            }
                        }
                    }
                }
                
                // Store for FLIP
                node.velNew = node.vel;
            }
        }
    }
}

void MpmSim::gridToParticle(float dt) {
    float invDx = 1.0f / dx;
    float flipAlpha = g_params.flipRatio;
    int numParticles = static_cast<int>(particles.size());
    
    // Thread-local storage for impact positions
    #ifdef USE_OPENMP
    std::vector<std::vector<glm::vec3>> threadImpacts(omp_get_max_threads());
    #pragma omp parallel for schedule(static)
    #endif
    for (int pi = 0; pi < numParticles; pi++) {
        Particle& p = particles[pi];
        glm::vec3 oldVel = p.v;
        
        // Find base cell
        glm::vec3 cellPos = (p.x - worldMin) * invDx;
        glm::ivec3 base = glm::ivec3(cellPos - 0.5f);
        glm::vec3 fx = cellPos - glm::vec3(base);
        
        // Quadratic B-spline weights
        glm::vec3 w[3];
        w[0] = 0.5f * glm::pow(1.5f - fx, glm::vec3(2.0f));
        w[1] = 0.75f - glm::pow(fx - 1.0f, glm::vec3(2.0f));
        w[2] = 0.5f * glm::pow(fx - 0.5f, glm::vec3(2.0f));
        
        // Gather velocity and affine field from grid
        glm::vec3 velPIC(0.0f);
        glm::vec3 velOld(0.0f);
        glm::mat3 B(0.0f);
        
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    int gi = base.x + i;
                    int gj = base.y + j;
                    int gk = base.z + k;
                    
                    if (!isValidCell(gi, gj, gk)) continue;
                    
                    float weight = w[i].x * w[j].y * w[k].z;
                    glm::vec3 dpos = (glm::vec3(i, j, k) - fx) * dx;
                    
                    int idx = gridIndex(gi, gj, gk);
                    glm::vec3 gridVelNew = grid[idx].velNew;
                    glm::vec3 gridVelOld = grid[idx].vel;
                    
                    velPIC += weight * gridVelNew;
                    velOld += weight * gridVelOld;
                    B += weight * glm::outerProduct(gridVelNew, dpos);
                }
            }
        }
        
        // APIC: C = B * D^-1
        p.C = B * (4.0f * invDx * invDx);
        
        // FLIP/PIC blend
        glm::vec3 dv_flip = velPIC - velOld;
        glm::vec3 velFLIP = p.v + dv_flip;
        p.v = flipAlpha * velFLIP + (1.0f - flipAlpha) * velPIC;
        
        // Detect high-speed impacts (thread-safe)
        float speedChange = glm::length(p.v - oldVel);
        if (speedChange > 2.0f && p.x.y < worldMin.y + 0.1f) {
            #ifdef USE_OPENMP
            threadImpacts[omp_get_thread_num()].push_back(p.x);
            #else
            impactPositions.push_back(p.x);
            #endif
        }
        
        // Update position
        p.x += dt * p.v;
        
        // Update deformation gradient
        glm::mat3 gradV = p.C;
        p.F = (glm::mat3(1.0f) + dt * gradV) * p.F;
        
        // For fluid: reset F to preserve volume ratio
        float J = glm::determinant(p.F);
        J = glm::clamp(J, 0.6f, 1.5f);
        J = glm::mix(J, 1.0f, 0.01f);
        float cbrtJ = std::cbrt(J);
        p.F = glm::mat3(cbrtJ);
        
        // Clamp to world bounds
        float margin = 2.0f * dx;
        p.x = glm::clamp(p.x, worldMin + margin, worldMax - margin);
    }
    
    // Merge thread-local impacts
    #ifdef USE_OPENMP
    for (const auto& ti : threadImpacts) {
        impactPositions.insert(impactPositions.end(), ti.begin(), ti.end());
    }
    #endif
}

int MpmSim::gridIndex(int i, int j, int k) const {
    return i + Nx * (j + Ny * k);
}

bool MpmSim::isValidCell(int i, int j, int k) const {
    return i >= 0 && i < Nx && j >= 0 && j < Ny && k >= 0 && k < Nz;
}

float MpmSim::N(float x) const {
    x = std::abs(x);
    if (x < 0.5f) {
        return 0.75f - x * x;
    } else if (x < 1.5f) {
        return 0.5f * (1.5f - x) * (1.5f - x);
    }
    return 0.0f;
}

float MpmSim::dN(float x) const {
    float sign = (x >= 0.0f) ? 1.0f : -1.0f;
    x = std::abs(x);
    if (x < 0.5f) {
        return -2.0f * x * sign;
    } else if (x < 1.5f) {
        return (x - 1.5f) * sign;
    }
    return 0.0f;
}

void MpmSim::applyCohesionForces(float dt) {
    const float h = dx * 2.5f;
    const float h2 = h * h;
    const float cohesionStrength = 800.0f;
    
    std::unordered_map<int64_t, std::vector<int>> spatialHash;
    auto hashPos = [&](const glm::vec3& pos) -> int64_t {
        int cx = static_cast<int>(std::floor(pos.x / h));
        int cy = static_cast<int>(std::floor(pos.y / h));
        int cz = static_cast<int>(std::floor(pos.z / h));
        return (static_cast<int64_t>(cx) * 73856093) ^
               (static_cast<int64_t>(cy) * 19349663) ^
               (static_cast<int64_t>(cz) * 83492791);
    };
    
    for (int i = 0; i < static_cast<int>(particles.size()); i++) {
        spatialHash[hashPos(particles[i].x)].push_back(i);
    }
    
    std::vector<float> densities(particles.size(), 0.0f);
    for (int i = 0; i < static_cast<int>(particles.size()); i++) {
        const auto& pi = particles[i];
        int cx = static_cast<int>(std::floor(pi.x.x / h));
        int cy = static_cast<int>(std::floor(pi.x.y / h));
        int cz = static_cast<int>(std::floor(pi.x.z / h));
        
        float density = 0.0f;
        
        for (int di = -1; di <= 1; di++) {
            for (int dj = -1; dj <= 1; dj++) {
                for (int dk = -1; dk <= 1; dk++) {
                    int64_t hash = (static_cast<int64_t>(cx + di) * 73856093) ^
                                   (static_cast<int64_t>(cy + dj) * 19349663) ^
                                   (static_cast<int64_t>(cz + dk) * 83492791);
                    
                    auto it = spatialHash.find(hash);
                    if (it == spatialHash.end()) continue;
                    
                    for (int j : it->second) {
                        glm::vec3 diff = pi.x - particles[j].x;
                        float r2 = glm::dot(diff, diff);
                        if (r2 < h2) {
                            float w = (h2 - r2);
                            density += particles[j].mass * w * w * w;
                        }
                    }
                }
            }
        }
        
        float poly6Coeff = 315.0f / (64.0f * 3.14159265f * std::pow(h, 9));
        densities[i] = density * poly6Coeff;
    }
    
    for (int i = 0; i < static_cast<int>(particles.size()); i++) {
        auto& pi = particles[i];
        int cx = static_cast<int>(std::floor(pi.x.x / h));
        int cy = static_cast<int>(std::floor(pi.x.y / h));
        int cz = static_cast<int>(std::floor(pi.x.z / h));
        
        glm::vec3 cohesionForce(0.0f);
        glm::vec3 normal(0.0f);
        
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
                        
                        glm::vec3 diff = pi.x - particles[j].x;
                        float r2 = glm::dot(diff, diff);
                        
                        if (r2 < h2 && r2 > 1e-10f) {
                            float r = std::sqrt(r2);
                            glm::vec3 dir = diff / r;
                            
                            float q = r / h;
                            if (q < 1.0f) {
                                float cohesionKernel = 0.0f;
                                if (q < 0.5f) {
                                    cohesionKernel = 2.0f * std::pow(1.0f - q, 3) * q * q * q - 1.0f / 64.0f;
                                } else {
                                    cohesionKernel = std::pow(1.0f - q, 3) * q * q * q;
                                }
                                cohesionForce -= cohesionStrength * particles[j].mass * cohesionKernel * dir;
                            }
                            
                            float w = (h2 - r2) * (h2 - r2);
                            normal += particles[j].mass * w * dir / (densities[j] + 1e-6f);
                        }
                    }
                }
            }
        }
        
        float normalLen = glm::length(normal);
        if (normalLen > 0.1f) {
            cohesionForce -= 50.0f * normal;
        }
        
        pi.v += dt * cohesionForce / pi.mass;
        
        float speed = glm::length(pi.v);
        if (speed > 10.0f) {
            pi.v *= 10.0f / speed;
        }
    }
}

glm::mat3 MpmSim::computeStress(const Particle& p) const {
    float E = p.E;
    float nu = p.nu;
    
    float mu = E / (2.0f * (1.0f + nu));
    float lambda = E * nu / ((1.0f + nu) * (1.0f - 2.0f * nu));
    
    float J = glm::determinant(p.F);
    J = std::max(J, 0.1f);
    
    float bulkModulus = lambda + 2.0f * mu / 3.0f;
    float pressure = bulkModulus * (J - 1.0f);
    
    glm::mat3 stress = -pressure * glm::mat3(1.0f);
    
    return stress;
}

// ==================== GPU Support Methods ====================

#ifdef USE_CUDA
extern void mpmCudaGetPositions(std::vector<glm::vec3>& outPositions);
extern void mpmCudaMarkDirty();
#endif

void MpmSim::getPositionsForRendering(std::vector<glm::vec3>& positions) const {
#ifdef USE_CUDA
    mpmCudaGetPositions(positions);
    return;
#endif
    positions.resize(particles.size());
    for (size_t i = 0; i < particles.size(); i++) {
        positions[i] = particles[i].x;
    }
}

void MpmSim::syncParticlesToCpu() {
#ifdef USE_CUDA
    mpmCudaDownloadParticles(particles);
#endif
}

void MpmSim::uploadToGpu() {
#ifdef USE_CUDA
    mpmCudaMarkDirty();
#endif
}

bool MpmSim::isGpuEnabled() const {
#ifdef USE_CUDA
    return true;
#else
    return false;
#endif
}
