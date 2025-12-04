#include "MpmSim.h"
#include <algorithm>
#include <cmath>
#include <iostream>

MpmSim::MpmSim() {
    dx = (worldMax.x - worldMin.x) / static_cast<float>(Nx);
}

void MpmSim::init(int nx, int ny, int nz) {
    Nx = nx;
    Ny = ny;
    Nz = nz;
    
    dx = (worldMax.x - worldMin.x) / static_cast<float>(Nx);
    
    grid.resize(Nx * Ny * Nz);
    reset();
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
    
    for (float x = min.x; x <= max.x; x += particleSpacing) {
        for (float y = min.y; y <= max.y; y += particleSpacing) {
            for (float z = min.z; z <= max.z; z += particleSpacing) {
                Particle p;
                p.x = glm::vec3(x, y, z);
                p.v = velocity;
                p.mass = mass;
                p.volume0 = volume;
                p.F = glm::mat3(1.0f);
                p.C = glm::mat3(0.0f);
                p.E = WATER_E;
                p.nu = WATER_NU;
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
    
    for (float x = center.x - radius; x <= center.x + radius; x += particleSpacing) {
        for (float y = center.y - radius; y <= center.y + radius; y += particleSpacing) {
            for (float z = center.z - radius; z <= center.z + radius; z += particleSpacing) {
                glm::vec3 pos(x, y, z);
                glm::vec3 diff = pos - center;
                if (glm::dot(diff, diff) <= r2) {
                    Particle p;
                    p.x = pos;
                    p.v = velocity;
                    p.mass = mass;
                    p.volume0 = volume;
                    p.F = glm::mat3(1.0f);
                    p.C = glm::mat3(0.0f);
                    p.E = WATER_E;
                    p.nu = WATER_NU;
                    particles.push_back(p);
                }
            }
        }
    }
}

void MpmSim::step(float dt) {
    impactPositions.clear();
    clearGrid();
    particleToGrid(dt);
    applyGridForcesAndBCs(dt);
    gridToParticle(dt);
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
    for (auto& node : grid) {
        node.mass = 0.0f;
        node.vel = glm::vec3(0.0f);
        node.velNew = glm::vec3(0.0f);
    }
}

void MpmSim::particleToGrid(float dt) {
    float invDx = 1.0f / dx;
    
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
        
        // Compute stress contribution (Neo-Hookean for fluid-like behavior)
        glm::mat3 stress = computeStress(p);
        
        // MLS-MPM: equation (29) from Hu et al.
        // Compute affine momentum contribution
        // NOTE: Simplified to just stress term for stability (remove APIC momentum for now)
        glm::mat3 affine = -dt * 4.0f * invDx * invDx * p.volume0 * stress;
        // glm::mat3 affine = -dt * 4.0f * invDx * invDx * p.volume0 * stress + p.mass * p.C;
        
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
    
    // Normalize velocities by mass
    for (auto& node : grid) {
        if (node.mass > 1e-6f) {
            node.vel /= node.mass;
        }
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
                
                // Apply gravity
                node.vel.y += dt * gravity;
                
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
    
    for (auto& p : particles) {
        glm::vec3 oldVel = p.v;
        
        // Find base cell
        glm::vec3 cellPos = (p.x - worldMin) * invDx;
        glm::ivec3 base = glm::ivec3(cellPos - 0.5f);
        glm::vec3 fx = cellPos - glm::vec3(base);
        
        // Quadratic B-spline weights and derivatives
        glm::vec3 w[3], dw[3];
        w[0] = 0.5f * glm::pow(1.5f - fx, glm::vec3(2.0f));
        w[1] = 0.75f - glm::pow(fx - 1.0f, glm::vec3(2.0f));
        w[2] = 0.5f * glm::pow(fx - 0.5f, glm::vec3(2.0f));
        
        dw[0] = fx - 1.5f;
        dw[1] = -2.0f * (fx - 1.0f);
        dw[2] = fx - 0.5f;
        
        // Gather velocity and velocity gradient from grid
        glm::vec3 velPIC(0.0f);
        glm::mat3 B(0.0f);  // Affine velocity field
        
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
                    glm::vec3 gridVel = grid[idx].velNew;
                    
                    velPIC += weight * gridVel;
                    
                    // APIC: compute affine velocity field
                    B += weight * glm::outerProduct(gridVel, dpos);
                }
            }
        }
        
        // APIC: C = B * D^-1, where D = (1/4) * dx^2 * I for quadratic B-splines
        p.C = B * (4.0f * invDx * invDx);
        
        // FLIP/PIC blend
        glm::vec3 velFLIP = p.v + (velPIC - oldVel);
        p.v = flipRatio * velFLIP + (1.0f - flipRatio) * velPIC;
        
        // Detect high-speed impacts
        float speedChange = glm::length(p.v - oldVel);
        if (speedChange > 2.0f && p.x.y < worldMin.y + 0.1f) {
            impactPositions.push_back(p.x);
        }
        
        // Update position
        p.x += dt * p.v;
        
        // Update deformation gradient (disabled for pure fluid - just track volume change)
        // For now, keep F = I to avoid numerical instability
        // glm::mat3 gradV = p.C;
        // p.F = (glm::mat3(1.0f) + dt * gradV) * p.F;
        p.F = glm::mat3(1.0f);  // Reset to identity for incompressible fluid
        
        // Clamp to world bounds (safety)
        float margin = 2.0f * dx;
        p.x = glm::clamp(p.x, worldMin + margin, worldMax - margin);
    }
}

int MpmSim::gridIndex(int i, int j, int k) const {
    return i + Nx * (j + Ny * k);
}

bool MpmSim::isValidCell(int i, int j, int k) const {
    return i >= 0 && i < Nx && j >= 0 && j < Ny && k >= 0 && k < Nz;
}

float MpmSim::N(float x) const {
    // Quadratic B-spline
    x = std::abs(x);
    if (x < 0.5f) {
        return 0.75f - x * x;
    } else if (x < 1.5f) {
        return 0.5f * (1.5f - x) * (1.5f - x);
    }
    return 0.0f;
}

float MpmSim::dN(float x) const {
    // Derivative of quadratic B-spline
    float sign = (x >= 0.0f) ? 1.0f : -1.0f;
    x = std::abs(x);
    if (x < 0.5f) {
        return -2.0f * x * sign;
    } else if (x < 1.5f) {
        return (x - 1.5f) * sign;
    }
    return 0.0f;
}

glm::mat3 MpmSim::computeStress(const Particle& p) const {
    // Neo-Hookean model for fluid-like behavior
    float E = p.E;
    float nu = p.nu;
    
    // Lame parameters
    float mu = E / (2.0f * (1.0f + nu));
    float lambda = E * nu / ((1.0f + nu) * (1.0f - 2.0f * nu));
    
    // For liquid: only use pressure (bulk modulus)
    // P = -p * I where p = -K * (J - 1)
    // K = lambda + 2/3 * mu (bulk modulus)
    
    float J = glm::determinant(p.F);
    
    // Clamp J to prevent instability
    J = std::max(J, 0.1f);
    
    // Pressure (equation of state for weakly compressible fluid)
    float bulkModulus = lambda + 2.0f * mu / 3.0f;
    float pressure = bulkModulus * (J - 1.0f);
    
    // Cauchy stress (just pressure for fluid)
    glm::mat3 stress = -pressure * glm::mat3(1.0f);
    
    return stress;
}

