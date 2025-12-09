#include "SmokeSim.h"
#include <algorithm>
#include <cmath>

#ifdef USE_CUDA
#include "../gpu/smoke_cuda.h"
#include <cuda_runtime.h>
#endif

SmokeSim::SmokeSim() {
    dx = (worldMax.x - worldMin.x) / static_cast<float>(Nx);
}

SmokeSim::~SmokeSim() {
    cleanupGPU();
}

void SmokeSim::init(int nx, int ny, int nz) {
    Nx = nx;
    Ny = ny;
    Nz = nz;
    
    dx = (worldMax.x - worldMin.x) / static_cast<float>(Nx);
    
    // Staggered grid: u has Nx+1 values in x
    u.resize(Nx + 1, Ny, Nz, 0.0f);
    v.resize(Nx, Ny + 1, Nz, 0.0f);
    w.resize(Nx, Ny, Nz + 1, 0.0f);
    
    uTemp.resize(Nx + 1, Ny, Nz, 0.0f);
    vTemp.resize(Nx, Ny + 1, Nz, 0.0f);
    wTemp.resize(Nx, Ny, Nz + 1, 0.0f);
    
    density.resize(Nx, Ny, Nz, 0.0f);
    temperature.resize(Nx, Ny, Nz, ambientTemperature);
    
    densityTemp.resize(Nx, Ny, Nz, 0.0f);
    temperatureTemp.resize(Nx, Ny, Nz, ambientTemperature);
    
    pressure.resize(Nx, Ny, Nz, 0.0f);
    divergence.resize(Nx, Ny, Nz, 0.0f);
    
    liquidMask.resize(Nx, Ny, Nz, 0);
    
    reset();
}

void SmokeSim::initGPU() {
#ifdef USE_CUDA
    if (!gpuInitialized && useGPU) {
        initSmokeCuda(Nx, Ny, Nz);
        gpuInitialized = true;
    }
#endif
}

void SmokeSim::cleanupGPU() {
#ifdef USE_CUDA
    if (gpuInitialized) {
        cleanupSmokeCuda();
        gpuInitialized = false;
    }
#endif
}

void SmokeSim::reset() {
    u.fill(0.0f);
    v.fill(0.0f);
    w.fill(0.0f);
    density.fill(0.0f);
    temperature.fill(ambientTemperature);
    pressure.fill(0.0f);
    liquidMask.fill(0);
    
#ifdef USE_CUDA
    if (gpuInitialized) {
        resetSmokeCuda(ambientTemperature);
    }
#endif
}

void SmokeSim::step(float dt) {
#ifdef USE_CUDA
    if (useGPU && gpuInitialized) {
        // GPU path - call CUDA kernels
        smokeCudaStep(
            dt, dx,
            buoyancyAlpha, buoyancyBeta, ambientTemperature,
            dissipation, tempDissipation,
            pressureIterations
        );
        
        // Download density for rendering (only needed if rendering smoke)
        smokeCudaDownloadDensity(density.data.data());
        return;
    }
#endif
    
    // CPU path
    addForces(dt);
    advectVelocity(dt);
    // diffuseVelocity(dt);  // Optional, can be slow
    project(dt);
    advectScalars(dt);
    setBoundaryConditions();
}

void SmokeSim::addDensity(const glm::vec3& worldPos, float amount, float radius) {
#ifdef USE_CUDA
    if (useGPU && gpuInitialized) {
        float3 wmin = make_float3(worldMin.x, worldMin.y, worldMin.z);
        float3 pos = make_float3(worldPos.x, worldPos.y, worldPos.z);
        smokeCudaAddDensity(wmin, dx, pos, amount, radius);
        return;
    }
#endif
    
    glm::vec3 gridPos = worldToGrid(worldPos);
    int r = static_cast<int>(std::ceil(radius / dx));
    
    int ci = static_cast<int>(gridPos.x);
    int cj = static_cast<int>(gridPos.y);
    int ck = static_cast<int>(gridPos.z);
    
    for (int i = ci - r; i <= ci + r; i++) {
        for (int j = cj - r; j <= cj + r; j++) {
            for (int k = ck - r; k <= ck + r; k++) {
                if (i < 0 || i >= Nx || j < 0 || j >= Ny || k < 0 || k >= Nz) continue;
                
                glm::vec3 cellCenter = gridToWorld(i, j, k) + glm::vec3(dx * 0.5f);
                float dist = glm::length(cellCenter - worldPos);
                if (dist < radius) {
                    float weight = 1.0f - dist / radius;
                    density(i, j, k) += amount * weight * weight;
                }
            }
        }
    }
}

void SmokeSim::addTemperature(const glm::vec3& worldPos, float amount, float radius) {
    glm::vec3 gridPos = worldToGrid(worldPos);
    int r = static_cast<int>(std::ceil(radius / dx));
    
    int ci = static_cast<int>(gridPos.x);
    int cj = static_cast<int>(gridPos.y);
    int ck = static_cast<int>(gridPos.z);
    
    for (int i = ci - r; i <= ci + r; i++) {
        for (int j = cj - r; j <= cj + r; j++) {
            for (int k = ck - r; k <= ck + r; k++) {
                if (i < 0 || i >= Nx || j < 0 || j >= Ny || k < 0 || k >= Nz) continue;
                
                glm::vec3 cellCenter = gridToWorld(i, j, k) + glm::vec3(dx * 0.5f);
                float dist = glm::length(cellCenter - worldPos);
                if (dist < radius) {
                    float weight = 1.0f - dist / radius;
                    temperature(i, j, k) += amount * weight * weight;
                }
            }
        }
    }
}

void SmokeSim::addVelocity(const glm::vec3& worldPos, const glm::vec3& vel, float radius) {
    glm::vec3 gridPos = worldToGrid(worldPos);
    int r = static_cast<int>(std::ceil(radius / dx)) + 1;
    
    int ci = static_cast<int>(gridPos.x);
    int cj = static_cast<int>(gridPos.y);
    int ck = static_cast<int>(gridPos.z);
    
    // Add to u-component
    for (int i = ci - r; i <= ci + r + 1; i++) {
        for (int j = cj - r; j <= cj + r; j++) {
            for (int k = ck - r; k <= ck + r; k++) {
                if (i < 0 || i > Nx || j < 0 || j >= Ny || k < 0 || k >= Nz) continue;
                
                glm::vec3 facePos = gridToWorld(i, j, k) + glm::vec3(0, dx * 0.5f, dx * 0.5f);
                float dist = glm::length(facePos - worldPos);
                if (dist < radius) {
                    float weight = 1.0f - dist / radius;
                    u(i, j, k) += vel.x * weight;
                }
            }
        }
    }
    
    // Add to v-component
    for (int i = ci - r; i <= ci + r; i++) {
        for (int j = cj - r; j <= cj + r + 1; j++) {
            for (int k = ck - r; k <= ck + r; k++) {
                if (i < 0 || i >= Nx || j < 0 || j > Ny || k < 0 || k >= Nz) continue;
                
                glm::vec3 facePos = gridToWorld(i, j, k) + glm::vec3(dx * 0.5f, 0, dx * 0.5f);
                float dist = glm::length(facePos - worldPos);
                if (dist < radius) {
                    float weight = 1.0f - dist / radius;
                    v(i, j, k) += vel.y * weight;
                }
            }
        }
    }
    
    // Add to w-component
    for (int i = ci - r; i <= ci + r; i++) {
        for (int j = cj - r; j <= cj + r; j++) {
            for (int k = ck - r; k <= ck + r + 1; k++) {
                if (i < 0 || i >= Nx || j < 0 || j >= Ny || k < 0 || k > Nz) continue;
                
                glm::vec3 facePos = gridToWorld(i, j, k) + glm::vec3(dx * 0.5f, dx * 0.5f, 0);
                float dist = glm::length(facePos - worldPos);
                if (dist < radius) {
                    float weight = 1.0f - dist / radius;
                    w(i, j, k) += vel.z * weight;
                }
            }
        }
    }
}

void SmokeSim::fillDensity(float value) {
    // Fill entire density grid with uniform value (for "fog chamber" effect)
    density.fill(value);
    
#ifdef USE_CUDA
    if (useGPU && gpuInitialized) {
        // Fill GPU density grid
        smokeCudaFillDensity(value);
    }
#endif
}

void SmokeSim::setLiquidMask(const std::vector<glm::vec3>& particlePositions, float particleRadius) {
    clearLiquidMask();
    
    for (const auto& pos : particlePositions) {
        glm::vec3 gridPos = worldToGrid(pos);
        int i = static_cast<int>(gridPos.x);
        int j = static_cast<int>(gridPos.y);
        int k = static_cast<int>(gridPos.z);
        
        if (i >= 0 && i < Nx && j >= 0 && j < Ny && k >= 0 && k < Nz) {
            liquidMask(i, j, k) = 1;
        }
    }
}

void SmokeSim::clearLiquidMask() {
    liquidMask.fill(0);
}

glm::vec3 SmokeSim::sampleVelocity(const glm::vec3& worldPos) const {
    glm::vec3 gridPos = worldToGrid(worldPos);
    
    return glm::vec3(
        sampleU(gridPos.x, gridPos.y - 0.5f, gridPos.z - 0.5f),
        sampleV(gridPos.x - 0.5f, gridPos.y, gridPos.z - 0.5f),
        sampleW(gridPos.x - 0.5f, gridPos.y - 0.5f, gridPos.z)
    );
}

float SmokeSim::sampleDensity(const glm::vec3& worldPos) const {
    glm::vec3 gridPos = worldToGrid(worldPos);
    return density.sample(gridPos.x - 0.5f, gridPos.y - 0.5f, gridPos.z - 0.5f);
}

float SmokeSim::sampleTemperature(const glm::vec3& worldPos) const {
    glm::vec3 gridPos = worldToGrid(worldPos);
    return temperature.sample(gridPos.x - 0.5f, gridPos.y - 0.5f, gridPos.z - 0.5f);
}

glm::vec3 SmokeSim::worldToGrid(const glm::vec3& worldPos) const {
    return (worldPos - worldMin) / dx;
}

glm::vec3 SmokeSim::gridToWorld(int i, int j, int k) const {
    return worldMin + glm::vec3(i, j, k) * dx;
}

void SmokeSim::addForces(float dt) {
    // Add buoyancy force
    for (int i = 0; i < Nx; i++) {
        for (int j = 1; j < Ny; j++) {
            for (int k = 0; k < Nz; k++) {
                // Average density and temperature to face
                float d = 0.5f * (density(i, j, k) + density(i, glm::max(j-1, 0), k));
                float t = 0.5f * (temperature(i, j, k) + temperature(i, glm::max(j-1, 0), k));
                
                // Buoyancy: hot air rises, dense smoke falls
                float buoyancy = (buoyancyBeta * (t - ambientTemperature) - buoyancyAlpha * d);
                v(i, j, k) += dt * buoyancy;
            }
        }
    }
}

void SmokeSim::advectVelocity(float dt) {
    // Semi-Lagrangian advection for u
    for (int i = 0; i <= Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            for (int k = 0; k < Nz; k++) {
                // Position of u sample in grid coordinates
                float x = static_cast<float>(i);
                float y = static_cast<float>(j) + 0.5f;
                float z = static_cast<float>(k) + 0.5f;
                
                // Get velocity at this point
                float velX = u(i, j, k);
                float velY = sampleV(x - 0.5f, y, z - 0.5f);
                float velZ = sampleW(x - 0.5f, y - 0.5f, z);
                
                // Backtrace
                float srcX = x - dt * velX / dx;
                float srcY = y - dt * velY / dx;
                float srcZ = z - dt * velZ / dx;
                
                // Sample old velocity
                uTemp(i, j, k) = sampleU(srcX, srcY - 0.5f, srcZ - 0.5f);
            }
        }
    }
    
    // Semi-Lagrangian advection for v
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j <= Ny; j++) {
            for (int k = 0; k < Nz; k++) {
                float x = static_cast<float>(i) + 0.5f;
                float y = static_cast<float>(j);
                float z = static_cast<float>(k) + 0.5f;
                
                float velX = sampleU(x, y - 0.5f, z - 0.5f);
                float velY = v(i, j, k);
                float velZ = sampleW(x - 0.5f, y - 0.5f, z);
                
                float srcX = x - dt * velX / dx;
                float srcY = y - dt * velY / dx;
                float srcZ = z - dt * velZ / dx;
                
                vTemp(i, j, k) = sampleV(srcX - 0.5f, srcY, srcZ - 0.5f);
            }
        }
    }
    
    // Semi-Lagrangian advection for w
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            for (int k = 0; k <= Nz; k++) {
                float x = static_cast<float>(i) + 0.5f;
                float y = static_cast<float>(j) + 0.5f;
                float z = static_cast<float>(k);
                
                float velX = sampleU(x, y - 0.5f, z - 0.5f);
                float velY = sampleV(x - 0.5f, y, z - 0.5f);
                float velZ = w(i, j, k);
                
                float srcX = x - dt * velX / dx;
                float srcY = y - dt * velY / dx;
                float srcZ = z - dt * velZ / dx;
                
                wTemp(i, j, k) = sampleW(srcX - 0.5f, srcY - 0.5f, srcZ);
            }
        }
    }
    
    // Copy back
    std::swap(u.data, uTemp.data);
    std::swap(v.data, vTemp.data);
    std::swap(w.data, wTemp.data);
}

void SmokeSim::diffuseVelocity(float dt) {
    // Simple explicit diffusion (for stability, keep coefficient small)
    float alpha = diffusion * dt / (dx * dx);
    
    if (alpha < 0.001f) return;  // Skip if negligible
    
    // This is a simplified version - proper implicit solve would be better
    // For now, just do a few Jacobi iterations
    for (int iter = 0; iter < 5; iter++) {
        for (int i = 1; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    uTemp(i, j, k) = (u(i, j, k) + alpha * (
                        u(i-1, j, k) + u(i+1, j, k) +
                        u(i, glm::max(j-1,0), k) + u(i, glm::min(j+1,Ny-1), k) +
                        u(i, j, glm::max(k-1,0)) + u(i, j, glm::min(k+1,Nz-1))
                    )) / (1.0f + 6.0f * alpha);
                }
            }
        }
        std::swap(u.data, uTemp.data);
    }
}

void SmokeSim::project(float dt) {
    float invDx = 1.0f / dx;
    
    // Compute divergence
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            for (int k = 0; k < Nz; k++) {
                // Skip liquid cells in pressure solve
                if (liquidMask(i, j, k)) {
                    divergence(i, j, k) = 0.0f;
                    continue;
                }
                
                float div = invDx * (
                    u(i + 1, j, k) - u(i, j, k) +
                    v(i, j + 1, k) - v(i, j, k) +
                    w(i, j, k + 1) - w(i, j, k)
                );
                divergence(i, j, k) = div;
            }
        }
    }
    
    // Solve pressure Poisson equation using Jacobi iteration
    pressure.fill(0.0f);
    
    for (int iter = 0; iter < pressureIterations; iter++) {
        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                for (int k = 0; k < Nz; k++) {
                    if (liquidMask(i, j, k)) continue;
                    
                    float pL = (i > 0) ? pressure(i - 1, j, k) : pressure(i, j, k);
                    float pR = (i < Nx - 1) ? pressure(i + 1, j, k) : pressure(i, j, k);
                    float pD = (j > 0) ? pressure(i, j - 1, k) : pressure(i, j, k);
                    float pU = (j < Ny - 1) ? pressure(i, j + 1, k) : pressure(i, j, k);
                    float pB = (k > 0) ? pressure(i, j, k - 1) : pressure(i, j, k);
                    float pF = (k < Nz - 1) ? pressure(i, j, k + 1) : pressure(i, j, k);
                    
                    pressure(i, j, k) = (pL + pR + pD + pU + pB + pF - 
                                         dx * dx * divergence(i, j, k)) / 6.0f;
                }
            }
        }
    }
    
    // Subtract pressure gradient from velocity
    for (int i = 1; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            for (int k = 0; k < Nz; k++) {
                u(i, j, k) -= invDx * (pressure(i, j, k) - pressure(i - 1, j, k));
            }
        }
    }
    
    for (int i = 0; i < Nx; i++) {
        for (int j = 1; j < Ny; j++) {
            for (int k = 0; k < Nz; k++) {
                v(i, j, k) -= invDx * (pressure(i, j, k) - pressure(i, j - 1, k));
            }
        }
    }
    
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            for (int k = 1; k < Nz; k++) {
                w(i, j, k) -= invDx * (pressure(i, j, k) - pressure(i, j, k - 1));
            }
        }
    }
}

void SmokeSim::advectScalars(float dt) {
    // Advect density
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            for (int k = 0; k < Nz; k++) {
                float x = static_cast<float>(i) + 0.5f;
                float y = static_cast<float>(j) + 0.5f;
                float z = static_cast<float>(k) + 0.5f;
                
                float velX = 0.5f * (u(i, j, k) + u(i + 1, j, k));
                float velY = 0.5f * (v(i, j, k) + v(i, j + 1, k));
                float velZ = 0.5f * (w(i, j, k) + w(i, j, k + 1));
                
                float srcX = x - dt * velX / dx;
                float srcY = y - dt * velY / dx;
                float srcZ = z - dt * velZ / dx;
                
                densityTemp(i, j, k) = density.sample(srcX - 0.5f, srcY - 0.5f, srcZ - 0.5f) * dissipation;
                temperatureTemp(i, j, k) = glm::mix(ambientTemperature,
                    temperature.sample(srcX - 0.5f, srcY - 0.5f, srcZ - 0.5f), tempDissipation);
            }
        }
    }
    
    std::swap(density.data, densityTemp.data);
    std::swap(temperature.data, temperatureTemp.data);
}

void SmokeSim::setBoundaryConditions() {
    // Zero normal velocity at boundaries
    for (int j = 0; j < Ny; j++) {
        for (int k = 0; k < Nz; k++) {
            u(0, j, k) = 0.0f;
            u(Nx, j, k) = 0.0f;
        }
    }
    
    for (int i = 0; i < Nx; i++) {
        for (int k = 0; k < Nz; k++) {
            v(i, 0, k) = 0.0f;
            v(i, Ny, k) = 0.0f;
        }
    }
    
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            w(i, j, 0) = 0.0f;
            w(i, j, Nz) = 0.0f;
        }
    }
}

float SmokeSim::sampleU(float x, float y, float z) const {
    x = glm::clamp(x, 0.0f, static_cast<float>(Nx));
    y = glm::clamp(y, 0.0f, static_cast<float>(Ny - 1));
    z = glm::clamp(z, 0.0f, static_cast<float>(Nz - 1));
    return u.sample(x, y, z);
}

float SmokeSim::sampleV(float x, float y, float z) const {
    x = glm::clamp(x, 0.0f, static_cast<float>(Nx - 1));
    y = glm::clamp(y, 0.0f, static_cast<float>(Ny));
    z = glm::clamp(z, 0.0f, static_cast<float>(Nz - 1));
    return v.sample(x, y, z);
}

float SmokeSim::sampleW(float x, float y, float z) const {
    x = glm::clamp(x, 0.0f, static_cast<float>(Nx - 1));
    y = glm::clamp(y, 0.0f, static_cast<float>(Ny - 1));
    z = glm::clamp(z, 0.0f, static_cast<float>(Nz));
    return w.sample(x, y, z);
}

float SmokeSim::getMaxSpeed() const {
    float maxSpeed = 0.0f;
    
    // Sample velocity at cell centers and find maximum
    for (int k = 0; k < Nz; k++) {
        for (int j = 0; j < Ny; j++) {
            for (int i = 0; i < Nx; i++) {
                float ux = (u(i, j, k) + u(i + 1, j, k)) * 0.5f;
                float vy = (v(i, j, k) + v(i, j + 1, k)) * 0.5f;
                float wz = (w(i, j, k) + w(i, j, k + 1)) * 0.5f;
                float speed = std::sqrt(ux * ux + vy * vy + wz * wz);
                maxSpeed = std::max(maxSpeed, speed);
            }
        }
    }
    
    return maxSpeed;
}

