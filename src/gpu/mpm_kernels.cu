/**
 * CUDA kernels for MLS-MPM simulation
 * 
 * Optimized GPU port with PERSISTENT memory - allocate once, simulate many frames.
 * No per-frame cudaMalloc/cudaFree or full particle uploads/downloads.
 */

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <vector>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>

#include "mpm_cuda.h"
#include "../sim/MpmSim.h"

#define CUDA_CHECK(expr)                                             \
    do {                                                             \
        cudaError_t _err = (expr);                                   \
        if (_err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error %s at %s:%d\n",              \
                    cudaGetErrorString(_err), __FILE__, __LINE__);   \
        }                                                            \
    } while (0)

// Particle structure for GPU (SOA-friendly layout)
struct ParticleGPU {
    float3 x;       // Position
    float3 v;       // Velocity  
    float mass;
    float volume0;
    float F[9];     // Deformation gradient (row-major)
    float C[9];     // Affine momentum
    float E;
    float nu;
};

// Grid node structure
struct GridNodeGPU {
    float mass;
    float3 vel;     // Pre-force velocity (for FLIP)
    float3 velNew;  // Post-force velocity
};

// ============== PERSISTENT GPU STATE ==============
static ParticleGPU* d_particles = nullptr;
static GridNodeGPU* d_grid = nullptr;
static float3* d_positions = nullptr;  // For fast position-only download
static int s_maxParticles = 0;
static int s_currentParticles = 0;
static size_t s_gridSize = 0;
static cudaStream_t s_stream = nullptr;

// Host staging buffers (reused across frames)
static std::vector<ParticleGPU> h_particles;
static std::vector<float3> h_positions;

// Debug flag
static bool s_firstStep = true;

void mpmCudaInit(int maxParticles, int Nx, int Ny, int Nz) {
    // Clean up any existing allocations
    mpmCudaCleanup();
    
    s_maxParticles = maxParticles;
    s_gridSize = static_cast<size_t>(Nx) * Ny * Nz;
    
    // Allocate persistent device buffers
    CUDA_CHECK(cudaMalloc(&d_particles, maxParticles * sizeof(ParticleGPU)));
    CUDA_CHECK(cudaMalloc(&d_grid, s_gridSize * sizeof(GridNodeGPU)));
    CUDA_CHECK(cudaMalloc(&d_positions, maxParticles * sizeof(float3)));
    
    // Create stream for async operations
    CUDA_CHECK(cudaStreamCreate(&s_stream));
    
    // Pre-allocate host staging buffers
    h_particles.resize(maxParticles);
    h_positions.resize(maxParticles);
    
    printf("[CUDA] Initialized: %d max particles, %zu grid nodes\n", maxParticles, s_gridSize);
}

void mpmCudaCleanup() {
    if (d_particles) { cudaFree(d_particles); d_particles = nullptr; }
    if (d_grid) { cudaFree(d_grid); d_grid = nullptr; }
    if (d_positions) { cudaFree(d_positions); d_positions = nullptr; }
    if (s_stream) { cudaStreamDestroy(s_stream); s_stream = nullptr; }
    s_maxParticles = 0;
    s_currentParticles = 0;
    s_gridSize = 0;
    h_particles.clear();
    h_positions.clear();
}

void mpmCudaUploadParticles(const std::vector<Particle>& particles) {
    int numParticles = static_cast<int>(particles.size());
    
    if (numParticles == 0) {
        fprintf(stderr, "[CUDA] Warning: no particles to upload\n");
        return;
    }
    
    if (numParticles > s_maxParticles) {
        fprintf(stderr, "[CUDA] Warning: particle count %d exceeds max %d\n", 
                numParticles, s_maxParticles);
        return;
    }
    
    if (!d_particles) {
        fprintf(stderr, "[CUDA] Error: d_particles not allocated! Call mpmCudaInit first.\n");
        return;
    }
    
    s_currentParticles = numParticles;
    
    // Convert to GPU format
    for (int i = 0; i < numParticles; i++) {
        const Particle& p = particles[i];
        ParticleGPU& gp = h_particles[i];
        gp.x = make_float3(p.x.x, p.x.y, p.x.z);
        gp.v = make_float3(p.v.x, p.v.y, p.v.z);
        gp.mass = p.mass;
        gp.volume0 = p.volume0;
        gp.E = p.E;
        gp.nu = p.nu;
        
        gp.F[0]=p.F[0][0]; gp.F[1]=p.F[0][1]; gp.F[2]=p.F[0][2];
        gp.F[3]=p.F[1][0]; gp.F[4]=p.F[1][1]; gp.F[5]=p.F[1][2];
        gp.F[6]=p.F[2][0]; gp.F[7]=p.F[2][1]; gp.F[8]=p.F[2][2];
        
        gp.C[0]=p.C[0][0]; gp.C[1]=p.C[0][1]; gp.C[2]=p.C[0][2];
        gp.C[3]=p.C[1][0]; gp.C[4]=p.C[1][1]; gp.C[5]=p.C[1][2];
        gp.C[6]=p.C[2][0]; gp.C[7]=p.C[2][1]; gp.C[8]=p.C[2][2];
    }
    
    // Upload to GPU
    CUDA_CHECK(cudaMemcpyAsync(d_particles, h_particles.data(), 
                               numParticles * sizeof(ParticleGPU), 
                               cudaMemcpyHostToDevice, s_stream));
    CUDA_CHECK(cudaStreamSynchronize(s_stream));
    
    s_firstStep = true;
}

int mpmCudaGetParticleCount() {
    return s_currentParticles;
}

void* mpmCudaGetParticleBuffer() {
    return d_particles;
}

// ---------------- Helper math ----------------
__device__ __forceinline__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 operator*(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ __forceinline__ float3 operator*(float s, const float3& a) {
    return a * s;
}

__device__ __forceinline__ void atomicAddFloat3(float3* dst, const float3& val) {
    atomicAdd(&dst->x, val.x);
    atomicAdd(&dst->y, val.y);
    atomicAdd(&dst->z, val.z);
}

__device__ __forceinline__ int gridIndex(int i, int j, int k, int Nx, int Ny) {
    return i + Nx * (j + Ny * k);
}

__device__ __forceinline__ float det3(const float* m) {
    return m[0]*(m[4]*m[8]-m[5]*m[7]) - m[1]*(m[3]*m[8]-m[5]*m[6]) + m[2]*(m[3]*m[7]-m[4]*m[6]);
}

__device__ __forceinline__ void mat3Identity(float* m) {
    m[0]=1; m[1]=0; m[2]=0;
    m[3]=0; m[4]=1; m[5]=0;
    m[6]=0; m[7]=0; m[8]=1;
}

__device__ __forceinline__ void mat3Mul(const float* A, const float* B, float* out) {
    out[0] = A[0]*B[0] + A[1]*B[3] + A[2]*B[6];
    out[1] = A[0]*B[1] + A[1]*B[4] + A[2]*B[7];
    out[2] = A[0]*B[2] + A[1]*B[5] + A[2]*B[8];
    out[3] = A[3]*B[0] + A[4]*B[3] + A[5]*B[6];
    out[4] = A[3]*B[1] + A[4]*B[4] + A[5]*B[7];
    out[5] = A[3]*B[2] + A[4]*B[5] + A[5]*B[8];
    out[6] = A[6]*B[0] + A[7]*B[3] + A[8]*B[6];
    out[7] = A[6]*B[1] + A[7]*B[4] + A[8]*B[7];
    out[8] = A[6]*B[2] + A[7]*B[5] + A[8]*B[8];
}

__device__ __forceinline__ float3 mulMat3Vec(const float* A, const float3& v) {
    return make_float3(
        A[0]*v.x + A[1]*v.y + A[2]*v.z,
        A[3]*v.x + A[4]*v.y + A[5]*v.z,
        A[6]*v.x + A[7]*v.y + A[8]*v.z
    );
}

__device__ __forceinline__ void outerProduct(const float3& v, const float3& w, float* out) {
    out[0] = v.x * w.x; out[1] = v.x * w.y; out[2] = v.x * w.z;
    out[3] = v.y * w.x; out[4] = v.y * w.y; out[5] = v.y * w.z;
    out[6] = v.z * w.x; out[7] = v.z * w.y; out[8] = v.z * w.z;
}

// ---------------- Kernels ----------------
__global__ void clearGridKernel(GridNodeGPU* grid, int numNodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numNodes) return;
    
    grid[idx].mass = 0.0f;
    grid[idx].vel = make_float3(0.0f, 0.0f, 0.0f);
    grid[idx].velNew = make_float3(0.0f, 0.0f, 0.0f);
}

__global__ void p2gKernel(
    const ParticleGPU* particles, int numParticles,
    GridNodeGPU* grid, int Nx, int Ny, int Nz,
    float dx, float dt, float apicBlend,
    float3 worldMin
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    const ParticleGPU& p = particles[idx];
    float invDx = 1.0f / dx;

    // Compute relative position (CUDA float3 needs component-wise math)
    float3 rel;
    rel.x = (p.x.x - worldMin.x) * invDx;
    rel.y = (p.x.y - worldMin.y) * invDx;
    rel.z = (p.x.z - worldMin.z) * invDx;
    
    int baseX = static_cast<int>(floorf(rel.x - 0.5f));
    int baseY = static_cast<int>(floorf(rel.y - 0.5f));
    int baseZ = static_cast<int>(floorf(rel.z - 0.5f));

    float3 fx = make_float3(rel.x - baseX, rel.y - baseY, rel.z - baseZ);

    // Quadratic B-spline weights
    float3 w[3];
    w[0] = make_float3(0.5f * (1.5f - fx.x)*(1.5f - fx.x),
                       0.5f * (1.5f - fx.y)*(1.5f - fx.y),
                       0.5f * (1.5f - fx.z)*(1.5f - fx.z));
    w[1] = make_float3(0.75f - (fx.x - 1.0f)*(fx.x - 1.0f),
                       0.75f - (fx.y - 1.0f)*(fx.y - 1.0f),
                       0.75f - (fx.z - 1.0f)*(fx.z - 1.0f));
    w[2] = make_float3(0.5f * (fx.x - 0.5f)*(fx.x - 0.5f),
                       0.5f * (fx.y - 0.5f)*(fx.y - 0.5f),
                       0.5f * (fx.z - 0.5f)*(fx.z - 0.5f));

    // Compute stress for FLUID (MLS-MPM, Hu et al. 2018)
    // Equation of state: p = K * (1 - J) where J = det(F)
    // When compressed (J < 1): p > 0 (positive pressure pushes outward)
    // When expanded (J > 1): p < 0 (negative pressure pulls inward)
    // Cauchy stress: σ = -p * I
    
    float J = det3(p.F);
    J = fmaxf(J, 0.1f);   // Prevent singularity
    J = fminf(J, 2.0f);   // Prevent extreme expansion
    
    // Bulk modulus from particle (set from g_params.bulkModulus)
    float K = p.E;
    
    // Tait-style equation of state (common for water in SPH/MPM)
    // p = K * (1 - J) is linear approximation
    // For more incompressible: p = K * ((1/J)^gamma - 1), gamma=7 for water
    float pressure = K * (1.0f - J);
    
    // Clamp pressure to prevent instability
    const float MAX_PRESSURE = 10000.0f;
    pressure = fmaxf(-MAX_PRESSURE, fminf(MAX_PRESSURE, pressure));
    
    // Cauchy stress: σ = -p * I (isotropic, no shear for fluid)
    float stress[9] = {0};
    stress[0] = -pressure; 
    stress[4] = -pressure; 
    stress[8] = -pressure;

    float coeff = -dt * 4.0f * invDx * invDx * p.volume0;
    float affine[9];
    for (int i = 0; i < 9; i++) {
        affine[i] = coeff * stress[i] + p.mass * p.C[i] * apicBlend;
    }

    // Scatter to 3x3x3 neighborhood
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                int gi = baseX + i;
                int gj = baseY + j;
                int gk = baseZ + k;
                if (gi < 0 || gi >= Nx || gj < 0 || gj >= Ny || gk < 0 || gk >= Nz) continue;

                float weight = w[i].x * w[j].y * w[k].z;
                float3 dpos = make_float3((float)i - fx.x, (float)j - fx.y, (float)k - fx.z) * dx;

                float3 momentum = p.v * p.mass;
                float3 affineDP = mulMat3Vec(affine, dpos);
                float3 contrib = (momentum + affineDP) * weight;

                int gidx = gridIndex(gi, gj, gk, Nx, Ny);
                atomicAdd(&grid[gidx].mass, weight * p.mass);
                atomicAddFloat3(&grid[gidx].vel, contrib);
            }
        }
    }
}

__global__ void gridUpdateKernel(
    GridNodeGPU* grid, int Nx, int Ny, int Nz,
    float dx, float dt, float gravity,
    float3 worldMin, float3 worldMax
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * Nz;
    if (idx >= total) return;

    GridNodeGPU& node = grid[idx];
    if (node.mass <= 1e-6f) {
        node.vel = make_float3(0,0,0);
        node.velNew = make_float3(0,0,0);
        return;
    }

    // Normalize momentum to velocity
    node.vel.x /= node.mass;
    node.vel.y /= node.mass;
    node.vel.z /= node.mass;

    float3 velOld = node.vel;

    // Apply gravity
    node.vel.y += dt * gravity;

    // Boundary conditions
    int k = idx / (Nx * Ny);
    int rem = idx - k * (Nx * Ny);
    int j = rem / Nx;
    int i = rem - j * Nx;

    float boundaryDist = 3.0f * dx;
    float x = worldMin.x + i * dx;
    float y = worldMin.y + j * dx;
    float z = worldMin.z + k * dx;
    
    // Floor boundary (Y min)
    if (y < worldMin.y + boundaryDist) {
        if (node.vel.y < 0.0f) node.vel.y = 0.0f;
    }
    // Ceiling boundary (Y max)
    if (y > worldMax.y - boundaryDist) {
        if (node.vel.y > 0.0f) node.vel.y = 0.0f;
    }
    // Wall boundaries (X)
    if (x < worldMin.x + boundaryDist) {
        if (node.vel.x < 0.0f) node.vel.x = 0.0f;
    }
    if (x > worldMax.x - boundaryDist) {
        if (node.vel.x > 0.0f) node.vel.x = 0.0f;
    }
    // Wall boundaries (Z)
    if (z < worldMin.z + boundaryDist) {
        if (node.vel.z < 0.0f) node.vel.z = 0.0f;
    }
    if (z > worldMax.z - boundaryDist) {
        if (node.vel.z > 0.0f) node.vel.z = 0.0f;
    }

    // Velocity clamp for stability
    const float MAX_GRID_SPEED = 8.0f;
    float speed = sqrtf(node.vel.x*node.vel.x + node.vel.y*node.vel.y + node.vel.z*node.vel.z);
    if (speed > MAX_GRID_SPEED) {
        float scale = MAX_GRID_SPEED / speed;
        node.vel.x *= scale;
        node.vel.y *= scale;
        node.vel.z *= scale;
    }

    node.velNew = node.vel;
    node.vel = velOld;  // Store old for FLIP
}


__global__ void g2pKernel(
    ParticleGPU* particles, int numParticles,
    const GridNodeGPU* grid, int Nx, int Ny, int Nz,
    float dx, float dt, float flipRatio,
    float3 worldMin, float3 worldMax
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    ParticleGPU& p = particles[idx];
    float invDx = 1.0f / dx;

    // Compute relative position (component-wise for CUDA)
    float3 rel;
    rel.x = (p.x.x - worldMin.x) * invDx;
    rel.y = (p.x.y - worldMin.y) * invDx;
    rel.z = (p.x.z - worldMin.z) * invDx;
    
    int baseX = static_cast<int>(floorf(rel.x - 0.5f));
    int baseY = static_cast<int>(floorf(rel.y - 0.5f));
    int baseZ = static_cast<int>(floorf(rel.z - 0.5f));
    float3 fx = make_float3(rel.x - baseX, rel.y - baseY, rel.z - baseZ);

    float3 w[3];
    w[0] = make_float3(0.5f * (1.5f - fx.x)*(1.5f - fx.x),
                       0.5f * (1.5f - fx.y)*(1.5f - fx.y),
                       0.5f * (1.5f - fx.z)*(1.5f - fx.z));
    w[1] = make_float3(0.75f - (fx.x - 1.0f)*(fx.x - 1.0f),
                       0.75f - (fx.y - 1.0f)*(fx.y - 1.0f),
                       0.75f - (fx.z - 1.0f)*(fx.z - 1.0f));
    w[2] = make_float3(0.5f * (fx.x - 0.5f)*(fx.x - 0.5f),
                       0.5f * (fx.y - 0.5f)*(fx.y - 0.5f),
                       0.5f * (fx.z - 0.5f)*(fx.z - 0.5f));

    float3 velPIC = make_float3(0,0,0);
    float3 velOld = make_float3(0,0,0);
    float B[9] = {0};

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                int gi = baseX + i;
                int gj = baseY + j;
                int gk = baseZ + k;
                if (gi < 0 || gi >= Nx || gj < 0 || gj >= Ny || gk < 0 || gk >= Nz) continue;
                
                float weight = w[i].x * w[j].y * w[k].z;
                float3 dpos = make_float3((float)i - fx.x, (float)j - fx.y, (float)k - fx.z) * dx;

                int gidx = gridIndex(gi, gj, gk, Nx, Ny);
                const GridNodeGPU& node = grid[gidx];

                velPIC = velPIC + node.velNew * weight;
                velOld = velOld + node.vel * weight;

                float outer[9];
                outerProduct(node.velNew, dpos, outer);
                for (int n = 0; n < 9; n++) B[n] += weight * outer[n];
            }
        }
    }

    // APIC C update
    float scale = 4.0f * invDx * invDx;
    for (int i = 0; i < 9; i++) {
        p.C[i] = B[i] * scale;
    }

    // FLIP/PIC blend
    float3 dv_flip = velPIC - velOld;
    float3 velFLIP = p.v + dv_flip;
    p.v = velFLIP * flipRatio + velPIC * (1.0f - flipRatio);

    // ============== HARD VELOCITY CLAMP (CFL SAFETY) ==============
    const float MAX_PARTICLE_SPEED = 5.0f;  // m/s - conservative limit
    
    // NaN/Inf check - reset to safe values if corrupted
    if (isnan(p.v.x) || isnan(p.v.y) || isnan(p.v.z) ||
        isinf(p.v.x) || isinf(p.v.y) || isinf(p.v.z)) {
        p.v = make_float3(0.0f, 0.0f, 0.0f);
    }
    
    float speed = sqrtf(p.v.x*p.v.x + p.v.y*p.v.y + p.v.z*p.v.z);
    if (speed > MAX_PARTICLE_SPEED) {
        float scale = MAX_PARTICLE_SPEED / speed;
        p.v.x *= scale;
        p.v.y *= scale;
        p.v.z *= scale;
    }

    // Update position (component-wise for CUDA)
    p.x.x = p.x.x + p.v.x * dt;
    p.x.y = p.x.y + p.v.y * dt;
    p.x.z = p.x.z + p.v.z * dt;

    // Update deformation gradient
    float Iplus[9];
    mat3Identity(Iplus);
    for (int i = 0; i < 9; i++) Iplus[i] += dt * p.C[i];

    float newF[9];
    mat3Mul(Iplus, p.F, newF);

    // For fluid: compute J (volume ratio), then reset F to isotropic
    // This removes shear (jelly) but KEEPS volume for pressure computation
    float J = det3(newF);
    J = fminf(fmaxf(J, 0.8f), 1.2f);  // Clamp for stability
    
    // Reset F to cbrt(J) * I - isotropic, no shear, but preserves volume ratio
    float cbrtJ = cbrtf(J);
    for (int i = 0; i < 9; i++) p.F[i] = 0.0f;
    p.F[0] = cbrtJ; p.F[4] = cbrtJ; p.F[8] = cbrtJ;

    // ============== CORRECT MPM BOUNDARY HANDLING ==============
    // Rule: NEVER modify particle positions for collisions!
    // Grid velocities are already clamped in gridUpdateKernel.
    // We only do a LAST-RESORT safety clamp here to prevent particles
    // from escaping the domain entirely (which would crash the sim).
    
    float safetyMargin = 1.5f * dx;  // Minimal safety margin
    
    // Safety clamp ONLY - no velocity modification here
    // (velocity was already handled correctly at grid level)
    p.x.x = fmaxf(worldMin.x + safetyMargin, fminf(worldMax.x - safetyMargin, p.x.x));
    p.x.y = fmaxf(worldMin.y + safetyMargin, fminf(worldMax.y - safetyMargin, p.x.y));
    p.x.z = fmaxf(worldMin.z + safetyMargin, fminf(worldMax.z - safetyMargin, p.x.z));
}

// Kernel to extract just positions for fast download
__global__ void extractPositionsKernel(const ParticleGPU* particles, float3* positions, int numParticles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;
    positions[idx] = particles[idx].x;
}

// ============== CFL-ADAPTIVE TIME STEPPING ==============
// Kernel to find maximum particle speed (for CFL calculation)
__global__ void maxSpeedKernel(const ParticleGPU* particles, float* maxSpeed, int numParticles) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread computes its local max
    float localMax = 0.0f;
    if (idx < numParticles) {
        float3 v = particles[idx].v;
        localMax = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
    }
    sdata[tid] = localMax;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    // Block max written by thread 0
    if (tid == 0) {
        atomicMax((int*)maxSpeed, __float_as_int(sdata[0]));
    }
}

// Device memory for max speed
static float* d_maxSpeed = nullptr;

// Compute CFL-safe timestep based on max particle velocity
float mpmCudaComputeSafeDt(int numParticles, float dx, float bulkModulus, float density) {
    if (numParticles == 0 || !d_particles) return 0.001f;  // Default
    
    // Allocate max speed buffer if needed
    if (!d_maxSpeed) {
        cudaMalloc(&d_maxSpeed, sizeof(float));
    }
    
    // Reset max speed to 0
    float zero = 0.0f;
    cudaMemcpy(d_maxSpeed, &zero, sizeof(float), cudaMemcpyHostToDevice);
    
    // Find max speed
    int threads = 256;
    int blocks = (numParticles + threads - 1) / threads;
    maxSpeedKernel<<<blocks, threads>>>(d_particles, d_maxSpeed, numParticles);
    cudaDeviceSynchronize();
    
    // Download result
    float maxSpeed;
    cudaMemcpy(&maxSpeed, d_maxSpeed, sizeof(float), cudaMemcpyDeviceToHost);
    
    // Speed of sound in the material (for pressure waves)
    float soundSpeed = sqrtf(bulkModulus / density);
    
    // CFL condition: dt < CFL_factor * dx / (max_vel + sound_speed)
    const float CFL_FACTOR = 0.4f;  // Conservative safety factor
    float safeDt = CFL_FACTOR * dx / (maxSpeed + soundSpeed + 1e-6f);
    
    // Clamp to reasonable range
    safeDt = fmaxf(0.00001f, fminf(0.002f, safeDt));
    
    return safeDt;
}

// ---------------- Host API ----------------
void mpmCudaStep(
    int numParticles,
    int Nx, int Ny, int Nz,
    float dx, float dt,
    float apicBlend,
    float flipRatio,
    float gravity,
    const glm::vec3& worldMin,
    const glm::vec3& worldMax) {

    if (numParticles == 0 || !d_particles || !d_grid) return;

    int gridTotal = Nx * Ny * Nz;
    int threads = 256;
    int blocksGrid = (gridTotal + threads - 1) / threads;
    int blocksP = (numParticles + threads - 1) / threads;

    float3 wMin = make_float3(worldMin.x, worldMin.y, worldMin.z);
    float3 wMax = make_float3(worldMax.x, worldMax.y, worldMax.z);

    if (s_firstStep) {
        printf("[CUDA] mpmCudaStep: %d particles, grid %dx%dx%d, dt=%.6f, gravity=%.2f\n",
               numParticles, Nx, Ny, Nz, dt, gravity);
    }

    // All kernels on the same stream - no sync between them for max throughput
    clearGridKernel<<<blocksGrid, threads, 0, s_stream>>>(d_grid, gridTotal);

    p2gKernel<<<blocksP, threads, 0, s_stream>>>(
        d_particles, numParticles, d_grid, Nx, Ny, Nz, dx, dt, apicBlend, wMin);

    gridUpdateKernel<<<blocksGrid, threads, 0, s_stream>>>(
        d_grid, Nx, Ny, Nz, dx, dt, gravity, wMin, wMax);

    g2pKernel<<<blocksP, threads, 0, s_stream>>>(
        d_particles, numParticles, d_grid, Nx, Ny, Nz, dx, dt, flipRatio, wMin, wMax);

    s_firstStep = false;
}

// ============== GPU-RESIDENT SIMULATION ==============
static ParticleGPU* d_particlesBuf = nullptr;
static GridNodeGPU* d_gridBuf = nullptr;
static float3* d_positionsBuf = nullptr;  // For position-only download
static std::vector<ParticleGPU> h_particlesBuf;
static std::vector<float3> h_positionsBuf;
static int s_allocatedParticles = 0;
static int s_allocatedGrid = 0;
static int s_gpuParticleCount = 0;
static bool s_needsUpload = true;  // Flag to track if we need to upload

// Kernel to extract just positions
__global__ void extractPositionsKernel2(const ParticleGPU* particles, float3* positions, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) positions[i] = particles[i].x;
}

// Call this when particles change (reset/init)
void mpmCudaMarkDirty() {
    s_needsUpload = true;
}

void mpmCudaStepSimple(
    std::vector<Particle>& particles,
    int Nx, int Ny, int Nz,
    float dx, float dt,
    float apicBlend,
    float flipRatio,
    float gravity,
    const glm::vec3& worldMin,
    const glm::vec3& worldMax) {

    int numParticles = static_cast<int>(particles.size());
    if (numParticles == 0) return;
    
    // Debug: print once on first call
    static bool firstCall = true;
    if (firstCall) {
        printf("[CUDA] mpmCudaStepSimple called: %d particles, dt=%.6f, gravity=%.2f\n", 
               numParticles, dt, gravity);
        printf("[CUDA] Grid: %dx%dx%d, dx=%.4f\n", Nx, Ny, Nz, dx);
        printf("[CUDA] World: (%.2f,%.2f,%.2f) to (%.2f,%.2f,%.2f)\n",
               worldMin.x, worldMin.y, worldMin.z, worldMax.x, worldMax.y, worldMax.z);
        firstCall = false;
    }

    int gridSize = Nx * Ny * Nz;

    // Reallocate buffers only if size increased
    if (numParticles > s_allocatedParticles) {
        if (d_particlesBuf) cudaFree(d_particlesBuf);
        if (d_positionsBuf) cudaFree(d_positionsBuf);
        cudaMalloc(&d_particlesBuf, numParticles * sizeof(ParticleGPU));
        cudaMalloc(&d_positionsBuf, numParticles * sizeof(float3));
        h_particlesBuf.resize(numParticles);
        h_positionsBuf.resize(numParticles);
        s_allocatedParticles = numParticles;
        s_needsUpload = true;
    }
    if (gridSize > s_allocatedGrid) {
        if (d_gridBuf) cudaFree(d_gridBuf);
        cudaMalloc(&d_gridBuf, gridSize * sizeof(GridNodeGPU));
        s_allocatedGrid = gridSize;
    }

    // Upload only when particles changed (reset/init)
    if (s_needsUpload || numParticles != s_gpuParticleCount) {
        for (int i = 0; i < numParticles; i++) {
            const Particle& p = particles[i];
            h_particlesBuf[i].x = make_float3(p.x.x, p.x.y, p.x.z);
            h_particlesBuf[i].v = make_float3(p.v.x, p.v.y, p.v.z);
            h_particlesBuf[i].mass = p.mass;
            h_particlesBuf[i].volume0 = p.volume0;
            h_particlesBuf[i].E = p.E;
            h_particlesBuf[i].nu = p.nu;
            h_particlesBuf[i].F[0]=p.F[0][0]; h_particlesBuf[i].F[1]=p.F[0][1]; h_particlesBuf[i].F[2]=p.F[0][2];
            h_particlesBuf[i].F[3]=p.F[1][0]; h_particlesBuf[i].F[4]=p.F[1][1]; h_particlesBuf[i].F[5]=p.F[1][2];
            h_particlesBuf[i].F[6]=p.F[2][0]; h_particlesBuf[i].F[7]=p.F[2][1]; h_particlesBuf[i].F[8]=p.F[2][2];
            h_particlesBuf[i].C[0]=p.C[0][0]; h_particlesBuf[i].C[1]=p.C[0][1]; h_particlesBuf[i].C[2]=p.C[0][2];
            h_particlesBuf[i].C[3]=p.C[1][0]; h_particlesBuf[i].C[4]=p.C[1][1]; h_particlesBuf[i].C[5]=p.C[1][2];
            h_particlesBuf[i].C[6]=p.C[2][0]; h_particlesBuf[i].C[7]=p.C[2][1]; h_particlesBuf[i].C[8]=p.C[2][2];
        }
        cudaMemcpy(d_particlesBuf, h_particlesBuf.data(), numParticles * sizeof(ParticleGPU), cudaMemcpyHostToDevice);
        s_gpuParticleCount = numParticles;
        s_needsUpload = false;
    }

    int threads = 256;
    int blocksGrid = (gridSize + threads - 1) / threads;
    int blocksP = (numParticles + threads - 1) / threads;
    float3 wMin = make_float3(worldMin.x, worldMin.y, worldMin.z);
    float3 wMax = make_float3(worldMax.x, worldMax.y, worldMax.z);

    // Run simulation entirely on GPU
    clearGridKernel<<<blocksGrid, threads>>>(d_gridBuf, gridSize);
    cudaError_t err1 = cudaGetLastError();
    if (err1 != cudaSuccess) printf("clearGrid error: %s\n", cudaGetErrorString(err1));
    
    p2gKernel<<<blocksP, threads>>>(d_particlesBuf, numParticles, d_gridBuf, Nx, Ny, Nz, dx, dt, apicBlend, wMin);
    cudaError_t err2 = cudaGetLastError();
    if (err2 != cudaSuccess) printf("p2g error: %s\n", cudaGetErrorString(err2));
    
    gridUpdateKernel<<<blocksGrid, threads>>>(d_gridBuf, Nx, Ny, Nz, dx, dt, gravity, wMin, wMax);
    cudaError_t err3 = cudaGetLastError();
    if (err3 != cudaSuccess) printf("gridUpdate error: %s\n", cudaGetErrorString(err3));
    
    g2pKernel<<<blocksP, threads>>>(d_particlesBuf, numParticles, d_gridBuf, Nx, Ny, Nz, dx, dt, flipRatio, wMin, wMax);
    cudaError_t err4 = cudaGetLastError();
    if (err4 != cudaSuccess) printf("g2p error: %s\n", cudaGetErrorString(err4));
    
    // Sync and check for any errors
    cudaDeviceSynchronize();
    cudaError_t errFinal = cudaGetLastError();
    if (errFinal != cudaSuccess) printf("CUDA sync error: %s\n", cudaGetErrorString(errFinal));
}

// Download just positions for rendering (12 bytes per particle vs ~100 bytes)
void mpmCudaGetPositions(std::vector<glm::vec3>& outPositions) {
    if (s_gpuParticleCount == 0 || !d_particlesBuf || !d_positionsBuf) {
        return;
    }
    
    int threads = 256;
    int blocks = (s_gpuParticleCount + threads - 1) / threads;
    
    // Extract positions on GPU
    extractPositionsKernel2<<<blocks, threads>>>(d_particlesBuf, d_positionsBuf, s_gpuParticleCount);
    cudaDeviceSynchronize();
    
    // Download just positions (12 bytes per particle)
    cudaMemcpy(h_positionsBuf.data(), d_positionsBuf, s_gpuParticleCount * sizeof(float3), cudaMemcpyDeviceToHost);
    
    outPositions.resize(s_gpuParticleCount);
    for (int i = 0; i < s_gpuParticleCount; i++) {
        outPositions[i] = glm::vec3(h_positionsBuf[i].x, h_positionsBuf[i].y, h_positionsBuf[i].z);
    }
}

// ============== PERSISTENT BUFFER VERSION (FOR OPTIMIZATION LATER) ==============
void mpmCudaDownloadPositions(std::vector<glm::vec3>& positions, int numParticles) {
    if (numParticles == 0 || !d_particles || !d_positions) return;

    int threads = 256;
    int blocks = (numParticles + threads - 1) / threads;

    // Extract positions on GPU
    extractPositionsKernel<<<blocks, threads, 0, s_stream>>>(d_particles, d_positions, numParticles);

    // Now sync and download just positions
    CUDA_CHECK(cudaStreamSynchronize(s_stream));

    if (static_cast<int>(h_positions.size()) < numParticles) {
        h_positions.resize(numParticles);
    }

    CUDA_CHECK(cudaMemcpy(h_positions.data(), d_positions, 
                          numParticles * sizeof(float3), cudaMemcpyDeviceToHost));

    positions.resize(numParticles);
    for (int i = 0; i < numParticles; i++) {
        positions[i] = glm::vec3(h_positions[i].x, h_positions[i].y, h_positions[i].z);
    }
}

void mpmCudaDownloadParticles(std::vector<Particle>& particles) {
    // Use the SAME buffers as mpmCudaStepSimple!
    int numParticles = s_gpuParticleCount;
    if (numParticles == 0 || !d_particlesBuf) return;

    // Download from GPU
    cudaDeviceSynchronize();
    cudaMemcpy(h_particlesBuf.data(), d_particlesBuf, 
               numParticles * sizeof(ParticleGPU), cudaMemcpyDeviceToHost);

    particles.resize(numParticles);
    for (int i = 0; i < numParticles; i++) {
        Particle& p = particles[i];
        const ParticleGPU& gp = h_particlesBuf[i];
        p.x = glm::vec3(gp.x.x, gp.x.y, gp.x.z);
        p.v = glm::vec3(gp.v.x, gp.v.y, gp.v.z);
        p.mass = gp.mass;
        p.volume0 = gp.volume0;
        p.E = gp.E;
        p.nu = gp.nu;

        p.F[0][0]=gp.F[0]; p.F[0][1]=gp.F[1]; p.F[0][2]=gp.F[2];
        p.F[1][0]=gp.F[3]; p.F[1][1]=gp.F[4]; p.F[1][2]=gp.F[5];
        p.F[2][0]=gp.F[6]; p.F[2][1]=gp.F[7]; p.F[2][2]=gp.F[8];

        p.C[0][0]=gp.C[0]; p.C[0][1]=gp.C[1]; p.C[0][2]=gp.C[2];
        p.C[1][0]=gp.C[3]; p.C[1][1]=gp.C[4]; p.C[1][2]=gp.C[5];
        p.C[2][0]=gp.C[6]; p.C[2][1]=gp.C[7]; p.C[2][2]=gp.C[8];
    }
}

#endif // USE_CUDA
