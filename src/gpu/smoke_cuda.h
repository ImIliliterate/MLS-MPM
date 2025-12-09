#pragma once

/**
 * CUDA smoke simulation interface
 * 
 * Provides GPU-accelerated Stable Fluids solver
 */

#ifdef USE_CUDA

#include <cuda_runtime.h>  // For float3

extern "C" {

// Initialize GPU smoke buffers
void initSmokeCuda(int Nx, int Ny, int Nz);

// Cleanup GPU buffers
void cleanupSmokeCuda();

// Reset simulation to initial state
void resetSmokeCuda(float ambientTemp);

// Main simulation step
void smokeCudaStep(
    float dt, float dx,
    float buoyancyAlpha, float buoyancyBeta, float ambientTemp,
    float densityDissipation, float tempDissipation,
    int pressureIterations
);

// Add smoke density source
void smokeCudaAddDensity(float3 worldMin, float dx, float3 pos, float amount, float radius);

// Fill entire density grid with uniform value (fog chamber)
void smokeCudaFillDensity(float value);

// Download density to CPU for rendering
void smokeCudaDownloadDensity(float* hostDensity);

// Get GPU buffer pointers (for coupling without extra copies)
float* getSmokeCudaU();
float* getSmokeCudaV();
float* getSmokeCudaW();
float* getSmokeCudaDensity();
float* getSmokeCudaTemperature();

// Query state
bool isSmokeCudaInitialized();
int getSmokeCudaNx();
int getSmokeCudaNy();
int getSmokeCudaNz();

// ============== COUPLING ==============
// Apply drag from smoke to MPM particles (Grid→Particle)
// d_mpmParticles: pointer to ParticleGPU array on GPU
void smokeCudaApplyDragToParticles(
    void* d_mpmParticles,
    int numParticles,
    float3 worldMin, float dx,
    float dt, float dragCoeff
);

// Apply particle momentum to smoke (Particle→Grid with atomicAdd)
void smokeCudaApplyParticlesToSmoke(
    void* d_mpmParticles,
    int numParticles,
    float3 worldMin, float dx,
    float couplingStrength
);

}  // extern "C"

#endif // USE_CUDA

