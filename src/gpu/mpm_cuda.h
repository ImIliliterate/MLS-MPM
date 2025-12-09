/**
 * CUDA interface for MLS-MPM simulation
 * 
 * Provides persistent GPU memory management to avoid per-frame allocation overhead.
 */

#pragma once

#ifdef USE_CUDA

#include <glm/glm.hpp>
#include <vector>

struct Particle;

/**
 * Initialize persistent GPU buffers for simulation.
 * Call once at startup or when grid size changes.
 */
void mpmCudaInit(int maxParticles, int Nx, int Ny, int Nz);

/**
 * Free GPU buffers. Call on shutdown.
 */
void mpmCudaCleanup();

/**
 * Upload particles to GPU. Call on reset/spawn.
 */
void mpmCudaUploadParticles(const std::vector<Particle>& particles);

/**
 * Run one simulation step entirely on GPU.
 * Particles stay on device; no download happens here.
 */
void mpmCudaStep(
    int numParticles,
    int Nx, int Ny, int Nz,
    float dx, float dt,
    float apicBlend,
    float flipRatio,
    float gravity,
    const glm::vec3& worldMin,
    const glm::vec3& worldMax);

/**
 * Download only particle positions for rendering.
 * Much faster than downloading full particle state.
 */
void mpmCudaDownloadPositions(std::vector<glm::vec3>& positions, int numParticles);

/**
 * Download full particle state (for debugging or CPU fallback).
 */
void mpmCudaDownloadParticles(std::vector<Particle>& particles);

/**
 * Get current particle count on GPU.
 */
int mpmCudaGetParticleCount();

/**
 * Get GPU particle buffer pointer (for coupling with smoke).
 * Returns nullptr if not initialized.
 */
void* mpmCudaGetParticleBuffer();

/**
 * Compute CFL-safe timestep based on max particle velocity.
 * Returns adaptive dt that guarantees stability.
 */
float mpmCudaComputeSafeDt(int numParticles, float dx, float bulkModulus, float density);

#endif // USE_CUDA
