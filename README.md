# MLS-MPM + Stable Fluids Coupled Simulation

Real-time liquid and smoke simulation with two-way coupling for GPU (designed for RTX 5070 Ti).

## Features

- **MLS-MPM Liquid Simulation**: Moving Least Squares Material Point Method for realistic liquid behavior
- **Stable Fluids Smoke**: Eulerian grid-based smoke simulation with buoyancy and advection
- **Two-Way Coupling**: Liquid splashes create smoke/mist; wind affects liquid surface
- **Real-Time Rendering**: Marching cubes mesh for liquid, ray-marched volume for smoke
- **Interactive UI**: ImGui-based controls for all parameters

## Dependencies

- CMake 3.20+
- C++17 compiler
- OpenGL 3.3+
- GLFW3 (fetched automatically)
- GLM (fetched automatically)
- Dear ImGui (fetched automatically)
- CUDA Toolkit (optional, for GPU acceleration)

## Building

```bash
mkdir build && cd build

# CPU-only build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8

# With CUDA support
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON
make -j8
```

## Running

```bash
./mpm_smoke
```

## Controls

| Key/Mouse | Action |
|-----------|--------|
| Left Mouse Drag | Rotate camera |
| Right Mouse Drag | Pan camera |
| Scroll | Zoom in/out |
| Space | Play/Pause simulation |
| R | Reset simulation |
| H | Toggle UI |
| 1 | Falling Block scene |
| 2 | Waterfall scene |
| 3 | Boiling Cauldron scene |
| 4 | Fan Test scene |
| Esc | Quit |

## Demo Scenes

1. **Falling Block**: Simple liquid block falling under gravity
2. **Waterfall**: Liquid flowing over rocks with splash effects
3. **Boiling Cauldron**: Heated liquid with steam rising
4. **Fan Test**: Wind from a fan affecting liquid surface

## Architecture

```
src/
  main.cpp              # Application entry, main loop, UI
  sim/
    MpmSim.h/.cpp       # MLS-MPM liquid simulation
    SmokeSim.h/.cpp     # Stable Fluids smoke simulation
    Coupling.h/.cpp     # Two-way liquid-smoke coupling
    SDF.h/.cpp          # Signed distance functions for collisions
  render/
    Camera.h/.cpp       # Orbit camera
    Shader.h/.cpp       # OpenGL shader wrapper
    LiquidRenderer.h/.cpp  # Particle/mesh rendering with marching cubes
    SmokeRenderer.h/.cpp   # Volume ray-marching
  gpu/
    mpm_kernels.cu      # CUDA kernels for MPM (placeholder)
    smoke_kernels.cu    # CUDA kernels for smoke (placeholder)
```

## Algorithm Overview

### MLS-MPM (Liquid)

1. **Clear Grid**: Zero all grid node masses and velocities
2. **P2G (Particle to Grid)**: Transfer particle mass/momentum to grid using quadratic B-spline weights
3. **Grid Update**: Apply gravity, enforce boundary conditions
4. **G2P (Grid to Particle)**: Interpolate grid velocities back to particles, update positions and deformation gradients

### Stable Fluids (Smoke)

1. **Add Forces**: Apply buoyancy from temperature/density
2. **Advect Velocity**: Semi-Lagrangian advection
3. **Diffuse** (optional): Implicit diffusion
4. **Project**: Pressure solve to enforce incompressibility
5. **Advect Scalars**: Move density and temperature with flow

### Coupling

**Liquid → Smoke:**
- Mark liquid-occupied cells as obstacles in smoke grid
- Detect high-velocity impacts and inject smoke density/temperature

**Smoke → Liquid:**
- Apply drag force from smoke velocity to surface particles
- Apply buoyancy from hot regions

## Performance Tips

- Reduce grid resolution for faster simulation
- Use point rendering mode for debugging (faster than mesh)
- Reduce ray-marching steps for smoke if needed
- Enable CUDA build for GPU acceleration (WIP)

## References

- Hu et al., "A Moving Least Squares Material Point Method with Displacement Discontinuity and Two-Way Rigid Body Coupling" (2018)
- Stam, "Stable Fluids" (1999)
- Fedkiw et al., "Visual Simulation of Smoke" (2001)

## License

MIT License

