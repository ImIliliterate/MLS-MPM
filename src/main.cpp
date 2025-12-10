/**
 * MLS-MPM + Stable Fluids Coupled Simulation
 * 
 * Real-time liquid and smoke simulation with two-way coupling.
 * Based on:
 *   - "A Moving Least Squares Material Point Method" (Hu et al. 2018)
 *   - "Stable Fluids" (Stam 1999)
 * 
 * Controls:
 *   - Left mouse drag: Rotate camera
 *   - Right mouse drag: Pan camera
 *   - Scroll: Zoom
 *   - Space: Toggle simulation
 *   - R: Reset simulation
 */

#include <glad/glad.h>
#ifndef GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_NONE
#endif
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "sim/MpmSim.h"
#include "sim/SmokeSim.h"
#include "sim/Coupling.h"
#include "sim/SDF.h"
#include "sim/SimulationParams.h"

#include "render/Camera.h"
#include "render/Shader.h"
#include "render/LiquidRenderer.h"
#include "render/SmokeRenderer.h"

#include <iostream>
#include <string>
#include <chrono>

// Window settings
constexpr int WINDOW_WIDTH = 1280;
constexpr int WINDOW_HEIGHT = 720;

// Simulation settings
constexpr float FIXED_DT = 1.0f / 60.0f;
constexpr int SUBSTEPS = 4;

// Demo scene types
enum class DemoScene {
    FallingBlock,
    Waterfall,
    BoilingCauldron,
    FanTest
};

// Global state
struct AppState {
    // Window
    GLFWwindow* window = nullptr;
    int windowWidth = WINDOW_WIDTH;
    int windowHeight = WINDOW_HEIGHT;
    
    // Camera
    Camera camera;
    bool mouseLeft = false;
    bool mouseRight = false;
    double lastMouseX = 0, lastMouseY = 0;
    
    // Simulation
    MpmSim mpm;
    SmokeSim smoke;
    Coupling coupling;
    SDF::Scene sdfScene;
    
    // Renderers
    LiquidRenderer liquidRenderer;
    SmokeRenderer smokeRenderer;
    
    // Time
    float simTime = 0.0f;
    float accumulator = 0.0f;
    bool running = true;
    float timeScale = 1.0f;
    
    // Demo
    DemoScene currentScene = DemoScene::FallingBlock;
    
    // UI settings
    bool showUI = true;
    bool renderLiquid = true;
    bool renderSmoke = true;       // Smoke rendering enabled by default
    int renderMode = 2;            // 0=SSFR, 1=Points, 2=Mesh (Zhu & Bridson surface)
    
    // Performance settings
    bool performanceMode = false;  // OFF by default - full features
    bool enableCohesion = false;   // OFF for now - focus on rendering
    bool enableSmokeSimulation = true;   // Smoke enabled by default
    int substeps = 4;  // Standard substeps
    
    // Stats
    float mpmTime = 0.0f;
    float smokeTime = 0.0f;
    float renderTime = 0.0f;
    float fps = 0.0f;
    
    AppState() : camera(glm::vec3(0.5f, 0.5f, 0.5f), 2.5f) {}
};

AppState g_app;

// Forward declarations
void initSimulation(DemoScene scene);
void simulationStep(float dt);
void render();
void renderUI();

// GLFW callbacks
void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    g_app.windowWidth = width;
    g_app.windowHeight = height;
    glViewport(0, 0, width, height);
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    if (ImGui::GetIO().WantCaptureMouse) return;
    
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        g_app.mouseLeft = (action == GLFW_PRESS);
    }
    if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        g_app.mouseRight = (action == GLFW_PRESS);
    }
}

void cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
    if (ImGui::GetIO().WantCaptureMouse) return;
    
    float dx = static_cast<float>(xpos - g_app.lastMouseX);
    float dy = static_cast<float>(ypos - g_app.lastMouseY);
    
    g_app.camera.processMouseMovement(dx, dy, g_app.mouseLeft, g_app.mouseRight);
    
    g_app.lastMouseX = xpos;
    g_app.lastMouseY = ypos;
}

void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    if (ImGui::GetIO().WantCaptureMouse) return;
    g_app.camera.processScroll(static_cast<float>(yoffset));
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action != GLFW_PRESS) return;
    
    switch (key) {
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(window, true);
            break;
        case GLFW_KEY_SPACE:
            g_app.running = !g_app.running;
            break;
        case GLFW_KEY_R:
            initSimulation(g_app.currentScene);
            break;
        case GLFW_KEY_H:
            g_app.showUI = !g_app.showUI;
            break;
        case GLFW_KEY_1:
            g_app.currentScene = DemoScene::FallingBlock;
            initSimulation(g_app.currentScene);
            break;
        case GLFW_KEY_2:
            g_app.currentScene = DemoScene::Waterfall;
            initSimulation(g_app.currentScene);
            break;
        case GLFW_KEY_3:
            g_app.currentScene = DemoScene::BoilingCauldron;
            initSimulation(g_app.currentScene);
            break;
        case GLFW_KEY_4:
            g_app.currentScene = DemoScene::FanTest;
            initSimulation(g_app.currentScene);
            break;
    }
}

void initSimulation(DemoScene scene) {
    // Resolution and performance settings
    g_params.fluidPreset = FluidPreset::Coarse;  // 48^3 grid
    g_params.particleSpacingFactor = 0.5f;
    g_app.substeps = 6;
    g_params.enableCohesion = false;
    
#ifdef USE_CUDA
    // GPU settings - higher quality
    g_params.fluidPreset = FluidPreset::Default;  // 64^3
    g_params.particleSpacingFactor = 0.6f;  // dx*0.6 = ~4.6x more particles, target Active PPC 3-5
    g_app.substeps = 4;  // Balanced
    g_app.renderSmoke = true;
    g_params.enableSmokeSimulation = true;
    g_params.enableCoupling = true;
#else
    // CPU: disable smoke for performance
    g_app.renderSmoke = false;
    g_params.enableSmokeSimulation = false;
    g_params.enableCoupling = false;
#endif
    g_params.skipDiagnosticsEveryFrame = true;
    g_params.diagnosticsInterval = 30;

    int gridRes = g_params.liquidGridRes;
    // Iteration 3: Resolution presets
    switch (g_params.fluidPreset) {
        case FluidPreset::Coarse:
            gridRes = 48;   // Faster, still decent
            break;
        case FluidPreset::Default:
            gridRes = 64;   // Iteration 3: 64³ default
            break;
        case FluidPreset::Fine:
            gridRes = 80;   // More detail
            break;
    }
    
    g_app.mpm.init(gridRes, gridRes, gridRes);
    g_app.smoke.init(g_params.smokeGridRes, g_params.smokeGridRes, g_params.smokeGridRes);
    
#ifdef USE_CUDA
    // Enable GPU smoke solver
    g_app.smoke.useGPU = true;
    g_app.smoke.initGPU();
    
    // Create visible GROUND FOG LAYER
    // Multiple overlapping spheres at floor level
    for (float x = 0.1f; x <= 0.9f; x += 0.2f) {
        for (float z = 0.1f; z <= 0.9f; z += 0.2f) {
            g_app.smoke.addDensity(glm::vec3(x, 0.08f, z), 0.4f, 0.15f);
        }
    }
    // Extra dense center
    g_app.smoke.addDensity(glm::vec3(0.5f, 0.1f, 0.5f), 0.6f, 0.25f);
#endif
    
    g_app.coupling.init(&g_app.mpm, &g_app.smoke);
    g_app.coupling.enabled = g_params.enableCoupling;
    
    g_app.sdfScene.clear();
    g_app.sdfScene.addFloor(0.05f);
    
    g_app.simTime = 0.0f;
    g_app.accumulator = 0.0f;
    
    // High-resolution fluid: use dx * spacingFactor for proper PPC
    float dx = 1.0f / static_cast<float>(gridRes);
    // spacing = dx * 0.5 gives 2^3 = 8 particles per cell in filled regions
    float spacing = dx * g_params.particleSpacingFactor;
    g_params.dx = dx;
    
    // Debug: small dots to see actual physics
    g_app.liquidRenderer.pointSize = 1.5f;  // Fine spray look (less grainy)
    g_app.liquidRenderer.isoLevel = g_params.isoThreshold;
    
    // Use Mesh mode for proper Zhu & Bridson surface reconstruction
    g_app.renderMode = 2;  // Mesh (marching cubes)
    
    switch (scene) {
        case DemoScene::FallingBlock: {
            // Simple falling block of liquid
            g_app.mpm.addBox(
                glm::vec3(0.3f, 0.5f, 0.3f),
                glm::vec3(0.7f, 0.8f, 0.7f),
                glm::vec3(0.0f),
                spacing,
                1000.0f
            );
            break;
        }
        
        case DemoScene::Waterfall: {
            // Add rocks at the bottom
            g_app.sdfScene.addBox(glm::vec3(0.3f, 0.15f, 0.5f), glm::vec3(0.15f, 0.1f, 0.15f));
            g_app.sdfScene.addBox(glm::vec3(0.6f, 0.12f, 0.4f), glm::vec3(0.1f, 0.07f, 0.12f));
            g_app.sdfScene.addSphere(glm::vec3(0.5f, 0.1f, 0.6f), 0.08f);
            
            // Initial water at top
            g_app.mpm.addBox(
                glm::vec3(0.2f, 0.7f, 0.4f),
                glm::vec3(0.4f, 0.9f, 0.6f),
                glm::vec3(0.5f, 0.0f, 0.0f),
                spacing,
                1000.0f
            );
            
            g_app.coupling.splashDensityAmount = 8.0f;
            g_app.coupling.splashTemperatureAmount = 3.0f;
            break;
        }
        
        case DemoScene::BoilingCauldron: {
            // Bowl shape
            g_app.sdfScene.addSphere(glm::vec3(0.5f, 0.0f, 0.5f), 0.35f);
            
            // Water in the bowl
            g_app.mpm.addSphere(
                glm::vec3(0.5f, 0.25f, 0.5f),
                0.2f,
                glm::vec3(0.0f),
                spacing,
                1000.0f
            );
            break;
        }
        
        case DemoScene::FanTest: {
            // Pool of water
            g_app.mpm.addBox(
                glm::vec3(0.25f, 0.05f, 0.25f),
                glm::vec3(0.75f, 0.25f, 0.75f),
                glm::vec3(0.0f),
                spacing * 1.2f,  // Even fewer particles
                1000.0f
            );
            break;
        }
    }
    
    // Set up SDF functions for MPM collision
    g_app.mpm.sdfFunc = [](const glm::vec3& p) -> float {
        return g_app.sdfScene.evaluate(p);
    };
    g_app.mpm.sdfGradFunc = [](const glm::vec3& p) -> glm::vec3 {
        return g_app.sdfScene.gradient(p);
    };
    
    // Mark GPU buffers dirty so particles get uploaded
    g_app.mpm.uploadToGpu();
    
    // Auto-start simulation after reset
    g_app.running = true;
    
    // Log resolution stats
    g_app.mpm.logResolutionStats();
    
    // Enhanced PPC diagnostics
    int numCells = gridRes * gridRes * gridRes;
    int numParticles = static_cast<int>(g_app.mpm.particles.size());
    float avgPPC = static_cast<float>(numParticles) / static_cast<float>(numCells);
    
    std::cout << "\n========== ITERATION 3 INIT ==========" << std::endl;
    std::cout << "Grid resolution: " << gridRes << "³ = " << numCells << " cells" << std::endl;
    std::cout << "Cell size (dx): " << dx << std::endl;
    std::cout << "Particle spacing: " << spacing << " (dx * " << g_params.particleSpacingFactor << ")" << std::endl;
    std::cout << "Total particles: " << numParticles << std::endl;
    std::cout << "Avg PPC (global): " << avgPPC << std::endl;
    std::cout << "Target: 150k+ particles, 8+ PPC in filled regions" << std::endl;
    std::cout << "---------------------------------------" << std::endl;
    std::cout << "FLIP ratio: " << g_params.flipRatio << std::endl;
    std::cout << "Bulk Modulus: " << g_params.bulkModulus << std::endl;
    std::cout << "Substeps: " << g_params.substeps << ", dt: " << g_params.dt << std::endl;
    std::cout << "Viscosity blend: " << g_params.viscosityBlend << std::endl;
    std::cout << "Grid smoothing: " << (g_params.enableGridSmoothing ? "ON" : "OFF") << std::endl;
    std::cout << "======================================\n" << std::endl;
}

void simulationStep(float dt) {
    int numSubsteps = g_app.substeps;
    float subDt = dt / numSubsteps;
    
    // Sync global params with simulation objects
    g_app.mpm.enableCohesion = g_params.enableCohesion;
    g_app.mpm.gravity = g_params.gravity;
    g_app.mpm.flipRatio = g_params.flipRatio;
    g_app.coupling.enabled = g_params.enableCoupling;
    g_app.enableCohesion = g_params.enableCohesion;  // Keep UI in sync
    
    for (int i = 0; i < numSubsteps; i++) {
        // MPM step
        auto mpmStart = std::chrono::high_resolution_clock::now();
        g_app.mpm.step(subDt);
        auto mpmEnd = std::chrono::high_resolution_clock::now();
        g_app.mpmTime = std::chrono::duration<float, std::milli>(mpmEnd - mpmStart).count();
        
        // Coupling FIRST - inject density/velocity from water impacts
        if (g_app.coupling.enabled) {
            g_app.coupling.apply(subDt);
        }
        
        // Smoke step - only on LAST substep (after all coupling injections)
        auto smokeStart = std::chrono::high_resolution_clock::now();
        
        if (g_params.enableSmokeSimulation && i == numSubsteps - 1) {
            // Scene-specific smoke sources
            switch (g_app.currentScene) {
                case DemoScene::FallingBlock:
                    // Continuous ground fog layer that water will blast through
                    for (float x = 0.2f; x <= 0.8f; x += 0.3f) {
                        for (float z = 0.2f; z <= 0.8f; z += 0.3f) {
                            g_app.smoke.addDensity(glm::vec3(x, 0.06f, z), 0.15f, 0.1f);
                        }
                    }
                    break;
                    
                case DemoScene::BoilingCauldron:
                    // Gentle heat source - prevents runaway velocities
                    g_app.smoke.addTemperature(glm::vec3(0.5f, 0.1f, 0.5f), 0.3f, 0.15f);
                    g_app.smoke.addDensity(glm::vec3(0.5f, 0.15f, 0.5f), 0.5f, 0.12f);
                    break;
                    
                case DemoScene::FanTest:
                    // Moderate wind from left side
                    g_app.smoke.addVelocity(glm::vec3(0.0f, 0.3f, 0.5f), 
                                            glm::vec3(2.0f, 0.0f, 0.0f), 0.2f);
                    g_app.smoke.addDensity(glm::vec3(0.1f, 0.3f, 0.5f), 0.5f, 0.1f);
                    break;
                    
                default:
                    break;
            }
            
            g_app.smoke.step(subDt * numSubsteps);  // Larger dt since we only step once
            
            // Update smoke diagnostics
            g_params.maxSmokeSpeed = g_app.smoke.getMaxSpeed();
        }
        auto smokeEnd = std::chrono::high_resolution_clock::now();
        g_app.smokeTime = std::chrono::duration<float, std::milli>(smokeEnd - smokeStart).count();
    }
    
    g_app.simTime += dt;
}

void render() {
    auto renderStart = std::chrono::high_resolution_clock::now();
    
    glClearColor(0.02f, 0.02f, 0.05f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    float aspect = static_cast<float>(g_app.windowWidth) / g_app.windowHeight;
    glm::mat4 view = g_app.camera.getViewMatrix();
    glm::mat4 projection = g_app.camera.getProjectionMatrix(aspect);
    glm::vec3 cameraPos = g_app.camera.getPosition();
    glm::vec3 lightDir = glm::normalize(glm::vec3(-0.5f, -1.0f, -0.3f));
    
    glEnable(GL_DEPTH_TEST);
    
    // Render smoke first (background)
    if (g_app.renderSmoke) {
        g_app.smokeRenderer.updateDensity(g_app.smoke);
        g_app.smokeRenderer.render(view, projection, cameraPos, lightDir,
                                   g_app.smoke.worldMin, g_app.smoke.worldMax);
    }
    
    // Render liquid
    if (g_app.renderLiquid) {
        // Set render mode: 0=SSFR, 1=Points, 2=Mesh
        switch (g_app.renderMode) {
            case 0: g_app.liquidRenderer.mode = LiquidRenderer::RenderMode::SSFR; break;
            case 1: g_app.liquidRenderer.mode = LiquidRenderer::RenderMode::Points; break;
            case 2: g_app.liquidRenderer.mode = LiquidRenderer::RenderMode::Mesh; break;
        }
        
        // Mesh mode: Generate smooth liquid surface using Marching Cubes
        if (g_app.renderMode == 2) {
            // OPTIMIZATION: Only regenerate mesh every 3rd frame
            // (Mesh changes slowly enough that this is nearly invisible)
            static int meshFrameCounter = 0;
            meshFrameCounter++;
            if (meshFrameCounter % 4 == 0) {  // Every 4 frames for better perf
                g_app.mpm.syncParticlesToCpu();
                g_app.liquidRenderer.generateMesh(g_app.mpm, g_params.isoThreshold);
            }
        } else {
            // Points mode: just download positions (fast - only 12 bytes per particle)
            static std::vector<glm::vec3> renderPositions;
            g_app.mpm.getPositionsForRendering(renderPositions);
            g_app.liquidRenderer.updatePositions(renderPositions);
        }
        
        g_app.liquidRenderer.render(view, projection, cameraPos, lightDir, 
                                       g_app.windowWidth, g_app.windowHeight);
    }
    
    auto renderEnd = std::chrono::high_resolution_clock::now();
    g_app.renderTime = std::chrono::duration<float, std::milli>(renderEnd - renderStart).count();
}

void renderUI() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    
    if (g_app.showUI) {
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(320, 500), ImGuiCond_FirstUseEver);
        
        ImGui::Begin("MLS-MPM + Smoke Simulation", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
        
        // ==================== DIAGNOSTICS ====================
        if (ImGui::CollapsingHeader("Diagnostics", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("FPS: %.1f", g_app.fps);
            ImGui::Text("Particles: %d", g_params.numLiquidParticles);
            
            // Phase 2: Particles per cell (critical for fluid behavior)
            ImGui::Text("Avg PPC (all cells): %.2f", g_params.avgParticlesPerCell);
            ImGui::Text("Active PPC: %.2f", g_params.activePPC);
            if (g_params.activePPC < 2.0f) {
                ImGui::TextColored(ImVec4(1,0.3f,0.3f,1), "  WARNING: Low PPC = gravel!");
            } else if (g_params.activePPC < 4.0f) {
                ImGui::TextColored(ImVec4(1,0.8f,0.3f,1), "  OK, but more is better");
            } else {
                ImGui::TextColored(ImVec4(0.3f,1,0.3f,1), "  Good fluid density");
            }
            ImGui::Text("Active Cells: %d", g_params.activeCellCount);
            
        ImGui::Separator();
            ImGui::Text("Fluid State:");
            ImGui::Text("  Max Density: %.1f kg/m³", g_params.maxDensity);
            ImGui::Text("  Max Pressure: %.2f", g_params.maxPressure);
            ImGui::Text("  Max Speed: %.2f m/s", g_params.maxLiquidSpeed);
            ImGui::Text("  Actual dt: %.4f s", g_params.actualDt);
        
            ImGui::Separator();
        ImGui::Text("Timing (ms):");
        ImGui::Text("  MPM:    %.2f", g_app.mpmTime);
        ImGui::Text("  Smoke:  %.2f", g_app.smokeTime);
        ImGui::Text("  Render: %.2f", g_app.renderTime);
            ImGui::Text("Triangles: %zu", g_app.liquidRenderer.getTriangleCount());
        }
        
        ImGui::Separator();
        
        // ==================== PLAYBACK ====================
        if (ImGui::Button(g_app.running ? "Pause" : "Play")) {
            g_app.running = !g_app.running;
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset")) {
            initSimulation(g_app.currentScene);
        }
        ImGui::SliderFloat("Time Scale", &g_app.timeScale, 0.1f, 2.0f);
        
        // Demo scenes
        const char* sceneNames[] = { "Falling Block", "Waterfall", "Boiling Cauldron", "Fan Test" };
        int currentScene = static_cast<int>(g_app.currentScene);
        if (ImGui::Combo("Scene", &currentScene, sceneNames, 4)) {
            g_app.currentScene = static_cast<DemoScene>(currentScene);
            initSimulation(g_app.currentScene);
        }
        ImGui::Separator();
        
        // ==================== MPM PARAMETERS ====================
        if (ImGui::CollapsingHeader("MPM Simulation")) {
            // Phase 1: Resolution Presets (for PPC control)
            ImGui::Text("Iteration 3: Resolution (PPC control)");
            int preset = static_cast<int>(g_params.fluidPreset);
            if (ImGui::Combo("Grid Preset", &preset, "Coarse (48)\0Default (64)\0Fine (80)\0")) {
                g_params.fluidPreset = static_cast<FluidPreset>(preset);
                initSimulation(g_app.currentScene);
            }
            ImGui::SliderFloat("Spacing Factor", &g_params.particleSpacingFactor, 0.3f, 1.0f, "%.2f");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("spacing = dx * factor\n0.5 = ~8 PPC, 0.4 = ~16 PPC, 1.0 = ~1 PPC");
            ImGui::Text("Particles: %d, Active PPC: %.1f", g_params.numLiquidParticles, g_params.activePPC);
            
            // Phase 2: FLIP/PIC (kill viscosity)
            ImGui::Separator();
            ImGui::Text("Phase 2: FLIP/PIC (viscosity)");
            ImGui::SliderFloat("FLIP Ratio", &g_params.flipRatio, 0.90f, 0.99f, "%.2f");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("0.97 = water, 0.95 = slightly damped\n0.98 = very splashy");
            
            if (ImGui::Button("0.95")) g_params.flipRatio = 0.95f;
            ImGui::SameLine();
            if (ImGui::Button("0.97")) g_params.flipRatio = 0.97f;
            ImGui::SameLine();
            if (ImGui::Button("0.98")) g_params.flipRatio = 0.98f;
            
            // Phase 3: Stiffness (less squishy)
            ImGui::Separator();
            ImGui::Text("Phase 3: Stiffness (less bouncy)");
            ImGui::SliderFloat("Stiffness K", &g_params.stiffnessK, 50.0f, 200.0f);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Higher = less squishy, more spread");
            
            if (ImGui::Button("50##k")) g_params.stiffnessK = 50.0f;
            ImGui::SameLine();
            if (ImGui::Button("100##k")) g_params.stiffnessK = 100.0f;
            ImGui::SameLine();
            if (ImGui::Button("150##k")) g_params.stiffnessK = 150.0f;
            
            ImGui::SliderFloat("Gravity", &g_params.gravity, -20.0f, 0.0f);
            ImGui::SliderInt("Substeps", &g_app.substeps, 1, 16);
            
            // Round 6: Quality presets for performance vs fidelity
            ImGui::Separator();
            ImGui::Text("Quality Preset:");
            if (ImGui::Button("Fast")) {
                g_app.substeps = 2;
                g_params.smoothingIterations = 1;
                g_params.skipDiagnosticsEveryFrame = true;
                g_params.diagnosticsInterval = 20;
            }
            ImGui::SameLine();
            if (ImGui::Button("Balanced")) {
                g_app.substeps = 3;
                g_params.smoothingIterations = 2;
                g_params.skipDiagnosticsEveryFrame = true;
                g_params.diagnosticsInterval = 10;
            }
            ImGui::SameLine();
            if (ImGui::Button("Quality")) {
                g_app.substeps = 4;
                g_params.smoothingIterations = 3;
                g_params.skipDiagnosticsEveryFrame = false;
            }
            
            // Viscosity: Should be OFF for water
            ImGui::Separator();
            ImGui::Text("Viscosity (OFF for water!):");
            ImGui::Checkbox("Enable Grid Smoothing", &g_params.enableGridSmoothing);
            if (g_params.enableGridSmoothing) {
                ImGui::TextColored(ImVec4(1,0.5f,0.5f,1), "WARNING: This adds viscosity!");
            }
        }
        
        // ==================== PHASE 4: MESHING ====================
        if (ImGui::CollapsingHeader("Meshing (Phase 4)")) {
            ImGui::SliderFloat("Kernel Radius", &g_params.smoothingRadius, 1.5f, 4.0f);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("2.5 = default, higher = smoother");
            ImGui::SliderInt("Smooth Iterations", &g_params.smoothingIterations, 1, 5);
            ImGui::SliderFloat("Iso Threshold", &g_params.isoThreshold, 0.2f, 1.0f);
            ImGui::SliderFloat("Anisotropy", &g_params.meshAnisotropy, 0.0f, 0.5f);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Stretch kernel along velocity (for waterfalls)");
        }
        
        // ==================== COUPLING ====================
        if (ImGui::CollapsingHeader("Coupling")) {
            ImGui::Checkbox("Enable Coupling", &g_params.enableCoupling);
            ImGui::Checkbox("Liquid -> Smoke", &g_app.coupling.liquidToSmoke);
            ImGui::Checkbox("Smoke -> Liquid", &g_app.coupling.smokeToLiquid);
            
            ImGui::Separator();
            ImGui::Text("Phase 1.1: Implicit Drag (stable)");
            ImGui::SliderFloat("Drag Coeff (C)", &g_params.dragCoeff, 0.0f, 5.0f);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Strength of drag. Formula: v_new = (v + k*v_smoke)/(1+k)");
            ImGui::SliderFloat("Max Drag Delta", &g_params.maxDragDelta, 1.0f, 20.0f);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Max velocity change per step (m/s)");
            
            ImGui::Separator();
            ImGui::Text("Phase 5: Buoyancy (heavily reduced)");
            ImGui::SliderFloat("Buoyancy Coeff", &g_params.buoyancyCoeff, 0.0f, 0.1f, "%.3f");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("0.01 = gentle surface bubbles only");
            ImGui::SliderFloat("Max Buoy Accel", &g_params.buoyancyMaxAccel, 0.0f, 3.0f);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%.2f g (keep < 0.2g)", g_params.buoyancyMaxAccel / 9.8f);
            
            ImGui::Separator();
            ImGui::Text("Phase 1.2: Surface Detection");
            ImGui::SliderInt("Neighbor Threshold", &g_params.surfaceNeighborThreshold, 5, 40);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Particles with fewer neighbors are 'surface'");
            ImGui::Text("Surface particles: %d", g_params.surfaceParticleCount);
            
            ImGui::Separator();
            ImGui::Text("Liquid -> Smoke:");
            ImGui::SliderFloat("Spray Density", &g_params.sprayDensityGain, 0.0f, 20.0f);
            ImGui::SliderFloat("Splash Threshold", &g_params.splashVelocityThreshold, 0.5f, 5.0f);
        }
        
        // ==================== RENDERING ====================
        if (ImGui::CollapsingHeader("Rendering")) {
        ImGui::Checkbox("Render Liquid", &g_app.renderLiquid);
        ImGui::Checkbox("Render Smoke", &g_app.renderSmoke);
        
        // Render mode selector
        const char* renderModes[] = { "SSFR (Smooth)", "Points", "Mesh" };
        ImGui::Combo("Render Mode", &g_app.renderMode, renderModes, 3);
        
        if (g_app.renderMode == 0) {  // SSFR mode
            ImGui::Text("Curvature Flow Smoothing:");
            ImGui::SliderFloat("Particle Radius", &g_app.liquidRenderer.ssfrParticleRadius, 0.02f, 0.06f);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Larger = more overlap = smoother");
            ImGui::SliderFloat("Point Scale", &g_app.liquidRenderer.ssfrPointScale, 800.0f, 2500.0f);
            ImGui::SliderInt("Smooth Iterations", &g_app.liquidRenderer.ssfrSmoothIterations, 10, 60);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("More = smoother surface (costs perf)");
            ImGui::SliderFloat("Smooth Rate", &g_app.liquidRenderer.ssfrSmoothDt, 0.0001f, 0.001f, "%.4f");
            ImGui::Separator();
            ImGui::Text("Water Shading:");
            ImGui::SliderFloat("Fresnel Power", &g_app.liquidRenderer.ssfrFresnelPower, 1.0f, 8.0f);
            ImGui::SliderFloat("Thickness", &g_app.liquidRenderer.ssfrThicknessScale, 1.0f, 10.0f);
        }
        if (g_app.renderMode == 2) {  // Mesh mode
            ImGui::SliderFloat("Iso Level", &g_params.isoThreshold, 0.1f, 2.0f);
            ImGui::SliderFloat("Smoothing Radius", &g_params.smoothingRadius, 1.0f, 5.0f);
            ImGui::SliderInt("Smooth Iterations", &g_params.smoothingIterations, 0, 5);
        }
        ImGui::SliderFloat("Smoke Density", &g_app.smokeRenderer.densityScale, 1.0f, 50.0f);
        ImGui::SliderInt("Ray Steps", &g_app.smokeRenderer.raySteps, 4, 64);
        }
        
        // ==================== PERFORMANCE ====================
        if (ImGui::CollapsingHeader("Performance")) {
            if (ImGui::Checkbox("Performance Mode", &g_app.performanceMode)) {
                if (g_app.performanceMode) {
                    g_app.renderMode = 2;  // Keep Mesh mode
                    g_params.enableCohesion = false;
                    g_params.enableSmokeSimulation = false;
                    g_params.enableGridSmoothing = false;
                } else {
                    g_params.enableCohesion = true;  // Re-enable for quality
                    g_params.enableGridSmoothing = true;
                }
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Quick toggle for laptop-friendly settings");
            }
            ImGui::Checkbox("Surface Tension", &g_params.enableCohesion);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Keeps liquid together - essential for water look!");
            if (g_params.enableCohesion) {
                ImGui::SliderFloat("Cohesion Strength", &g_params.cohesionStrength, 100.0f, 1500.0f);
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Higher = more surface tension");
            }
            ImGui::Checkbox("Smoke Simulation", &g_params.enableSmokeSimulation);
        }
        
        ImGui::Separator();
        if (ImGui::CollapsingHeader("Controls")) {
            ImGui::Text("LMB: Rotate camera");
            ImGui::Text("RMB: Pan camera");
            ImGui::Text("Scroll: Zoom");
            ImGui::Text("Space: Play/Pause");
            ImGui::Text("R: Reset");
            ImGui::Text("H: Toggle UI");
            ImGui::Text("1-4: Switch scenes");
        }
        
        ImGui::End();
    }
    
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    
    g_app.window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, 
        "MLS-MPM + Smoke Coupled Simulation", nullptr, nullptr);
    if (!g_app.window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    
    glfwMakeContextCurrent(g_app.window);
    glfwSwapInterval(0);  // Disable VSync for max FPS
    
    // Set callbacks
    glfwSetFramebufferSizeCallback(g_app.window, framebufferSizeCallback);
    glfwSetMouseButtonCallback(g_app.window, mouseButtonCallback);
    glfwSetCursorPosCallback(g_app.window, cursorPosCallback);
    glfwSetScrollCallback(g_app.window, scrollCallback);
    glfwSetKeyCallback(g_app.window, keyCallback);
    
    // Initialize GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return -1;
    }
    
    std::cout << "OpenGL: " << glGetString(GL_VERSION) << std::endl;
    const char* rendererCStr = reinterpret_cast<const char*>(glGetString(GL_RENDERER));
    std::string rendererStr = rendererCStr ? rendererCStr : "";
    std::cout << "Renderer: " << rendererStr << std::endl;
    // Fallback: if we are on a software or basic driver, force light settings
    if (rendererStr.find("Microsoft") != std::string::npos ||
        rendererStr.find("llvmpipe")  != std::string::npos ||
        rendererStr.find("SwiftShader") != std::string::npos) {
        g_app.performanceMode = true;
        g_app.renderMode = 1;                   // Points for software renderer
        g_app.substeps = 2;                     // fewer physics steps
        g_params.fluidPreset = FluidPreset::Coarse; // lower grid res
        g_params.smoothingIterations = 1;
        g_params.skipDiagnosticsEveryFrame = true;
        g_params.diagnosticsInterval = 30;
        g_params.enableSmokeSimulation = false;
        g_params.enableCoupling = false;
        g_app.smokeRenderer.raySteps = 24;      // cheaper smoke
    }
    
    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(g_app.window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
    
    // Initialize renderers
    if (!g_app.liquidRenderer.init()) {
        std::cerr << "Failed to initialize liquid renderer" << std::endl;
        return -1;
    }
    
    if (!g_app.smokeRenderer.init()) {
        std::cerr << "Failed to initialize smoke renderer" << std::endl;
        return -1;
    }
    
    // Initialize simulation
    initSimulation(DemoScene::FallingBlock);
    
    // Main loop
    auto lastTime = std::chrono::high_resolution_clock::now();
    int frameCount = 0;
    float fpsAccum = 0.0f;
    
    while (!glfwWindowShouldClose(g_app.window)) {
        auto currentTime = std::chrono::high_resolution_clock::now();
        float deltaTime = std::chrono::duration<float>(currentTime - lastTime).count();
        lastTime = currentTime;
        
        // FPS counter
        frameCount++;
        fpsAccum += deltaTime;
        if (fpsAccum >= 1.0f) {
            g_app.fps = frameCount / fpsAccum;
            printf("FPS: %.1f | Particles: %d | Substeps: %d\n", 
                   g_app.fps, g_params.numLiquidParticles, g_app.substeps);
            frameCount = 0;
            fpsAccum = 0.0f;
        }
        
        glfwPollEvents();
        
        // Update simulation with fixed timestep
        if (g_app.running) {
            g_app.accumulator += deltaTime * g_app.timeScale;
            
            // Prevent spiral of death
            if (g_app.accumulator > FIXED_DT * 4) {
                g_app.accumulator = FIXED_DT * 4;
            }
            
            while (g_app.accumulator >= FIXED_DT) {
                simulationStep(FIXED_DT);
                g_app.accumulator -= FIXED_DT;
            }
        }
        
        // Render
        render();
        renderUI();
        
        glfwSwapBuffers(g_app.window);
    }
    
    // Cleanup
    g_app.liquidRenderer.cleanup();
    g_app.smokeRenderer.cleanup();
    
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    
    glfwDestroyWindow(g_app.window);
    glfwTerminate();
    
    return 0;
}

