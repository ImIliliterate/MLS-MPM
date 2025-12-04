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

#include "render/Camera.h"
#include "render/Shader.h"
#include "render/LiquidRenderer.h"
#include "render/SmokeRenderer.h"

#include <iostream>
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
    bool renderSmoke = true;
    bool renderPoints = true;  // Start with point mode to see particles
    
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
    // Reset simulations
    g_app.mpm.init(64, 64, 64);
    g_app.smoke.init(32, 32, 32);
    g_app.coupling.init(&g_app.mpm, &g_app.smoke);
    g_app.coupling.enabled = false;  // Disable coupling for debugging
    
    g_app.sdfScene.clear();
    g_app.sdfScene.addFloor(0.05f);
    
    g_app.simTime = 0.0f;
    g_app.accumulator = 0.0f;
    
    // Increase point size for visibility
    g_app.liquidRenderer.pointSize = 8.0f;
    
    switch (scene) {
        case DemoScene::FallingBlock: {
            // Simple falling block of liquid
            g_app.mpm.addBox(
                glm::vec3(0.3f, 0.5f, 0.3f),
                glm::vec3(0.7f, 0.8f, 0.7f),
                glm::vec3(0.0f),
                0.015f,
                1000.0f
            );
            std::cout << "Created " << g_app.mpm.particles.size() << " particles" << std::endl;
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
                glm::vec3(0.5f, 0.0f, 0.0f),  // Slight horizontal velocity
                0.012f,
                1000.0f
            );
            
            // Increase smoke splash effects
            g_app.coupling.splashDensityAmount = 8.0f;
            g_app.coupling.splashTemperatureAmount = 3.0f;
            break;
        }
        
        case DemoScene::BoilingCauldron: {
            // Bowl shape (approximated with a low sphere)
            g_app.sdfScene.addSphere(glm::vec3(0.5f, 0.0f, 0.5f), 0.35f);
            
            // Water in the bowl
            g_app.mpm.addSphere(
                glm::vec3(0.5f, 0.25f, 0.5f),
                0.2f,
                glm::vec3(0.0f),
                0.015f,
                1000.0f
            );
            
            // Hot region will be added during simulation
            break;
        }
        
        case DemoScene::FanTest: {
            // Pool of water
            g_app.mpm.addBox(
                glm::vec3(0.2f, 0.05f, 0.2f),
                glm::vec3(0.8f, 0.3f, 0.8f),
                glm::vec3(0.0f),
                0.02f,
                1000.0f
            );
            
            // Fan will inject velocity during simulation
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
}

void simulationStep(float dt) {
    float subDt = dt / SUBSTEPS;
    
    for (int i = 0; i < SUBSTEPS; i++) {
        // MPM step
        auto mpmStart = std::chrono::high_resolution_clock::now();
        g_app.mpm.step(subDt);
        auto mpmEnd = std::chrono::high_resolution_clock::now();
        g_app.mpmTime = std::chrono::duration<float, std::milli>(mpmEnd - mpmStart).count();
        
        // Smoke step
        auto smokeStart = std::chrono::high_resolution_clock::now();
        
        // Scene-specific smoke sources
        switch (g_app.currentScene) {
            case DemoScene::BoilingCauldron:
                // Add heat at the bottom
                g_app.smoke.addTemperature(glm::vec3(0.5f, 0.1f, 0.5f), 0.5f, 0.15f);
                g_app.smoke.addDensity(glm::vec3(0.5f, 0.15f, 0.5f), 0.2f, 0.1f);
                break;
                
            case DemoScene::FanTest:
                // Fan blowing from the side
                g_app.smoke.addVelocity(glm::vec3(0.0f, 0.3f, 0.5f), 
                                        glm::vec3(3.0f, 0.0f, 0.0f), 0.15f);
                // Some smoke to visualize
                g_app.smoke.addDensity(glm::vec3(0.1f, 0.3f, 0.5f), 0.3f, 0.08f);
                break;
                
            default:
                break;
        }
        
        g_app.smoke.step(subDt);
        auto smokeEnd = std::chrono::high_resolution_clock::now();
        g_app.smokeTime = std::chrono::duration<float, std::milli>(smokeEnd - smokeStart).count();
        
        // Coupling
        if (g_app.coupling.enabled) {
            g_app.coupling.apply(subDt);
        }
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
        g_app.liquidRenderer.mode = g_app.renderPoints ? 
            LiquidRenderer::RenderMode::Points : LiquidRenderer::RenderMode::Mesh;
        
        g_app.liquidRenderer.updateParticles(g_app.mpm.particles);
        
        if (!g_app.renderPoints) {
            g_app.liquidRenderer.generateMesh(g_app.mpm, 0.5f);
        }
        
        g_app.liquidRenderer.render(view, projection, cameraPos, lightDir);
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
        
        // Stats
        ImGui::Text("FPS: %.1f", g_app.fps);
        ImGui::Text("Particles: %zu", g_app.mpm.particles.size());
        ImGui::Text("Triangles: %zu", g_app.liquidRenderer.getTriangleCount());
        ImGui::Separator();
        
        ImGui::Text("Timing (ms):");
        ImGui::Text("  MPM:    %.2f", g_app.mpmTime);
        ImGui::Text("  Smoke:  %.2f", g_app.smokeTime);
        ImGui::Text("  Render: %.2f", g_app.renderTime);
        ImGui::Separator();
        
        // Playback controls
        if (ImGui::Button(g_app.running ? "Pause" : "Play")) {
            g_app.running = !g_app.running;
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset")) {
            initSimulation(g_app.currentScene);
        }
        ImGui::SliderFloat("Time Scale", &g_app.timeScale, 0.1f, 2.0f);
        ImGui::Separator();
        
        // Demo scenes
        ImGui::Text("Demo Scenes:");
        const char* sceneNames[] = { "Falling Block", "Waterfall", "Boiling Cauldron", "Fan Test" };
        int currentScene = static_cast<int>(g_app.currentScene);
        if (ImGui::Combo("Scene", &currentScene, sceneNames, 4)) {
            g_app.currentScene = static_cast<DemoScene>(currentScene);
            initSimulation(g_app.currentScene);
        }
        ImGui::Separator();
        
        // Rendering options
        ImGui::Text("Rendering:");
        ImGui::Checkbox("Render Liquid", &g_app.renderLiquid);
        ImGui::Checkbox("Render Smoke", &g_app.renderSmoke);
        ImGui::Checkbox("Point Mode", &g_app.renderPoints);
        
        if (!g_app.renderPoints) {
            ImGui::SliderFloat("Iso Level", &g_app.liquidRenderer.isoLevel, 0.1f, 2.0f);
        }
        ImGui::SliderFloat("Smoke Density", &g_app.smokeRenderer.densityScale, 1.0f, 50.0f);
        ImGui::SliderInt("Ray Steps", &g_app.smokeRenderer.raySteps, 16, 128);
        ImGui::Separator();
        
        // Simulation parameters
        ImGui::Text("Simulation:");
        ImGui::SliderFloat("Gravity", &g_app.mpm.gravity, -20.0f, 0.0f);
        ImGui::SliderFloat("FLIP Ratio", &g_app.mpm.flipRatio, 0.0f, 1.0f);
        ImGui::Separator();
        
        // Coupling
        ImGui::Text("Coupling:");
        ImGui::Checkbox("Enable Coupling", &g_app.coupling.enabled);
        ImGui::Checkbox("Liquid -> Smoke", &g_app.coupling.liquidToSmoke);
        ImGui::Checkbox("Smoke -> Liquid", &g_app.coupling.smokeToLiquid);
        ImGui::SliderFloat("Drag", &g_app.coupling.dragCoefficient, 0.0f, 2.0f);
        ImGui::SliderFloat("Buoyancy", &g_app.coupling.buoyancyCoefficient, 0.0f, 1.0f);
        
        ImGui::Separator();
        ImGui::Text("Controls:");
        ImGui::Text("  LMB: Rotate camera");
        ImGui::Text("  RMB: Pan camera");
        ImGui::Text("  Scroll: Zoom");
        ImGui::Text("  Space: Play/Pause");
        ImGui::Text("  R: Reset");
        ImGui::Text("  H: Toggle UI");
        ImGui::Text("  1-4: Switch scenes");
        
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
    glfwSwapInterval(1);  // VSync
    
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
    std::cout << "Renderer: " << glGetString(GL_RENDERER) << std::endl;
    
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

