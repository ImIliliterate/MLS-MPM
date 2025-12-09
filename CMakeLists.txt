cmake_minimum_required(VERSION 3.20)
project(MpmSmokeSim VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# Find packages
find_package(OpenGL REQUIRED)

# OpenMP for parallelization
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found - enabling parallel simulation")
endif()

# Option for CUDA support
option(USE_CUDA "Enable CUDA support for GPU acceleration" OFF)
if(USE_CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

# Add external dependencies via FetchContent
include(FetchContent)

# GLFW
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.3.8
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# GLM
FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 1.0.1
)
FetchContent_MakeAvailable(glm)

# Dear ImGui
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.89.9
)
FetchContent_MakeAvailable(imgui)

# Create imgui library
add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui PUBLIC 
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
    ${CMAKE_SOURCE_DIR}/external/glad/include
)
target_compile_definitions(imgui PUBLIC GLFW_INCLUDE_NONE)
target_link_libraries(imgui PUBLIC glfw glad OpenGL::GL)

# GLAD - we'll generate this ourselves
add_library(glad STATIC
    ${CMAKE_SOURCE_DIR}/external/glad/src/glad.c
)
target_include_directories(glad PUBLIC ${CMAKE_SOURCE_DIR}/external/glad/include)

# Collect source files
set(SIM_SOURCES
    src/sim/MpmSim.cpp
    src/sim/SmokeSim.cpp
    src/sim/Coupling.cpp
    src/sim/SDF.cpp
    src/sim/SimulationParams.cpp
)

set(RENDER_SOURCES
    src/render/LiquidRenderer.cpp
    src/render/SmokeRenderer.cpp
    src/render/Shader.cpp
    src/render/Camera.cpp
)

set(MAIN_SOURCES
    src/main.cpp
)

# Main executable
add_executable(mpm_smoke
    ${MAIN_SOURCES}
    ${SIM_SOURCES}
    ${RENDER_SOURCES}
)

target_include_directories(mpm_smoke PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(mpm_smoke PRIVATE
    glfw
    glad
    glm::glm
    imgui
    OpenGL::GL
)

# Add OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(mpm_smoke PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(mpm_smoke PRIVATE USE_OPENMP)
    if(MSVC)
        target_compile_options(mpm_smoke PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/openmp:llvm>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/openmp:llvm>
        )
    endif()
endif()

# CUDA sources if enabled
if(USE_CUDA)
    # Target Ada/Lovelace (RTX 40-series); fallback to PTX if unknown
    set(CMAKE_CUDA_ARCHITECTURES 89)
    target_sources(mpm_smoke PRIVATE
        src/gpu/mpm_kernels.cu
        src/gpu/smoke_kernels.cu
    )
    target_compile_definitions(mpm_smoke PRIVATE USE_CUDA)
    target_link_libraries(mpm_smoke PRIVATE CUDA::cudart)
endif()

# Copy shaders to build directory
file(COPY ${CMAKE_SOURCE_DIR}/shaders DESTINATION ${CMAKE_BINARY_DIR})

# Print configuration
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA support: ${USE_CUDA}")

