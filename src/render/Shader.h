#pragma once

#ifndef GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_NONE
#endif
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <string>
#include <unordered_map>

/**
 * OpenGL shader program wrapper
 * Supports vertex + fragment shaders, or compute shaders
 */
class Shader {
public:
    Shader() = default;
    ~Shader();
    
    // Non-copyable
    Shader(const Shader&) = delete;
    Shader& operator=(const Shader&) = delete;
    
    // Movable
    Shader(Shader&& other) noexcept;
    Shader& operator=(Shader&& other) noexcept;
    
    // Load from files
    bool loadFromFiles(const std::string& vertexPath, const std::string& fragmentPath);
    bool loadComputeFromFile(const std::string& computePath);
    
    // Load from source strings
    bool loadFromSource(const std::string& vertexSrc, const std::string& fragmentSrc);
    bool loadComputeFromSource(const std::string& computeSrc);
    
    // Use this shader
    void use() const;
    
    // Dispatch compute shader
    void dispatch(GLuint groupsX, GLuint groupsY, GLuint groupsZ) const;
    
    // Uniform setters
    void setInt(const std::string& name, int value);
    void setFloat(const std::string& name, float value);
    void setVec2(const std::string& name, const glm::vec2& value);
    void setVec3(const std::string& name, const glm::vec3& value);
    void setVec4(const std::string& name, const glm::vec4& value);
    void setMat3(const std::string& name, const glm::mat3& value);
    void setMat4(const std::string& name, const glm::mat4& value);
    
    GLuint getProgram() const { return program; }
    bool isValid() const { return program != 0; }
    
private:
    GLuint program = 0;
    std::unordered_map<std::string, GLint> uniformCache;
    
    GLint getUniformLocation(const std::string& name);
    static GLuint compileShader(GLenum type, const std::string& source);
    static bool linkProgram(GLuint program);
    static std::string readFile(const std::string& path);
};

