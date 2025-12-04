#include "Shader.h"
#include <glm/gtc/type_ptr.hpp>
#include <fstream>
#include <sstream>
#include <iostream>

Shader::~Shader() {
    if (program != 0) {
        glDeleteProgram(program);
    }
}

Shader::Shader(Shader&& other) noexcept
    : program(other.program)
    , uniformCache(std::move(other.uniformCache))
{
    other.program = 0;
}

Shader& Shader::operator=(Shader&& other) noexcept {
    if (this != &other) {
        if (program != 0) {
            glDeleteProgram(program);
        }
        program = other.program;
        uniformCache = std::move(other.uniformCache);
        other.program = 0;
    }
    return *this;
}

bool Shader::loadFromFiles(const std::string& vertexPath, const std::string& fragmentPath) {
    std::string vertexSrc = readFile(vertexPath);
    std::string fragmentSrc = readFile(fragmentPath);
    
    if (vertexSrc.empty() || fragmentSrc.empty()) {
        std::cerr << "Failed to read shader files: " << vertexPath << ", " << fragmentPath << std::endl;
        return false;
    }
    
    return loadFromSource(vertexSrc, fragmentSrc);
}

bool Shader::loadComputeFromFile(const std::string& computePath) {
    std::string computeSrc = readFile(computePath);
    if (computeSrc.empty()) {
        std::cerr << "Failed to read compute shader: " << computePath << std::endl;
        return false;
    }
    return loadComputeFromSource(computeSrc);
}

bool Shader::loadFromSource(const std::string& vertexSrc, const std::string& fragmentSrc) {
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexSrc);
    if (vertexShader == 0) return false;
    
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentSrc);
    if (fragmentShader == 0) {
        glDeleteShader(vertexShader);
        return false;
    }
    
    // Create and link program
    program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    
    bool success = linkProgram(program);
    
    // Cleanup shaders (they're linked now)
    glDetachShader(program, vertexShader);
    glDetachShader(program, fragmentShader);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    if (!success) {
        glDeleteProgram(program);
        program = 0;
    }
    
    uniformCache.clear();
    return success;
}

bool Shader::loadComputeFromSource(const std::string& computeSrc) {
    GLuint computeShader = compileShader(GL_COMPUTE_SHADER, computeSrc);
    if (computeShader == 0) return false;
    
    program = glCreateProgram();
    glAttachShader(program, computeShader);
    
    bool success = linkProgram(program);
    
    glDetachShader(program, computeShader);
    glDeleteShader(computeShader);
    
    if (!success) {
        glDeleteProgram(program);
        program = 0;
    }
    
    uniformCache.clear();
    return success;
}

void Shader::use() const {
    glUseProgram(program);
}

void Shader::dispatch(GLuint groupsX, GLuint groupsY, GLuint groupsZ) const {
    use();
    glDispatchCompute(groupsX, groupsY, groupsZ);
}

GLint Shader::getUniformLocation(const std::string& name) {
    auto it = uniformCache.find(name);
    if (it != uniformCache.end()) {
        return it->second;
    }
    
    GLint location = glGetUniformLocation(program, name.c_str());
    uniformCache[name] = location;
    return location;
}

void Shader::setInt(const std::string& name, int value) {
    glUniform1i(getUniformLocation(name), value);
}

void Shader::setFloat(const std::string& name, float value) {
    glUniform1f(getUniformLocation(name), value);
}

void Shader::setVec2(const std::string& name, const glm::vec2& value) {
    glUniform2f(getUniformLocation(name), value.x, value.y);
}

void Shader::setVec3(const std::string& name, const glm::vec3& value) {
    glUniform3f(getUniformLocation(name), value.x, value.y, value.z);
}

void Shader::setVec4(const std::string& name, const glm::vec4& value) {
    glUniform4f(getUniformLocation(name), value.x, value.y, value.z, value.w);
}

void Shader::setMat3(const std::string& name, const glm::mat3& value) {
    glUniformMatrix3fv(getUniformLocation(name), 1, GL_FALSE, glm::value_ptr(value));
}

void Shader::setMat4(const std::string& name, const glm::mat4& value) {
    glUniformMatrix4fv(getUniformLocation(name), 1, GL_FALSE, glm::value_ptr(value));
}

GLuint Shader::compileShader(GLenum type, const std::string& source) {
    GLuint shader = glCreateShader(type);
    const char* src = source.c_str();
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    
    if (!success) {
        GLint logLength;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);
        std::string infoLog(logLength, ' ');
        glGetShaderInfoLog(shader, logLength, nullptr, infoLog.data());
        
        const char* typeName = (type == GL_VERTEX_SHADER) ? "vertex" :
                               (type == GL_FRAGMENT_SHADER) ? "fragment" :
                               (type == GL_COMPUTE_SHADER) ? "compute" : "unknown";
        std::cerr << "Shader compilation failed (" << typeName << "):\n" << infoLog << std::endl;
        
        glDeleteShader(shader);
        return 0;
    }
    
    return shader;
}

bool Shader::linkProgram(GLuint program) {
    glLinkProgram(program);
    
    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    
    if (!success) {
        GLint logLength;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);
        std::string infoLog(logLength, ' ');
        glGetProgramInfoLog(program, logLength, nullptr, infoLog.data());
        
        std::cerr << "Shader program linking failed:\n" << infoLog << std::endl;
        return false;
    }
    
    return true;
}

std::string Shader::readFile(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Could not open file: " << path << std::endl;
        return "";
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

