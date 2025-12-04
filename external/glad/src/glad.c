/*
    GLAD - OpenGL Loader Implementation
*/

#include <glad/glad.h>
#include <string.h>

PFNGLCLEARPROC glClear;
PFNGLCLEARCOLORPROC glClearColor;
PFNGLVIEWPORTPROC glViewport;
PFNGLENABLEPROC glEnable;
PFNGLDISABLEPROC glDisable;
PFNGLBLENDFUNCPROC glBlendFunc;
PFNGLDEPTHFUNCPROC glDepthFunc;
PFNGLCULLFACEPROC glCullFace;
PFNGLFRONTFACEPROC glFrontFace;
PFNGLPOLYGONMODEPROC glPolygonMode;
PFNGLGETSTRINGPROC glGetString;

PFNGLGENBUFFERSPROC glGenBuffers;
PFNGLDELETEBUFFERSPROC glDeleteBuffers;
PFNGLBINDBUFFERPROC glBindBuffer;
PFNGLBUFFERDATAPROC glBufferData;
PFNGLBUFFERSUBDATAPROC glBufferSubData;
PFNGLMAPBUFFERPROC glMapBuffer;
PFNGLUNMAPBUFFERPROC glUnmapBuffer;
PFNGLBINDBUFFERBASEPROC glBindBufferBase;

PFNGLGENVERTEXARRAYSPROC glGenVertexArrays;
PFNGLDELETEVERTEXARRAYSPROC glDeleteVertexArrays;
PFNGLBINDVERTEXARRAYPROC glBindVertexArray;
PFNGLENABLEVERTEXATTRIBARRAYPROC glEnableVertexAttribArray;
PFNGLDISABLEVERTEXATTRIBARRAYPROC glDisableVertexAttribArray;
PFNGLVERTEXATTRIBPOINTERPROC glVertexAttribPointer;

PFNGLDRAWARRAYSPROC glDrawArrays;
PFNGLDRAWELEMENTSPROC glDrawElements;

PFNGLCREATESHADERPROC glCreateShader;
PFNGLDELETESHADERPROC glDeleteShader;
PFNGLSHADERSOURCEPROC glShaderSource;
PFNGLCOMPILESHADERPROC glCompileShader;
PFNGLGETSHADERIVPROC glGetShaderiv;
PFNGLGETSHADERINFOLOGPROC glGetShaderInfoLog;
PFNGLCREATEPROGRAMPROC glCreateProgram;
PFNGLDELETEPROGRAMPROC glDeleteProgram;
PFNGLATTACHSHADERPROC glAttachShader;
PFNGLDETACHSHADERPROC glDetachShader;
PFNGLLINKPROGRAMPROC glLinkProgram;
PFNGLUSEPROGRAMPROC glUseProgram;
PFNGLGETPROGRAMIVPROC glGetProgramiv;
PFNGLGETPROGRAMINFOLOGPROC glGetProgramInfoLog;
PFNGLGETUNIFORMLOCATIONPROC glGetUniformLocation;

PFNGLUNIFORM1IPROC glUniform1i;
PFNGLUNIFORM1FPROC glUniform1f;
PFNGLUNIFORM2FPROC glUniform2f;
PFNGLUNIFORM3FPROC glUniform3f;
PFNGLUNIFORM4FPROC glUniform4f;
PFNGLUNIFORMMATRIX3FVPROC glUniformMatrix3fv;
PFNGLUNIFORMMATRIX4FVPROC glUniformMatrix4fv;

PFNGLGENTEXTURESPROC glGenTextures;
PFNGLDELETETEXTURESPROC glDeleteTextures;
PFNGLBINDTEXTUREPROC glBindTexture;
PFNGLACTIVETEXTUREPROC glActiveTexture;
PFNGLTEXIMAGE2DPROC glTexImage2D;
PFNGLTEXIMAGE3DPROC glTexImage3D;
PFNGLTEXSUBIMAGE3DPROC glTexSubImage3D;
PFNGLTEXPARAMETERIPROC glTexParameteri;
PFNGLGENERATEMIPMAPPROC glGenerateMipmap;

PFNGLGENFRAMEBUFFERSPROC glGenFramebuffers;
PFNGLDELETEFRAMEBUFFERSPROC glDeleteFramebuffers;
PFNGLBINDFRAMEBUFFERPROC glBindFramebuffer;
PFNGLFRAMEBUFFERTEXTURE2DPROC glFramebufferTexture2D;
PFNGLCHECKFRAMEBUFFERSTATUSPROC glCheckFramebufferStatus;
PFNGLGENRENDERBUFFERSPROC glGenRenderbuffers;
PFNGLDELETERENDERBUFFERSPROC glDeleteRenderbuffers;
PFNGLBINDRENDERBUFFERPROC glBindRenderbuffer;
PFNGLRENDERBUFFERSTORAGEPROC glRenderbufferStorage;
PFNGLFRAMEBUFFERRENDERBUFFERPROC glFramebufferRenderbuffer;

PFNGLDISPATCHCOMPUTEPROC glDispatchCompute;
PFNGLMEMORYBARRIERPROC glMemoryBarrier;

static void* get_proc(GLADloadproc load, const char* name) {
    void* result = load(name);
    return result;
}

int gladLoadGLLoader(GLADloadproc load) {
    if (load == NULL) return 0;
    
    glClear = (PFNGLCLEARPROC)get_proc(load, "glClear");
    glClearColor = (PFNGLCLEARCOLORPROC)get_proc(load, "glClearColor");
    glViewport = (PFNGLVIEWPORTPROC)get_proc(load, "glViewport");
    glEnable = (PFNGLENABLEPROC)get_proc(load, "glEnable");
    glDisable = (PFNGLDISABLEPROC)get_proc(load, "glDisable");
    glBlendFunc = (PFNGLBLENDFUNCPROC)get_proc(load, "glBlendFunc");
    glDepthFunc = (PFNGLDEPTHFUNCPROC)get_proc(load, "glDepthFunc");
    glCullFace = (PFNGLCULLFACEPROC)get_proc(load, "glCullFace");
    glFrontFace = (PFNGLFRONTFACEPROC)get_proc(load, "glFrontFace");
    glPolygonMode = (PFNGLPOLYGONMODEPROC)get_proc(load, "glPolygonMode");
    glGetString = (PFNGLGETSTRINGPROC)get_proc(load, "glGetString");
    
    glGenBuffers = (PFNGLGENBUFFERSPROC)get_proc(load, "glGenBuffers");
    glDeleteBuffers = (PFNGLDELETEBUFFERSPROC)get_proc(load, "glDeleteBuffers");
    glBindBuffer = (PFNGLBINDBUFFERPROC)get_proc(load, "glBindBuffer");
    glBufferData = (PFNGLBUFFERDATAPROC)get_proc(load, "glBufferData");
    glBufferSubData = (PFNGLBUFFERSUBDATAPROC)get_proc(load, "glBufferSubData");
    glMapBuffer = (PFNGLMAPBUFFERPROC)get_proc(load, "glMapBuffer");
    glUnmapBuffer = (PFNGLUNMAPBUFFERPROC)get_proc(load, "glUnmapBuffer");
    glBindBufferBase = (PFNGLBINDBUFFERBASEPROC)get_proc(load, "glBindBufferBase");
    
    glGenVertexArrays = (PFNGLGENVERTEXARRAYSPROC)get_proc(load, "glGenVertexArrays");
    glDeleteVertexArrays = (PFNGLDELETEVERTEXARRAYSPROC)get_proc(load, "glDeleteVertexArrays");
    glBindVertexArray = (PFNGLBINDVERTEXARRAYPROC)get_proc(load, "glBindVertexArray");
    glEnableVertexAttribArray = (PFNGLENABLEVERTEXATTRIBARRAYPROC)get_proc(load, "glEnableVertexAttribArray");
    glDisableVertexAttribArray = (PFNGLDISABLEVERTEXATTRIBARRAYPROC)get_proc(load, "glDisableVertexAttribArray");
    glVertexAttribPointer = (PFNGLVERTEXATTRIBPOINTERPROC)get_proc(load, "glVertexAttribPointer");
    
    glDrawArrays = (PFNGLDRAWARRAYSPROC)get_proc(load, "glDrawArrays");
    glDrawElements = (PFNGLDRAWELEMENTSPROC)get_proc(load, "glDrawElements");
    
    glCreateShader = (PFNGLCREATESHADERPROC)get_proc(load, "glCreateShader");
    glDeleteShader = (PFNGLDELETESHADERPROC)get_proc(load, "glDeleteShader");
    glShaderSource = (PFNGLSHADERSOURCEPROC)get_proc(load, "glShaderSource");
    glCompileShader = (PFNGLCOMPILESHADERPROC)get_proc(load, "glCompileShader");
    glGetShaderiv = (PFNGLGETSHADERIVPROC)get_proc(load, "glGetShaderiv");
    glGetShaderInfoLog = (PFNGLGETSHADERINFOLOGPROC)get_proc(load, "glGetShaderInfoLog");
    glCreateProgram = (PFNGLCREATEPROGRAMPROC)get_proc(load, "glCreateProgram");
    glDeleteProgram = (PFNGLDELETEPROGRAMPROC)get_proc(load, "glDeleteProgram");
    glAttachShader = (PFNGLATTACHSHADERPROC)get_proc(load, "glAttachShader");
    glDetachShader = (PFNGLDETACHSHADERPROC)get_proc(load, "glDetachShader");
    glLinkProgram = (PFNGLLINKPROGRAMPROC)get_proc(load, "glLinkProgram");
    glUseProgram = (PFNGLUSEPROGRAMPROC)get_proc(load, "glUseProgram");
    glGetProgramiv = (PFNGLGETPROGRAMIVPROC)get_proc(load, "glGetProgramiv");
    glGetProgramInfoLog = (PFNGLGETPROGRAMINFOLOGPROC)get_proc(load, "glGetProgramInfoLog");
    glGetUniformLocation = (PFNGLGETUNIFORMLOCATIONPROC)get_proc(load, "glGetUniformLocation");
    
    glUniform1i = (PFNGLUNIFORM1IPROC)get_proc(load, "glUniform1i");
    glUniform1f = (PFNGLUNIFORM1FPROC)get_proc(load, "glUniform1f");
    glUniform2f = (PFNGLUNIFORM2FPROC)get_proc(load, "glUniform2f");
    glUniform3f = (PFNGLUNIFORM3FPROC)get_proc(load, "glUniform3f");
    glUniform4f = (PFNGLUNIFORM4FPROC)get_proc(load, "glUniform4f");
    glUniformMatrix3fv = (PFNGLUNIFORMMATRIX3FVPROC)get_proc(load, "glUniformMatrix3fv");
    glUniformMatrix4fv = (PFNGLUNIFORMMATRIX4FVPROC)get_proc(load, "glUniformMatrix4fv");
    
    glGenTextures = (PFNGLGENTEXTURESPROC)get_proc(load, "glGenTextures");
    glDeleteTextures = (PFNGLDELETETEXTURESPROC)get_proc(load, "glDeleteTextures");
    glBindTexture = (PFNGLBINDTEXTUREPROC)get_proc(load, "glBindTexture");
    glActiveTexture = (PFNGLACTIVETEXTUREPROC)get_proc(load, "glActiveTexture");
    glTexImage2D = (PFNGLTEXIMAGE2DPROC)get_proc(load, "glTexImage2D");
    glTexImage3D = (PFNGLTEXIMAGE3DPROC)get_proc(load, "glTexImage3D");
    glTexSubImage3D = (PFNGLTEXSUBIMAGE3DPROC)get_proc(load, "glTexSubImage3D");
    glTexParameteri = (PFNGLTEXPARAMETERIPROC)get_proc(load, "glTexParameteri");
    glGenerateMipmap = (PFNGLGENERATEMIPMAPPROC)get_proc(load, "glGenerateMipmap");
    
    glGenFramebuffers = (PFNGLGENFRAMEBUFFERSPROC)get_proc(load, "glGenFramebuffers");
    glDeleteFramebuffers = (PFNGLDELETEFRAMEBUFFERSPROC)get_proc(load, "glDeleteFramebuffers");
    glBindFramebuffer = (PFNGLBINDFRAMEBUFFERPROC)get_proc(load, "glBindFramebuffer");
    glFramebufferTexture2D = (PFNGLFRAMEBUFFERTEXTURE2DPROC)get_proc(load, "glFramebufferTexture2D");
    glCheckFramebufferStatus = (PFNGLCHECKFRAMEBUFFERSTATUSPROC)get_proc(load, "glCheckFramebufferStatus");
    glGenRenderbuffers = (PFNGLGENRENDERBUFFERSPROC)get_proc(load, "glGenRenderbuffers");
    glDeleteRenderbuffers = (PFNGLDELETERENDERBUFFERSPROC)get_proc(load, "glDeleteRenderbuffers");
    glBindRenderbuffer = (PFNGLBINDRENDERBUFFERPROC)get_proc(load, "glBindRenderbuffer");
    glRenderbufferStorage = (PFNGLRENDERBUFFERSTORAGEPROC)get_proc(load, "glRenderbufferStorage");
    glFramebufferRenderbuffer = (PFNGLFRAMEBUFFERRENDERBUFFERPROC)get_proc(load, "glFramebufferRenderbuffer");
    
    glDispatchCompute = (PFNGLDISPATCHCOMPUTEPROC)get_proc(load, "glDispatchCompute");
    glMemoryBarrier = (PFNGLMEMORYBARRIERPROC)get_proc(load, "glMemoryBarrier");
    
    return glClear != NULL;
}

