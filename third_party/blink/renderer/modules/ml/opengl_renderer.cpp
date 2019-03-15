/*
 Copyright © 2018 Apple Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy of
 this software and associated documentation files (the "Software"), to deal in
 the Software without restriction, including without limitation the rights to
 use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 of the Software, and to permit persons to whom the Software is furnished to do
 so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
*/

#import "opengl_renderer.h"

#import <simd/simd.h>

#include "gpu/command_buffer/client/gles2_interface.h"

#include "third_party/blink/renderer/platform/wtf/text/wtf_string.h"

#include "third_party/blink/renderer/platform/wtf/text/string_utf8_adaptor.h"
#include "base/logging.h"
#include "fragment_shader.h"
#include "vertex_shader.h"


// #define BUFFER_OFFSET(i) ((char*)nullptr + (i))
namespace blink {

namespace {
// Indicies to which we will set vertex array attibutes
// See buildVAO and buildProgram
enum { POS_ATTRIB_IDX, TEXCOORD_ATTRIB_IDX };


GLuint buildVAO(gpu::gles2::GLES2Interface* gl) {
  typedef struct {
    vector_float4 position;
    packed_float2 texCoord;
  } AAPLVertex;

  static const AAPLVertex QuadVertices[] = {
      // x, y, z, w
      {{-1.0, -1.0, 0.0, 1.0}, {0.0, 0.0}}, {{1.0, -1.0, 0.0, 1.0}, {1.0, 0.0}},
      {{-1.0, 1.0, 0.0, 1.0}, {0.0, 1.0}},

      {{1.0, -1.0, 0.0, 1.0}, {1.0, 0.0}},  {{-1.0, 1.0, 0.0, 1.0}, {0.0, 1.0}},
      {{1.0, 1.0, 0.0, 1.0}, {1.0, 1.0}}};

  GLuint vaoName;

  gl->GenVertexArraysOES(1, &vaoName);
  gl->BindVertexArrayOES(vaoName);

  GLuint bufferName;

  gl->GenBuffers(1, &bufferName);
  gl->BindBuffer(GL_ARRAY_BUFFER, bufferName);

  gl->BufferData(GL_ARRAY_BUFFER, sizeof(QuadVertices), QuadVertices,
               GL_STATIC_DRAW);

  gl->EnableVertexAttribArray(POS_ATTRIB_IDX);

  GLuint stride = sizeof(AAPLVertex);
  GLuint positionOffset = offsetof(AAPLVertex, position);

  gl->VertexAttribPointer(POS_ATTRIB_IDX, 2, GL_FLOAT, GL_FALSE, stride,
                        reinterpret_cast<GLvoid*>(positionOffset));

  // Enable the position attribute for this VAO
  gl->EnableVertexAttribArray(TEXCOORD_ATTRIB_IDX);

  GLuint texCoordOffset = offsetof(AAPLVertex, texCoord);

  gl->VertexAttribPointer(TEXCOORD_ATTRIB_IDX, 2, GL_FLOAT, GL_FALSE, stride,
                        reinterpret_cast<GLvoid*>(texCoordOffset));

  gl->GetError();

  return vaoName;
}

// void destroyVAO(gpu::gles2::GLES2Interface* gl, GLuint vaoName) {
//   // Bind the VAO so we can get data from it
//   gl->BindVertexArrayOES(vaoName);

//   // For every possible attribute set in the VAO, delete the attached buffer
//   for (GLuint index = 0; index < 16; index++) {
//     GLuint bufName;
//     gl->GetVertexAttribiv(index, GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING,
//                         (GLint*)&bufName);
//     if (bufName) {
//       gl->DeleteBuffers(1, &bufName);
//     }
//   }

//   gl->DeleteVertexArraysOES(1, &vaoName);

//   gl->GetError();
// }

GLuint CompileShaderFromSource(gpu::gles2::GLES2Interface* gl,
                               const GLchar* source,
                               GLenum type) {
  GLuint shader = gl->CreateShader(type);
  GLint length = base::checked_cast<GLint>(strlen(source));
  gl->ShaderSource(shader, 1, &source, &length);
  gl->CompileShader(shader);
  GLint compile_status = 0;
  gl->GetShaderiv(shader, GL_COMPILE_STATUS, &compile_status);
  if (!compile_status) {
    GLint log_length = 0;
    gl->GetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_length);
    if (log_length) {
      std::unique_ptr<GLchar[]> log(new GLchar[log_length]);
      GLsizei returned_log_length = 0;
      gl->GetShaderInfoLog(shader, log_length, &returned_log_length, log.get());
      LOG(ERROR) << std::string(log.get(), returned_log_length);
    }
    gl->DeleteShader(shader);
    return 0;
  }
  return shader;
}

GLuint buildProgram(gpu::gles2::GLES2Interface* gl) {
  std::basic_string<GLchar> fragment_directives;

  std::basic_string<GLchar> vertex_header;
  vertex_header.append(
      "precision highp float;\n"
      "attribute vec4 inPosition;\n"
      "attribute vec2 inTexcoord;\n");

  std::basic_string<GLchar> shared_variables;
  shared_variables.append("varying vec2 varTexcoord;\n");

  std::basic_string<GLchar> vertex_program;
  vertex_program.append(
      "  gl_Position = inPosition;\n"
      "  varTexcoord = inTexcoord;\n");

  std::basic_string<GLchar> fragment_header;
  fragment_header.append(
      "precision mediump float;\n"
      // "varying vec2 varTexcoord;\n"
      "uniform sampler2D texture1;\n");

  std::basic_string<GLchar> fragment_program;
  fragment_program.append(
      "  vec4 Color  = texture2D(texture1, varTexcoord.st, 0.0);\n"
      "  gl_FragColor = ((1.0 - Color.w)) + (Color * Color.w);\n");

  vertex_program = vertex_header + shared_variables + "void main() {\n" +
                     vertex_program + "}\n";

  fragment_program = fragment_directives + fragment_header +
                     shared_variables + "void main() {\n" + fragment_program +
                     "}\n";

  const GLuint vertex_shader =
      CompileShaderFromSource(gl, vertex_program.c_str(), GL_VERTEX_SHADER);
  if (vertex_shader == 0)
    return 0;

  // Create a program object
  GLuint program = gl->CreateProgram();

  gl->BindAttribLocation(program, POS_ATTRIB_IDX, "inPosition");
  gl->BindAttribLocation(program, TEXCOORD_ATTRIB_IDX, "inTexcoord");

  gl->AttachShader(program, vertex_shader);
  gl->DeleteShader(vertex_shader);

  const GLuint fragment_shader =
      CompileShaderFromSource(gl, fragment_program.c_str(), GL_FRAGMENT_SHADER);
  if (fragment_shader == 0)
    return 0;
  gl->AttachShader(program, fragment_shader);
  // Delete the vertex shader since it is now attached to the program, which
  // will retain a reference to it
  gl->DeleteShader(fragment_shader);

  gl->LinkProgram(program);

  GLint link_status = 0;
  gl->GetProgramiv(program, GL_LINK_STATUS, &link_status);
  if (!link_status)
    return 0;

  gl->UseProgram(program);

  GLint location = gl->GetUniformLocation(program, "texture1");
  if (location < 0) {
    LOG(ERROR) << "Could not get sampler Uniform Index";
    return 0;
  }
  // Indicate that the diffuse texture will be bound to texture unit 1
  GLint unit = 1;
  gl->Uniform1i(location, unit);

  gl->GetError();

  return program;
}

} // namespace

OpenGLRenderer::OpenGLRenderer(gpu::gles2::GLES2Interface* gl) {

  _vertexArrayName = buildVAO(gl);

  _programName = buildProgram(gl);

  gl_ = gl;
}

OpenGLRenderer::~OpenGLRenderer() = default;

void OpenGLRenderer::Draw(GLuint frameBufferName,
              GLenum texTarget,
            GLuint texName) {
  gl_->BindFramebuffer(GL_FRAMEBUFFER, frameBufferName);

  gl_->ClearColor(0, 1, 0, 1);
  gl_->Clear(GL_COLOR_BUFFER_BIT);

  gl_->UseProgram(_programName);

  gl_->ActiveTexture(GL_TEXTURE1);
  gl_->BindTexture(texTarget, texName);

  gl_->BindVertexArrayOES(_vertexArrayName);

  gl_->DrawArrays(GL_TRIANGLES, 0, 6);

  gl_->GetError();
}

} //namspace blink
