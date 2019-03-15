// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/execution.h"

#import <OpenGL/gl3.h>

#include "gpu/GLES2/gl2extchromium.h"
#include "gpu/command_buffer/client/gles2_interface.h"
#include "gpu/command_buffer/client/gles2_lib.h"
#include "gpu/command_buffer/client/gpu_memory_buffer_manager.h"
#include "gpu/command_buffer/client/shared_image_interface.h"
#include "gpu/command_buffer/common/shared_image_usage.h"

#include "mojo/public/cpp/system/platform_handle.h"
#include "third_party/blink/renderer/modules/ml/opengl_renderer.h"
#include "third_party/blink/renderer/platform/graphics/gpu/drawing_buffer.h"
#include "ui/gfx/mojom/buffer_types.mojom-blink.h"

// #include "third_party/blink/renderer/modules/ml/vertex_shader.h"
#include "mojo/public/cpp/bindings/interface_ptr.h"
#include "services/ml/public/mojom/constants.mojom-blink.h"
#include "third_party/blink/renderer/core/dom/document.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"
#include "third_party/blink/renderer/platform/bindings/exception_code.h"
#include "third_party/blink/renderer/platform/wtf/functional.h"

namespace blink {

namespace {

uint32_t product(const WTF::Vector<uint32_t>& dims) {
  uint32_t prod = 1;

  for (wtf_size_t i = 0; i < dims.size(); ++i)
    prod *= dims[i];

  return prod;
}

uint32_t requiredSize(int32_t type, const WTF::Vector<uint32_t>& dimensions) {
  if (type == ml::mojom::blink::FLOAT32) {
    return sizeof(float);
  } else if (type == ml::mojom::blink::INT32) {
    return sizeof(int32_t);
  } else if (type == ml::mojom::blink::UINT32) {
    return sizeof(uint32_t);
  } else if (type == ml::mojom::blink::TENSOR_FLOAT32) {
    return product(dimensions) * sizeof(float);
  } else if (type == ml::mojom::blink::TENSOR_INT32) {
    return product(dimensions) * sizeof(int32_t);
  } else if (type == ml::mojom::blink::TENSOR_QUANT8_ASYMM) {
    return product(dimensions) * sizeof(int8_t);
  } else {
    NOTREACHED();
  }

  return 0;
}

}  // namespace

Execution::Execution(ml::mojom::blink::ExecutionInitParamsPtr init_params) {
  execution_.Bind(std::move(init_params->execution));
  execution_.set_connection_error_handler(
      WTF::Bind(&Execution::OnConnectionError, WrapWeakPersistent(this)));

  uint32_t total_length = 0;
  memory_ = std::move(init_params->memory);
  for (wtf_size_t i = 0; i < init_params->inputs.size(); ++i) {
    uint32_t offset = total_length;
    uint32_t length = requiredSize(init_params->inputs[i]->type,
                                   init_params->inputs[i]->dimensions);
    inputs_.push_back(std::make_unique<OperandInfo>(
        offset, length, memory_->MapAtOffset(length, offset)));
    total_length += length;
  }

  for (wtf_size_t i = 0; i < init_params->outputs.size(); ++i) {
    uint32_t offset = total_length;
    uint32_t length = requiredSize(init_params->outputs[i]->type,
                                   init_params->outputs[i]->dimensions);
    outputs_.push_back(std::make_unique<OperandInfo>(
        offset, length, memory_->MapAtOffset(length, offset)));
    total_length += length;
  }

  output_buffer_views_.resize(init_params->outputs.size());
}

Execution::~Execution() = default;

void Execution::setInput(uint32_t index,
                         MaybeShared<DOMArrayBufferView> data,
                         ExceptionState& exception_state) {
  if (index >= inputs_.size()) {
    exception_state.ThrowDOMException(DOMExceptionCode::kInvalidStateError,
                                      "Invalid index");
    return;
  }

  std::unique_ptr<OperandInfo>& info = inputs_.at(index);
  uint32_t length = data.View()->byteLength();
  if (info->length != length) {
    exception_state.ThrowDOMException(DOMExceptionCode::kInvalidStateError,
                                      "Invalid data");
    return;
  }

  memcpy(static_cast<void*>(info->mapping.get()), data.View()->BaseAddress(),
         length);
}

void Execution::setInput(uint32_t index,
                         WebGL2RenderingContext* context,
                         WebGLTexture* texture,
                         ExceptionState& exception_state) {
  if (index >= inputs_.size()) {
    exception_state.ThrowDOMException(DOMExceptionCode::kInvalidStateError,
                                      "Invalid index");
    return;
  }

  // std::unique_ptr<OperandInfo>& info = inputs_.at(index);
  // // 32 is the length of GLuint type, see "typedef unsigned int GLuint".
  // if (info->length < 32) {
  //   exception_state.ThrowDOMException(DOMExceptionCode::kInvalidStateError,
  //                                     "Invalid data");
  //   return;
  // }

  DrawingBuffer* drawing_buffer = context->GetDrawingBuffer();
  gpu::gles2::GLES2Interface* gl = drawing_buffer->ContextGL();
  WebGraphicsContext3DProvider* context_provider =
      drawing_buffer->ContextProvider();
  // {
  //   WTF::String vertSourceString(VERTEX_SHADER);
  //   WebGLShader* shader = context->createShader(GL_VERTEX_SHADER);
  //   context->shaderSource(shader, vertSourceString);

  //   context->compileShader(shader);

  //   LOG(ERROR) << "======== Vtx Shader compile log:"
  //              << context->getShaderInfoLog(shader);
  // }

  gpu::SharedImageInterface* sii = context_provider->SharedImageInterface();
  gpu::GpuMemoryBufferManager* gpu_memory_buffer_manager =
      Platform::Current()->GetGpuMemoryBufferManager();

  gpu::Mailbox mailbox;
  GLuint texture_id = 0;
  std::unique_ptr<gfx::GpuMemoryBuffer> gpu_memory_buffer;
  // uint32_t n, width, height, channels;
  // if (!GetOperandInfo(operand_info_inputs_[index], n, width, height,
  // channels)) {
  //   return;
  // }
  IntSize size(2, 2);
  gpu_memory_buffer = gpu_memory_buffer_manager->CreateGpuMemoryBuffer(
      gfx::Size(size), gfx::BufferFormat::RGBA_8888, gfx::BufferUsage::SCANOUT,
      gpu::kNullSurfaceHandle);
  if (!gpu_memory_buffer)
    return;

  mailbox = sii->CreateSharedImage(
      gpu_memory_buffer.get(), gpu_memory_buffer_manager,
      gfx::ColorSpace::CreateSRGB(),
      gpu::SHARED_IMAGE_USAGE_GLES2 | gpu::SHARED_IMAGE_USAGE_GLES2_FRAMEBUFFER_HINT |
          gpu::SHARED_IMAGE_USAGE_DISPLAY |
          gpu::SHARED_IMAGE_USAGE_SCANOUT);
  // gl::GLImplementation impl = gl::GetGLImplementation();
  // gl::SetGLImplementation(gl::kGLImplementationNone);

  // Import the allocated SharedImage into GL.
  gpu::SyncToken sync_token = sii->GenUnverifiedSyncToken();
  gl->WaitSyncTokenCHROMIUM(sync_token.GetConstData());
  texture_id = gl->CreateAndTexStorage2DSharedImageCHROMIUM(mailbox.name);

  // gl->DeleteTextures(1, &texture_id);
  // gl_->DeleteFramebuffers(1, &framebuffer);

  // gl->BindTexture(GC3D_TEXTURE_RECTANGLE_ARB, texture_id);
  // gl->BeginSharedImageAccessDirectCHROMIUM(
  //     texture_id, GL_SHARED_IMAGE_ACCESS_MODE_READWRITE_CHROMIUM);
  // GLuint image_id =
  //     gl->CreateImageCHROMIUM(gpu_memory_buffer->AsClientBuffer(), 10, 10, GL_RGBA);
  // LOG(ERROR) << "=======image_id " <<image_id;
  // gl->BindTexImage2DCHROMIUM(GC3D_TEXTURE_RECTANGLE_ARB, image_id);
  if (true) {
    // gl->Viewport(0, 0, 10, 10);
    // gl->ClearColor(1, 1, 0, 1);
    // gl->Clear(GL_COLOR_BUFFER_BIT);

    // gl->EndSharedImageAccessDirectCHROMIUM(texture_id);
    // gl->DeleteTextures(1, &texture_id);

    // gl->GenUnverifiedSyncTokenCHROMIUM(sync_token.GetData());
  }

  if (true) {

    // uint32_t result;
    // gl->ReadPixels(0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, &result);
    // // The expected result is 4278255360.

    GLuint resolve_fbo, resolve_tex;
    gl->GenTextures(1, &resolve_tex);
    gl->BindTexture(GL_TEXTURE_2D, resolve_tex);
    gl->ClearColor(0, 1, 0, 1);
    gl->Clear(GL_COLOR_BUFFER_BIT);
    GLuint sample_fbo = 0;
    gl->GenFramebuffers(1, &sample_fbo);
    gl->BindFramebuffer(GL_FRAMEBUFFER, sample_fbo);
    gl->FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             GL_TEXTURE_2D, resolve_tex, 0);

    // float dirty_color[16] = {1.31f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f, 1.9f, 2.0f,
    //                          2.3f,  2.4f, 2.5f, 2.6f, 2.7f, 2.8f, 2.9f, 3.0f};
    // gl->TexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 2, 2, 0, GL_RGBA, GL_UNSIGNED_BYTE,
    //                dirty_color);

    unsigned char red[4][3] = {
      {231, 0, 0},
      {212, 0, 0},
      {213, 0, 0},
      {214, 0, 0}
    };
    gl->TexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 2, 2, 0, GL_RGB,
      GL_UNSIGNED_BYTE, red);

    gl->Flush();
    // unsigned char color[16] = {0};
    unsigned char color[16] = {0};
    gl->ReadPixels(0, 0, 2, 2, GL_RGB, GL_UNSIGNED_BYTE, color);
    // gl->ReadPixels(0, 0, 2, 2, GL_RGBA, GL_FLOAT, color);
    for (size_t i = 0; i < 16; i++) {
      LOG(ERROR) << "======the input data = " << static_cast<int>(color[i]);
    }

    // Create another FBO to resolve the multisample buffer into.
    gl->BindTexture(GC3D_TEXTURE_RECTANGLE_ARB, texture_id);
    gl->BeginSharedImageAccessDirectCHROMIUM(
        texture_id, GL_SHARED_IMAGE_ACCESS_MODE_READWRITE_CHROMIUM);
    // gl->TexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 4, 4, 0, GL_RGBA,
    //              GL_FLOAT, nullptr);
    gl->TexParameteri(GC3D_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    gl->TexParameteri(GC3D_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    gl->TexParameteri(GC3D_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    gl->TexParameteri(GC3D_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    gl->GenFramebuffers(1, &resolve_fbo);
    gl->BindFramebuffer(GL_FRAMEBUFFER, resolve_fbo);
    gl->FramebufferTexture2D(GL_FRAMEBUFFER,
                           GL_COLOR_ATTACHMENT0,
                           GC3D_TEXTURE_RECTANGLE_ARB,
                           texture_id,
                           0);
    // Resolve.
    gl->BindFramebuffer(GL_READ_FRAMEBUFFER, sample_fbo);
    gl->BindFramebuffer(GL_DRAW_FRAMEBUFFER, resolve_fbo);
    gl->BlitFramebufferCHROMIUM(0,
                              0,
                              2,
                              2,
                              0,
                              0,
                              2,
                              2,
                              GL_COLOR_BUFFER_BIT,
                              GL_NEAREST);
    gl->ShallowFlushCHROMIUM();
    gl->DeleteFramebuffers(1, &sample_fbo);

    gl->BindTexture(GC3D_TEXTURE_RECTANGLE_ARB, texture_id);
    gl->BindFramebuffer(GL_FRAMEBUFFER, resolve_fbo);
    gl->FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             GC3D_TEXTURE_RECTANGLE_ARB, texture_id, 0);

    // gl->ClearColor(33.0 / 255.0, 44.0 / 255.0, 55.0 / 255.0, 66.0 / 255.0);
    // gl->Clear(GL_COLOR_BUFFER_BIT);
    gl->Flush();

    GLint format = 0;
    GLint type = 0;
    gl->GetIntegerv(GL_IMPLEMENTATION_COLOR_READ_FORMAT, &format);
    gl->GetIntegerv(GL_IMPLEMENTATION_COLOR_READ_TYPE, &type);
    LOG(ERROR) << format << " " << type;
    unsigned char color_new[4 * 4] = {0};
    gl->ReadPixels(0, 0, 2, 2, format, type, color_new);
    for (size_t i = 0; i < 4 * 4; i++) {
      LOG(ERROR) << "======the 1111input data = "
                 << static_cast<int>(color[i]);
    }
    // gl->CopyTextureCHROMIUM(ObjectOrZero(texture), 0,
    //                       GC3D_TEXTURE_RECTANGLE_ARB, texture_id, 0, format,
    //                       type, false, false, false);
  }

  // gl->CopyTextureCHROMIUM(ObjectOrZero(texture), 0,
  //                         GC3D_TEXTURE_RECTANGLE_ARB, texture_id, 0, GL_RGBA,
  //                         GL_FLOAT, false, true, false);

  // gl->CopySubTextureCHROMIUM(
  //     ObjectOrZero(texture), 0, GC3D_TEXTURE_RECTANGLE_ARB, texture_id, 0,
  //     0, 0, 
  //     0, 0, 4, 4, GL_FALSE, GL_FALSE,
  //     GL_FALSE);
  // gl->ShallowFlushCHROMIUM();
  // gl->EndSharedImageAccessDirectCHROMIUM(ObjectOrZero(texture));
  // dst_gl->DeleteTextures(1, &src_texture);

  //     {
  //   // GLuint tex = 1;
  //   // gl->GenTextures(1, &tex);
  //   // gl->BindTexture(GL_TEXTURE_2D, tex);
  //   // gl->TexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, 256, 256, 0, GL_RGB,
  //   GL_HALF_FLOAT, nullptr);

  //   GLuint fbo = 1;
  //     gl->GenFramebuffers(1, &fbo);
  //   gl->BindFramebuffer(GL_FRAMEBUFFER, fbo);
  //   gl->FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
  //   GC3D_TEXTURE_RECTANGLE_ARB, texture_id, 0);

  //   // const float clear_color[4] = { 1.0f, 32.0f, 0.5f, 1.0f };
  //   // gl->ClearColor(clear_color[0], clear_color[1], clear_color[2], 1.0f);
  //   // gl->Clear(GL_COLOR_BUFFER_BIT);

  //   // uint16_t pixel[3] = { 0x1234, 0x3F80, 0xAAAA };
  //   // GLint x = 6;
  //   // GLint y = 3;
  //   // gl->TexSubImage2D(GL_TEXTURE_2D, 0, x, y, 1, 1, GL_RGB, GL_HALF_FLOAT,
  //   pixel);

  //   // This relies on GL_HALF_FLOAT being a valid type for read-back,
  //   // which isn't guaranteed by the spec but is supported by SwiftShader.
  //   uint16_t color[3] = { 0, 0, 0 };
  //   gl->ReadPixels(0, 0, 1, 1, GL_RGB, GL_FLOAT, &color);
  //     LOG(ERROR) << "====pixel " << color[0] << " " << color[1] << " " <<
  //     color[2];
  // }
  GLuint temp_texture;
  if (false) {
    gl->GenTextures(1, &temp_texture);
    gl->BindTexture(GL_TEXTURE_2D, temp_texture);
    gl->TexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 10, 10, 0, GL_RGBA, GL_FLOAT,
                   NULL);
  }

  // if (true) {
  //   GLuint fbo = 0;
  //   // Create a "frameBufferObject" with the interop texture as the color
  //   // buffer.
  //   gl->GenFramebuffers(1, &fbo);
  //   gl->BindFramebuffer(GL_FRAMEBUFFER, fbo);
  //   // macOS CVPixelBuffer textures created as rectangle textures.
  //   gl->FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
  //                            GC3D_TEXTURE_RECTANGLE_ARB, texture_id, 0);

  //   // Initialize the openGL render.
  //   OpenGLRenderer* render = new OpenGLRenderer(gl);

  //   // Set initial OpenGL rendering size to interop texture size.
  //   gl->Viewport(0, 0, 10, 10);

  //   // Execute OpenGL renderer draw routine to build.
  //   render->Draw(fbo, GL_TEXTURE_2D, ObjectOrZero(texture));

  //   // When rendering to a CVPixelBuffer with OpenGL, call glFlush to ensure
  //   // OpenGL commands are excuted on the pixel buffer before Metal reads the
  //   // buffer.
  //   gl->Flush();
  // }
  if (false) {
    gl->BindTexture(GC3D_TEXTURE_RECTANGLE_ARB, texture_id);

    gl->TexParameteri(GC3D_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    gl->TexParameteri(GC3D_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // gl->TexParameteri(GC3D_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    // gl->TexParameteri(GC3D_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    GLuint fbo = 0;
    gl->GenFramebuffers(1, &fbo);
    gl->BindFramebuffer(GL_FRAMEBUFFER, fbo);
    gl->FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             GL_TEXTURE_RECTANGLE, texture_id, 0);
    gl->Viewport(0, 0, 10, 10);
    gl->ClearColor(33.0 / 255.0, 44.0 / 255.0, 55.0 / 255.0, 66.0 / 255.0);
    gl->Clear(GL_COLOR_BUFFER_BIT);
    gl->Flush();

    char color[4 * 4] = {0};
    GLint format = 0;
    GLint type = 0;
    gl->GetIntegerv(GL_IMPLEMENTATION_COLOR_READ_FORMAT, &format);
    gl->GetIntegerv(GL_IMPLEMENTATION_COLOR_READ_TYPE, &type);
    gl->ReadPixels(0, 0, 2, 2, format, type, &color);
    for (size_t i = 0; i < 4 * 4; i++) {
      LOG(ERROR) << "======the data after clear color = "
                 << static_cast<int>(
                        (reinterpret_cast<unsigned char*>(color))[i]);
    }
  }

  // gfx::GpuMemoryBufferHandle handle = gpu_memory_buffer->CloneHandle();
  // auto buffer_handle = gfx::mojom::blink::GpuMemoryBufferHandle::New();
  // buffer_handle->id = gfx::mojom::blink::GpuMemoryBufferId::New(handle.id.id);
  // buffer_handle->offset = handle.offset;
  // buffer_handle->stride = handle.stride;
  // buffer_handle->platform_handle =
  //     gfx::mojom::blink::GpuMemoryBufferPlatformHandle::NewMachPort(
  //         mojo::WrapMachPort(handle.mach_port.get()));

  execution_->SetGpuMemoryBufferHandle(index, gpu_memory_buffer->CloneHandle());
}

void Execution::setOutput(uint32_t index,
                          WebGL2RenderingContext* context,
                          WebGLTexture* texture,
                          ExceptionState& exception_state) {
}

void Execution::setOutput(uint32_t index,
                          MaybeShared<DOMArrayBufferView> data,
                          ExceptionState& exception_state) {
  if (index >= output_buffer_views_.size()) {
    exception_state.ThrowDOMException(DOMExceptionCode::kInvalidStateError,
                                      "Invalid index");
    return;
  }

  std::unique_ptr<OperandInfo>& info = outputs_.at(index);
  uint32_t length = data.View()->byteLength();
  if (info->length != length) {
    exception_state.ThrowDOMException(DOMExceptionCode::kInvalidStateError,
                                      "Invalid data");
    return;
  }

  output_buffer_views_[index] = data.View();
}

ScriptPromise Execution::startCompute(ScriptState* script_state) {
  auto* resolver = MakeGarbageCollected<ScriptPromiseResolver>(script_state);
  ScriptPromise promise = resolver->Promise();
  if (!execution_) {
    resolver->Reject(MakeGarbageCollected<DOMException>(
        DOMExceptionCode::kNotSupportedError,
        "Neural Network service unavailable."));
    return promise;
  }

  requests_.insert(resolver);

  execution_->StartCompute(WTF::Bind(&Execution::OnStartCompute,
                                     WrapPersistent(this),
                                     WrapPersistent(resolver)));
  return promise;
}

void Execution::OnStartCompute(ScriptPromiseResolver* resolver,
                               int32_t result_code) {
  DCHECK(requests_.Contains(resolver));
  requests_.erase(resolver);

  if (result_code != ml::mojom::blink::NOT_ERROR) {
    return resolver->Reject(MakeGarbageCollected<DOMException>(
        DOMExceptionCode::kInvalidStateError,
        "startCompute fails " + String::Number(result_code)));
  }

  for (wtf_size_t i = 0; i < outputs_.size(); ++i) {
    DOMArrayBufferView* view = output_buffer_views_.at(i);
    if (view) {
      uint32_t length = view->byteLength();
      std::unique_ptr<OperandInfo>& info = outputs_.at(i);
      memcpy(view->BaseAddress(), static_cast<const void*>(info->mapping.get()),
             length);
    }
  }
  resolver->Resolve(result_code);
}

void Execution::OnResultCode(ScriptPromiseResolver* resolver,
                             const String& operation_name,
                             int32_t result_code) {
  DCHECK(requests_.Contains(resolver));
  requests_.erase(resolver);

  if (result_code != ml::mojom::blink::NOT_ERROR) {
    return resolver->Reject(MakeGarbageCollected<DOMException>(
        DOMExceptionCode::kInvalidStateError,
        operation_name + "fails: " + String::Number(result_code)));
  }

  resolver->Resolve(result_code);
}

void Execution::Trace(blink::Visitor* visitor) {
  visitor->Trace(requests_);
  visitor->Trace(output_buffer_views_);
  ScriptWrappable::Trace(visitor);
}

void Execution::OnConnectionError() {
  for (const auto& request : requests_) {
    request->Reject(MakeGarbageCollected<DOMException>(
        DOMExceptionCode::kNotSupportedError, "Execution is not implemented."));
  }

  requests_.clear();
  execution_.reset();
}

}  // namespace blink
