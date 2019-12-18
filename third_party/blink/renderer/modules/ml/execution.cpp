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
                         uint32_t width,
                         uint32_t height,
                         ExceptionState& exception_state) {
  DrawingBuffer* drawing_buffer = context->GetDrawingBuffer();
  gpu::gles2::GLES2Interface* gl = drawing_buffer->ContextGL();
  WebGraphicsContext3DProvider* context_provider =
      drawing_buffer->ContextProvider();
  gpu::SharedImageInterface* sii = context_provider->SharedImageInterface();
  gpu::GpuMemoryBufferManager* gpu_memory_buffer_manager =
      Platform::Current()->GetGpuMemoryBufferManager();
  std::unique_ptr<gfx::GpuMemoryBuffer> gpu_memory_buffer;
  gpu_memory_buffer = gpu_memory_buffer_manager->CreateGpuMemoryBuffer(
      gfx::Size(width, height), gfx::BufferFormat::RGBA_F16,
      gfx::BufferUsage::SCANOUT, gpu::kNullSurfaceHandle);
  if (!gpu_memory_buffer)
    return;

  gpu::Mailbox mailbox = sii->CreateSharedImage(
      gpu_memory_buffer.get(), gpu_memory_buffer_manager,
      gfx::ColorSpace::CreateSRGB(),
      gpu::SHARED_IMAGE_USAGE_GLES2 |
          gpu::SHARED_IMAGE_USAGE_GLES2_FRAMEBUFFER_HINT |
          gpu::SHARED_IMAGE_USAGE_DISPLAY | gpu::SHARED_IMAGE_USAGE_SCANOUT);

  // Import the allocated SharedImage into GL.
  gpu::SyncToken sync_token = sii->GenUnverifiedSyncToken();
  gl->WaitSyncTokenCHROMIUM(sync_token.GetConstData());
  GLuint texture_id =
      gl->CreateAndTexStorage2DSharedImageCHROMIUM(mailbox.name);

  gl->BindTexture(GL_TEXTURE_2D, ObjectOrZero(texture));
  GLuint sample_fbo = 0;
  gl->GenFramebuffers(1, &sample_fbo);
  gl->BindFramebuffer(GL_FRAMEBUFFER, sample_fbo);
  gl->FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                           ObjectOrZero(texture), 0);

  // float color[16] = {0};
  // gl->ReadPixels(0, 0, 2, 2, GL_RGBA, GL_FLOAT, color);
  // for (size_t i = 0; i < 16; i++) {
  //   LOG(ERROR) << "======the input data = " << color[i];
  // }

  // Create another FBO to resolve the multisample buffer into.
  gl->BindTexture(GC3D_TEXTURE_RECTANGLE_ARB, texture_id);
  gl->BeginSharedImageAccessDirectCHROMIUM(
      texture_id, GL_SHARED_IMAGE_ACCESS_MODE_READWRITE_CHROMIUM);
  gl->TexParameteri(GC3D_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER,
                    GL_LINEAR);
  gl->TexParameteri(GC3D_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER,
                    GL_LINEAR);
  gl->TexParameteri(GC3D_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S,
                    GL_CLAMP_TO_EDGE);
  gl->TexParameteri(GC3D_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T,
                    GL_CLAMP_TO_EDGE);
  GLuint resolve_fbo;
  gl->GenFramebuffers(1, &resolve_fbo);
  gl->BindFramebuffer(GL_FRAMEBUFFER, resolve_fbo);
  gl->FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GC3D_TEXTURE_RECTANGLE_ARB, texture_id, 0);
  // Resolve.
  gl->BindFramebuffer(GL_READ_FRAMEBUFFER_ANGLE, sample_fbo);
  gl->BindFramebuffer(GL_DRAW_FRAMEBUFFER_ANGLE, resolve_fbo);
  gl->Disable(GL_SCISSOR_TEST);
  gl->BlitFramebufferCHROMIUM(0, 0, width, height, 0, 0, width, height,
                              GL_COLOR_BUFFER_BIT, GL_NEAREST);
  gl->ShallowFlushCHROMIUM();
  // gl->DeleteFramebuffers(1, &sample_fbo);

  gl->BindTexture(GC3D_TEXTURE_RECTANGLE_ARB, texture_id);
  gl->BindFramebuffer(GL_FRAMEBUFFER, resolve_fbo);
  gl->FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GC3D_TEXTURE_RECTANGLE_ARB, texture_id, 0);

  // float copied_color[16] = {0};
  // gl->ReadPixels(0, 0, 2, 2, GL_RGBA, GL_FLOAT, copied_color);
  // for (size_t i = 0; i < 16; i++) {
  //   LOG(ERROR) << "======the copied data = " << copied_color[i];
  // }

  execution_->SetGpuMemoryBufferHandle(index, gpu_memory_buffer->CloneHandle());
  // gl->EndSharedImageAccessDirectCHROMIUM(texture_id);
  // gl->DeleteTextures(1, &texture_id);
}

void Execution::setOutput(uint32_t index,
                          WebGL2RenderingContext* context,
                          WebGLTexture* texture,
                          uint32_t width,
                          uint32_t height,
                          ExceptionState& exception_state) {
  drawing_buffer_ = context->GetDrawingBuffer();
  output_texture_ = ObjectOrZero(texture);
  output_texture_width_ = width;
  output_texture_height_ = height;

  gpu::GpuMemoryBufferManager* gpu_memory_buffer_manager =
      Platform::Current()->GetGpuMemoryBufferManager();
  output_gpu_memory_buffer_ = gpu_memory_buffer_manager->CreateGpuMemoryBuffer(
      gfx::Size(width, height), gfx::BufferFormat::RGBA_F16,
      gfx::BufferUsage::SCANOUT, gpu::kNullSurfaceHandle);

  execution_->SetGpuMemoryBufferHandle(
      index, output_gpu_memory_buffer_->CloneHandle());
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
    } else {
      WebGraphicsContext3DProvider* context_provider =
          drawing_buffer_->ContextProvider();
      gpu::SharedImageInterface* sii = context_provider->SharedImageInterface();
      gpu::GpuMemoryBufferManager* gpu_memory_buffer_manager =
          Platform::Current()->GetGpuMemoryBufferManager();
      gpu::Mailbox mailbox = sii->CreateSharedImage(
          output_gpu_memory_buffer_.get(), gpu_memory_buffer_manager,
          gfx::ColorSpace::CreateSRGB(),
          gpu::SHARED_IMAGE_USAGE_GLES2 |
              gpu::SHARED_IMAGE_USAGE_GLES2_FRAMEBUFFER_HINT |
              gpu::SHARED_IMAGE_USAGE_DISPLAY |
              gpu::SHARED_IMAGE_USAGE_SCANOUT);

      // Import the allocated SharedImage into GL.
      gpu::SyncToken sync_token = sii->GenUnverifiedSyncToken();

      gpu::gles2::GLES2Interface* gl = drawing_buffer_->ContextGL();
      gl->WaitSyncTokenCHROMIUM(sync_token.GetConstData());
      GLuint texture_id =
          gl->CreateAndTexStorage2DSharedImageCHROMIUM(mailbox.name);

      // Create another FBO to resolve the multisample buffer into.
      gl->BindTexture(GC3D_TEXTURE_RECTANGLE_ARB, texture_id);
      gl->BeginSharedImageAccessDirectCHROMIUM(
          texture_id, GL_SHARED_IMAGE_ACCESS_MODE_READWRITE_CHROMIUM);
      gl->TexParameteri(GC3D_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER,
                        GL_LINEAR);
      gl->TexParameteri(GC3D_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER,
                        GL_LINEAR);
      gl->TexParameteri(GC3D_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S,
                        GL_CLAMP_TO_EDGE);
      gl->TexParameteri(GC3D_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T,
                        GL_CLAMP_TO_EDGE);
      GLuint sample_fbo;
      gl->GenFramebuffers(1, &sample_fbo);
      gl->BindFramebuffer(GL_FRAMEBUFFER, sample_fbo);
      gl->FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GC3D_TEXTURE_RECTANGLE_ARB, texture_id, 0);
      // float color[16] = {0};
      // gl->ReadPixels(0, 0, 2, 2, GL_RGBA, GL_FLOAT, color);
      // for (size_t i = 0; i < 16; i++) {
      //   LOG(ERROR) << "======the input data = " << color[i];
      // }

      gl->BindTexture(GL_TEXTURE_2D, output_texture_);
      GLuint resolve_fbo;
      gl->GenFramebuffers(1, &resolve_fbo);
      gl->BindFramebuffer(GL_FRAMEBUFFER, resolve_fbo);
      gl->FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, output_texture_, 0);

      // Resolve.
      gl->BindFramebuffer(GL_READ_FRAMEBUFFER_ANGLE, sample_fbo);
      gl->BindFramebuffer(GL_DRAW_FRAMEBUFFER, resolve_fbo);
      // gl->Disable(GL_SCISSOR_TEST);
      gl->BlitFramebufferCHROMIUM(0, 0, output_texture_width_,
                                  output_texture_height_, 0, 0,
                                  output_texture_width_, output_texture_height_,
                                  GL_COLOR_BUFFER_BIT, GL_NEAREST);
      gl->ShallowFlushCHROMIUM();

      gl->BindTexture(GL_TEXTURE_2D, output_texture_);
      gl->BindFramebuffer(GL_FRAMEBUFFER, resolve_fbo);
      gl->FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, output_texture_, 0);

      // float copied_color[16] = {0};
      // gl->ReadPixels(0, 0, 2, 2, GL_RGBA, GL_FLOAT, copied_color);
      // for (size_t i = 0; i < 16; i++) {
      //   LOG(ERROR) << "======the copied data = " << copied_color[i];
      // }
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
