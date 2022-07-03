// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/mojo_context.h"

#include "third_party/blink/renderer/bindings/core/v8/script_promise_resolver.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"
#include "third_party/blink/renderer/core/execution_context/execution_context.h"
#include "third_party/blink/renderer/modules/ml/ml.h"
#include "third_party/blink/renderer/modules/ml/ml_context.h"
#include "third_party/blink/renderer/platform/heap/persistent.h"
#include "third_party/blink/renderer/platform/wtf/functional.h"

namespace blink {

MojoContext::MojoContext(ScriptState* script_state,
                         ScriptPromiseResolver* resolver,
                         scoped_refptr<MojoClient> mojo_client,
                         ML* ml)
    : MLContext(
          V8MLDevicePreference(V8MLDevicePreference::Enum::kGpu),
          V8MLPowerPreference(V8MLPowerPreference::Enum::kHighPerformance),
          V8MLModelFormat(V8MLModelFormat::Enum::kTflite),
          1,
          ml),
      ObjectHandle(mojo_client),
      remote_context_(ExecutionContext::From(script_state)) {
  auto options = ml::webnn::mojom::blink::ContextOptions::New();
  options->device_preference =
      ml::model_loader::mojom::blink::DevicePreference::kGpu;
  ml->GetMojoClient()->CreateMojoContext(
      resolver, GetObjectId(), std::move(options),
      WTF::Bind(&MojoContext::OnContextCreated, WrapPersistent(this),
                WrapPersistent(script_state), WrapPersistent(resolver)));
}

MojoContext::~MojoContext() = default;

void MojoContext::CreateGraph(ScriptPromiseResolver* resolver,
                              ObjectId graph_id,
                              Context::CreateGraphCallback callback) {
  if (!remote_context_.is_bound()) {
    resolver->Reject(MakeGarbageCollected<DOMException>(
        DOMExceptionCode::kUnknownError, "Remote context isn't bound."));
    return;
  }
  remote_context_->CreateGraph(graph_id, std::move(callback));
}

void MojoContext::Trace(Visitor* visitor) const {
  visitor->Trace(remote_context_);
  MLContext::Trace(visitor);
}

void MojoContext::OnContextCreated(
    ScriptState* script_state,
    ScriptPromiseResolver* resolver,
    mojo::PendingRemote<Context> pending_remote) {
  if (!script_state->ContextIsValid()) {
    resolver->Reject(MakeGarbageCollected<DOMException>(
        DOMExceptionCode::kInvalidStateError, "Invalid script state."));
    return;
  }
  auto* execution_context = ExecutionContext::From(script_state);
  remote_context_.Bind(
      std::move(pending_remote),
      execution_context->GetTaskRunner(TaskType::kInternalDefault));

  resolver->Resolve(this);
}

}  // namespace blink
