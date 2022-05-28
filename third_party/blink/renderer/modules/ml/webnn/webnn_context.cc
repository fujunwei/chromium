// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/webnn_context.h"

#include "components/ml/webnn/mojom/webnn_service.mojom-blink.h"
#include "mojo/public/cpp/system/handle.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise_resolver.h"
#include "third_party/blink/renderer/core/execution_context/execution_context.h"
#include "third_party/blink/renderer/modules/ml/ml.h"
#include "third_party/blink/renderer/modules/ml/ml_context.h"
#include "third_party/blink/renderer/platform/heap/persistent.h"
#include "third_party/blink/renderer/platform/wtf/functional.h"

namespace blink {

namespace {

using ml::webnn::mojom::blink::Context;

}
class ScriptState;
class ScriptPromiseResolver;

WebnnContext::WebnnContext(ScriptState* script_state,
                           ScriptPromiseResolver* resolver,
                           const V8MLPowerPreference power_preference,
                           ML* ml,
                           scoped_refptr<WebnnClient> webnn_client)
    : MLContext(V8MLDevicePreference(V8MLDevicePreference::Enum::kGpu),
                power_preference,
                V8MLModelFormat(V8MLModelFormat::Enum::kTflite),
                1,
                ml),
      WebnnObjectBase(webnn_client),
      remote_context_(ExecutionContext::From(script_state)) {
  auto options = ml::webnn::mojom::blink::ContextOptions::New();
  // TODO: set the power preference to the context options
  // mojo::ScopedHandle(mojo::Handle(static_cast<MojoHandle>(GetObjectId())))
  GetWebnnClient()->CreateWebnnContext(
      GetObjectId(), std::move(options),
      WTF::Bind(&WebnnContext::OnContextCreated, WrapPersistent(this),
                WrapPersistent(script_state), WrapPersistent(resolver)));
}

WebnnContext::~WebnnContext() = default;

void WebnnContext::CreateGraph(uint32_t graph_id,
                               uint32_t context_id,
                               Context::CreateGraphCallback callback) {
  if (!remote_context_.is_bound()) {
    // TODO: Throw exception here
    return;
  }
  remote_context_->CreateGraph(graph_id, context_id, std::move(callback));
}

void WebnnContext::Trace(Visitor* visitor) const {
  visitor->Trace(remote_context_);
  MLContext::Trace(visitor);
}

void WebnnContext::OnContextCreated(
    ScriptState* script_state,
    ScriptPromiseResolver* resolver,
    mojo::PendingRemote<Context> pending_remote) {
  auto* execution_context = ExecutionContext::From(script_state);
  remote_context_.Bind(
      std::move(pending_remote),
      execution_context->GetTaskRunner(TaskType::kInternalDefault));
  resolver->Resolve(this);
}

}  // namespace blink
