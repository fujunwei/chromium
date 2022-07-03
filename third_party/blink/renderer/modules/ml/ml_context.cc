// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/ml_context.h"

#include "third_party/blink/public/common/features.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise_resolver.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"
#include "third_party/blink/renderer/modules/ml/ml.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph.h"
#include "third_party/blink/renderer/platform/bindings/exception_state.h"

namespace blink {

MLContext::MLContext(const V8MLDevicePreference device_preference,
                     const V8MLPowerPreference power_preference,
                     const V8MLModelFormat model_format,
                     const unsigned int num_threads,
                     ExecutionContext* execution_context,
                     ML* ml)
    : device_preference_(device_preference),
      power_preference_(power_preference),
      model_format_(model_format),
      num_threads_(num_threads),
      ml_(ml),
      webnn_context_(execution_context) {}

MLContext::~MLContext() = default;

V8MLDevicePreference MLContext::GetDevicePreference() const {
  return device_preference_;
}

V8MLPowerPreference MLContext::GetPowerPreference() const {
  return power_preference_;
}

V8MLModelFormat MLContext::GetModelFormat() const {
  return model_format_;
}

unsigned int MLContext::GetNumThreads() const {
  return num_threads_;
}

ML* MLContext::GetML() {
  return ml_.Get();
}

void MLContext::Trace(Visitor* visitor) const {
  visitor->Trace(ml_);
  visitor->Trace(webnn_context_);

  ScriptWrappable::Trace(visitor);
}

// TODO(crbug.com/1273291): Remove this and calls once investigation is
// complete.
BASE_FEATURE(kWebnnMojoContext,
             "WebnnMojoContext",
             base::FEATURE_DISABLED_BY_DEFAULT);

bool MLContext::IsWebnnMojoContextEnabled() const {
  return base::FeatureList::IsEnabled(kWebnnMojoContext) &&
         device_preference_ != V8MLDevicePreference::Enum::kCpu;
}

void MLContext::CreateWebnnMojoContext(ScriptPromiseResolver* resolver) {
  auto options = ml::webnn::mojom::blink::ContextOptions::New();
  // TODO(crbug.com/1273291): Set power preference in the context option.
  options->device_preference =
      ml::model_loader::mojom::blink::DevicePreference::kGpu;
  ml_->CreateWebnnMojoContext(
      resolver, std::move(options),
      WTF::BindOnce(&MLContext::OnWebnnContextCreated, WrapPersistent(this),
                    WrapPersistent(resolver)));
}

void MLContext::CreateWebnnGraph(
    ScriptPromiseResolver* resolver,
    ml::webnn::mojom::blink::Context::CreateGraphCallback callback) {
  if (!webnn_context_.is_bound()) {
    resolver->Reject(MakeGarbageCollected<DOMException>(
        DOMExceptionCode::kUnknownError, "Remote context isn't bound."));
    return;
  }
  webnn_context_->CreateGraph(std::move(callback));
}

void MLContext::OnWebnnContextCreated(
    ScriptPromiseResolver* resolver,
    mojo::PendingRemote<ml::webnn::mojom::blink::Context> pending_remote) {
  ScriptState* script_state = resolver->GetScriptState();
  if (!script_state->ContextIsValid()) {
    resolver->Reject(MakeGarbageCollected<DOMException>(
        DOMExceptionCode::kInvalidStateError, "Invalid script state."));
    return;
  }
  auto* execution_context = ExecutionContext::From(script_state);
  webnn_context_.Bind(
      std::move(pending_remote),
      execution_context->GetTaskRunner(TaskType::kInternalDefault));

  resolver->Resolve(this);
}

ScriptPromise MLContext::compute(ScriptState* script_state,
                                 MLGraph* graph,
                                 const MLNamedArrayInputs& inputs,
                                 const MLNamedArrayOutputs& outputs,
                                 ExceptionState& exception_state) {
  if (!script_state->ContextIsValid()) {
    exception_state.ThrowDOMException(DOMExceptionCode::kInvalidStateError,
                                      "Invalid script state");
    return ScriptPromise();
  }
  return graph->ComputeImpl(script_state, inputs, outputs, exception_state);
}

void MLContext::computeSync(MLGraph* graph,
                            const MLNamedArrayInputs& inputs,
                            const MLNamedArrayOutputs& outputs,
                            ExceptionState& exception_state) {
  return graph->ComputeSyncImpl(inputs, outputs, exception_state);
}

}  // namespace blink
