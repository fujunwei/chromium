// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/ml_context.h"

#include "third_party/blink/renderer/bindings/core/v8/script_promise_resolver.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"
#include "third_party/blink/renderer/modules/ml/ml.h"
#include "third_party/blink/renderer/platform/bindings/exception_state.h"

namespace blink {

MLContext::MLContext(const V8MLDevicePreference device_preference,
                     const V8MLPowerPreference power_preference,
                     const V8MLModelFormat model_format,
                     const unsigned int num_threads,
                     ML* ml)
    : device_preference_(device_preference),
      power_preference_(power_preference),
      model_format_(model_format),
      num_threads_(num_threads),
      ml_(ml)
#if BUILDFLAG(BUILD_WEBNN_WITH_SERVICE)
      ,
      webnn_context_(ml->GetExecutionContext())
#endif
{
}

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
#if BUILDFLAG(BUILD_WEBNN_WITH_SERVICE)
  visitor->Trace(webnn_context_);
#endif

  ScriptWrappable::Trace(visitor);
}

ScriptPromise MLContext::compute(ScriptState* script_state,
                                 MLGraph* graph,
                                 const MLNamedArrayBufferViews& inputs,
                                 const MLNamedArrayBufferViews& outputs,
                                 ExceptionState& exception_state) {
  if (!script_state->ContextIsValid()) {
    exception_state.ThrowDOMException(DOMExceptionCode::kInvalidStateError,
                                      "Invalid script state");
    return ScriptPromise();
  }

  auto* resolver = MakeGarbageCollected<ScriptPromiseResolver>(
      script_state, exception_state.GetContext());
  auto promise = resolver->Promise();

  if (graph->Context() != this) {
    resolver->Reject(MakeGarbageCollected<DOMException>(
        DOMExceptionCode::kDataError,
        "The graph isn't built within this context."));
  } else {
    graph->ComputeAsync(inputs, outputs, resolver, exception_state);
  }

  return promise;
}

void MLContext::computeSync(MLGraph* graph,
                            const MLNamedArrayBufferViews& inputs,
                            const MLNamedArrayBufferViews& outputs,
                            ExceptionState& exception_state) {
  if (graph->Context() != this) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kDataError,
        "The graph isn't built within this context.");
    return;
  }
  graph->ComputeSync(inputs, outputs, exception_state);
}

#if BUILDFLAG(BUILD_WEBNN_WITH_SERVICE)
void MLContext::CreateWebnnGraph(
    ScriptPromiseResolver* resolver,
    webnn::mojom::blink::WebnnContext::CreateGraphCallback callback) {
  if (!webnn_context_.is_bound()) {
    // Needs to create `WebnnContext` interface first.
    auto options = webnn::mojom::blink::CreateContextOptions::New();
    // TODO(crbug.com/1273291): Set power preference in the context option.
    ml_->CreateWebnnContext(
        resolver, std::move(options),
        WTF::BindOnce(&MLContext::OnWebnnContextCreated, WrapPersistent(this),
                      WrapPersistent(resolver)));
    create_graph_callback_ = std::move(callback);
  } else {
    // Directly use `webnn_context_` to create `WebnnGraph` message pipe.
    webnn_context_->CreateGraph(std::move(callback));
  }
}

void MLContext::OnWebnnContextCreated(
    ScriptPromiseResolver* resolver,
    webnn::mojom::blink::CreateContextResult result,
    mojo::PendingRemote<webnn::mojom::blink::WebnnContext> pending_remote) {
  ScriptState* script_state = resolver->GetScriptState();
  if (!script_state->ContextIsValid()) {
    resolver->Reject(MakeGarbageCollected<DOMException>(
        DOMExceptionCode::kInvalidStateError, "Invalid script state."));
    return;
  }
  switch (result) {
    case webnn::mojom::blink::CreateContextResult::kUnknownError: {
      resolver->Reject(MakeGarbageCollected<DOMException>(
          DOMExceptionCode::kUnknownError, "Internal error."));
      return;
    }
    case webnn::mojom::blink::CreateContextResult::kNotSupported: {
      resolver->Reject(MakeGarbageCollected<DOMException>(
          DOMExceptionCode::kNotSupportedError,
          "Input configuration not supported."));
      return;
    }
    case webnn::mojom::blink::CreateContextResult::kOk: {
      auto* execution_context = ExecutionContext::From(script_state);
      webnn_context_.Bind(
          std::move(pending_remote),
          execution_context->GetTaskRunner(TaskType::kInternalDefault));

      webnn_context_->CreateGraph(std::move(create_graph_callback_));
      return;
    }
  }
}
#endif

}  // namespace blink
