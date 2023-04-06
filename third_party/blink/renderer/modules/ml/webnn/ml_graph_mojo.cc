// Copyright 2023 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_mojo.h"

#include "mojo/public/cpp/bindings/pending_remote.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise_resolver.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"
#include "third_party/blink/renderer/core/execution_context/execution_context.h"
#include "third_party/blink/renderer/modules/ml/ml.h"
#include "third_party/blink/renderer/modules/ml/ml_context.h"

namespace blink {

// static
void MLGraphMojo::ValidateAndBuildAsync(MLContext* context,
                                        const MLNamedOperands& named_outputs,
                                        ScriptPromiseResolver* resolver) {
  auto* graph =
      MakeGarbageCollected<MLGraphMojo>(resolver->GetScriptState(), context);
  graph->BuildAsync(named_outputs, resolver);
}

MLGraphMojo::MLGraphMojo(ScriptState* script_state, MLContext* context)
    : MLGraph(context), remote_graph_(ExecutionContext::From(script_state)) {}

MLGraphMojo::~MLGraphMojo() = default;

void MLGraphMojo::Trace(Visitor* visitor) const {
  visitor->Trace(remote_graph_);
  MLGraph::Trace(visitor);
}

void MLGraphMojo::BuildAsyncImpl(const MLNamedOperands& outputs,
                                 ScriptPromiseResolver* resolver) {
  auto* named_outputs = MakeGarbageCollected<MLNamedOperands>(outputs);
  // Create `WebnnGraph` message pipe with `WebnnContext` mojo interface which
  // is owned by `ML` object of navigator.
  auto& remote_context = ml_context_->GetRemoteWebnnContext();
  if (!remote_context.is_bound()) {
    // Needs to create `WebnnContext` interface first.
    auto options = webnn::mojom::blink::CreateContextOptions::New();
    // TODO(crbug.com/1273291): Set power preference in the context option.
    ml_context_->GetML()->CreateWebnnContext(
        std::move(options),
        WTF::BindOnce(&MLGraphMojo::OnWebnnContextCreated, WrapPersistent(this),
                      WrapPersistent(resolver), WrapPersistent(named_outputs)));
  } else {
    // Directly use `WebnnContext` to create `WebnnGraph` message pipe.
    remote_context->CreateGraph(
        WTF::BindOnce(&MLGraphMojo::OnWebnnGraphCreated, WrapPersistent(this),
                      WrapPersistent(resolver), WrapPersistent(named_outputs)));
  }
}

MLGraph* MLGraphMojo::BuildSyncImpl(const MLNamedOperands& named_outputs,
                                    ExceptionState& exception_state) {
  // TODO(crbug.com/1273291): Support Sync build.
  NOTIMPLEMENTED();
  exception_state.ThrowDOMException(DOMExceptionCode::kNotSupportedError,
                                    "Not implemented");
  return nullptr;
}

void MLGraphMojo::ComputeAsyncImpl(const MLNamedArrayBufferViews& inputs,
                                   const MLNamedArrayBufferViews& outputs,
                                   ScriptPromiseResolver* resolver) {
  // TODO(crbug.com/1273291): Support Async compute.
  NOTIMPLEMENTED();
  resolver->Reject(MakeGarbageCollected<DOMException>(
      DOMExceptionCode::kNotSupportedError, "Async compute not supported."));
}

void MLGraphMojo::ComputeSyncImpl(const MLNamedArrayBufferViews& inputs,
                                  const MLNamedArrayBufferViews& outputs,
                                  ExceptionState& exception_state) {
  // TODO(crbug.com/1273291): Support Sync compute.
  NOTIMPLEMENTED();
  exception_state.ThrowDOMException(DOMExceptionCode::kNotSupportedError,
                                    "Not implemented");
}

void MLGraphMojo::OnWebnnContextCreated(
    ScriptPromiseResolver* resolver,
    const MLNamedOperands* named_outputs,
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
      auto& remote_context = ml_context_->GetRemoteWebnnContext();
      auto* execution_context = ExecutionContext::From(script_state);
      remote_context.Bind(
          std::move(pending_remote),
          execution_context->GetTaskRunner(TaskType::kInternalDefault));

      remote_context->CreateGraph(WTF::BindOnce(
          &MLGraphMojo::OnWebnnGraphCreated, WrapPersistent(this),
          WrapPersistent(resolver), WrapPersistent(named_outputs)));
      return;
    }
  }
}

void MLGraphMojo::OnWebnnGraphCreated(
    ScriptPromiseResolver* resolver,
    const MLNamedOperands* named_output,
    mojo::PendingRemote<webnn::mojom::blink::WebnnGraph> pending_remote) {
  auto* script_state = resolver->GetScriptState();
  auto* execution_context = ExecutionContext::From(script_state);
  // Bind the end point of `WebnnGraph` mojo interface in the blink side.
  remote_graph_.Bind(
      std::move(pending_remote),
      execution_context->GetTaskRunner(TaskType::kInternalDefault));

  // TODO(crbug.com/1273291): Build the graph in the WebNN Service.
  resolver->Resolve(this);
  return;
}

}  // namespace blink
