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
  ml_context_->CreateWebnnGraph(
      resolver,
      WTF::BindOnce(&MLGraphMojo::OnGraphCreated, WrapPersistent(this),
                    WrapPersistent(named_outputs), WrapPersistent(resolver)));
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

void MLGraphMojo::OnGraphCreated(
    const MLNamedOperands* named_output,
    ScriptPromiseResolver* resolver,
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
