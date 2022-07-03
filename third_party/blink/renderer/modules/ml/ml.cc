// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/ml.h"

#include "build/buildflag.h"
#include "components/ml/mojom/web_platform_model.mojom-blink.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise_resolver.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_context_options.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"
#include "third_party/blink/renderer/modules/ml/webnn/mojo_client.h"
#include "third_party/blink/renderer/platform/bindings/exception_state.h"

namespace blink {

namespace {

using ml::model_loader::mojom::blink::CreateModelLoaderOptionsPtr;
using ml::model_loader::mojom::blink::MLService;

}  // namespace

ML::ML(ExecutionContext* execution_context)
    : execution_context_(execution_context),
      remote_service_(execution_context_.Get()) {}

void ML::CreateModelLoader(ScriptState* script_state,
                           ExceptionState& exception_state,
                           CreateModelLoaderOptionsPtr options,
                           MLService::CreateModelLoaderCallback callback) {
  if (!BootstrapMojoConnectionIfNeeded(script_state, exception_state)) {
    // An exception has already been thrown in
    // `BootstrapMojoConnectionIfNeeded()`.
    return;
  }
  remote_service_->CreateModelLoader(std::move(options), std::move(callback));
}

void ML::CreateWebnnMojoContext(ScriptPromiseResolver* resolver,
                                ContextOptionsPtr options,
                                MojoServer::CreateContextCallback callback) {
  if (!webnn_mojo_client_) {
    webnn_mojo_client_ = MakeGarbageCollected<MojoClient>(execution_context_);
  }

  webnn_mojo_client_->CreateMojoContext(resolver, std::move(options),
                                        std::move(callback));
}

void ML::Trace(Visitor* visitor) const {
  visitor->Trace(execution_context_);
  visitor->Trace(remote_service_);
  visitor->Trace(webnn_mojo_client_);
  ScriptWrappable::Trace(visitor);
}

ScriptPromise ML::createContext(ScriptState* script_state,
                                MLContextOptions* option,
                                ExceptionState& exception_state) {
  if (!script_state->ContextIsValid()) {
    exception_state.ThrowDOMException(DOMExceptionCode::kInvalidStateError,
                                      "Invalid script state");
    return ScriptPromise();
  }

  ScriptPromiseResolver* resolver =
      MakeGarbageCollected<ScriptPromiseResolver>(script_state);

  MLContext* ml_context = MakeGarbageCollected<MLContext>(
      option->devicePreference(), option->powerPreference(),
      option->modelFormat(), option->numThreads(), execution_context_, this);
  if (ml_context->IsWebnnMojoContextEnabled()) {
    // WebNN mojo context need to be created in server side to map different
    // hardware acceleration and manage the processes of graph execution, so the
    // ml context will await the callback from server side and then resolve.
    ml_context->CreateWebnnMojoContext(resolver);
  } else {
    resolver->Resolve(ml_context);
  }

  return resolver->Promise();
}

bool ML::BootstrapMojoConnectionIfNeeded(ScriptState* script_state,
                                         ExceptionState& exception_state) {
  // We need to do the following check because the execution context of this
  // navigator may be invalid (e.g. the frame is detached).
  if (!script_state->ContextIsValid()) {
    exception_state.ThrowDOMException(DOMExceptionCode::kInvalidStateError,
                                      "The execution context is invalid");
    return false;
  }
  // Note that we do not use `ExecutionContext::From(script_state)` because
  // the ScriptState passed in may not be guaranteed to match the execution
  // context associated with this navigator, especially with
  // cross-browsing-context calls.
  if (!remote_service_.is_bound()) {
    execution_context_->GetBrowserInterfaceBroker().GetInterface(
        remote_service_.BindNewPipeAndPassReceiver(
            execution_context_->GetTaskRunner(TaskType::kInternalDefault)));
  }
  return true;
}

}  // namespace blink
