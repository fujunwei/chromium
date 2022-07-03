// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/ml.h"

#include "build/buildflag.h"
#include "components/ml/mojom/web_platform_model.mojom-blink.h"
#include "third_party/blink/public/common/features.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise_resolver.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_context_options.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"
#include "third_party/blink/renderer/modules/ml/webnn/mojo_client.h"
#include "third_party/blink/renderer/modules/ml/webnn/mojo_context.h"
#include "third_party/blink/renderer/platform/bindings/exception_state.h"

namespace blink {

namespace {

using ml::model_loader::mojom::blink::CreateModelLoaderOptionsPtr;
using ml::model_loader::mojom::blink::MLService;

}  // namespace

// static
const char ML::kSupplementName[] = "ML";

// static
ML* ML::ml(NavigatorBase& navigator) {
  ML* ml = Supplement<NavigatorBase>::From<ML>(navigator);
  if (!ml) {
    ml = MakeGarbageCollected<ML>(navigator);
    ProvideTo(navigator, ml);
  }
  return ml;
}

ML::ML(NavigatorBase& navigator)
    : Supplement<NavigatorBase>(navigator),
      execution_context_(navigator.GetExecutionContext()),
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

scoped_refptr<MojoClient> ML::GetMojoClient() {
  if (!webnn_mojo_client_) {
    webnn_mojo_client_ = MojoClient::Create(execution_context_);
  }
  return webnn_mojo_client_;
}

void ML::Trace(Visitor* visitor) const {
  visitor->Trace(execution_context_);
  visitor->Trace(remote_service_);
  Supplement<NavigatorBase>::Trace(visitor);
  ScriptWrappable::Trace(visitor);
  if (webnn_mojo_client_) {
    webnn_mojo_client_->Trace(visitor);
  }
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

  auto promise = resolver->Promise();

  MLContext* ml_context = nullptr;
  // Create MLContext for WebNN GPU backend.
  if (option->type() == V8MLContextType::Enum::kWebnn &&
      option->devicePreference().AsEnum() == V8MLDevicePreference::Enum::kGpu) {
    if (!base::FeatureList::IsEnabled(
            features::kEnableMachineLearningNeuralNetworkApi)) {
      resolver->Reject(MakeGarbageCollected<DOMException>(
          DOMExceptionCode::kNotSupportedError,
          "Machine Learning Neural Network feature is not supported."));
      return promise;
    }
    ml_context = MakeGarbageCollected<MojoContext>(WrapPersistent(script_state),
                                                   WrapPersistent(resolver),
                                                   GetMojoClient(), this);
  } else {
    // Create MLContext for Model Loader.
    ml_context = MakeGarbageCollected<MLContext>(
        option->devicePreference(), option->powerPreference(),
        option->modelFormat(), option->numThreads(), this);
    resolver->Resolve(ml_context);
  }

  return promise;
}

MLContext* ML::createContextSync(MLContextOptions* option,
                                 ExceptionState& exception_state) {
  return MakeGarbageCollected<MLContext>(
      option->devicePreference(), option->powerPreference(),
      option->modelFormat(), option->numThreads(), this);
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
