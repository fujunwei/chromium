// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/ml.h"

#include "build/buildflag.h"
#include "components/ml/mojom/web_platform_model.mojom-blink.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise_resolver.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_context_options.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"
#include "third_party/blink/renderer/core/inspector/console_message.h"
#include "third_party/blink/renderer/modules/ml/webnn/webnn_client.h"
#include "third_party/blink/renderer/modules/ml/webnn/webnn_context.h"
#include "third_party/blink/renderer/platform/bindings/exception_state.h"

#if BUILDFLAG(IS_LINUX)
#include "third_party/blink/renderer/modules/ml/ml_context_xnnpack.h"
#endif

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

void ML::Trace(Visitor* visitor) const {
  visitor->Trace(execution_context_);
  visitor->Trace(remote_service_);

  // Webnn
  webnn_client_->Trace(visitor);

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

  // Notice that currently, we just create the context in the renderer. In the
  // future we may add backend query ability to check whether a context is
  // supportable or not. At that time, this function will be truly asynced.
  ScriptPromiseResolver* resolver =
      MakeGarbageCollected<ScriptPromiseResolver>(script_state);
  auto promise = resolver->Promise();
  MLContext* ml_context = nullptr;
  if (option->type() == V8MLContextType::Enum::kWebnn) {
    // Create MLContext for WebNN
    if (option->devicePreference().AsEnum() ==
            V8MLDevicePreference::Enum::kDefault ||
        option->devicePreference().AsEnum() ==
            V8MLDevicePreference::Enum::kCpu) {
      // Create XNNPACK backed MLContext for WebNN
#if BUILDFLAG(IS_LINUX)
      MLContextXnnpack* ml_context_xnnpack =
          MakeGarbageCollected<MLContextXnnpack>(option->numThreads(), this);
      if (!ml_context_xnnpack->Initialize()) {
        exception_state.ThrowDOMException(DOMExceptionCode::kOperationError,
                                          "failed to initialize XNNPACK");
        return ScriptPromise();
      }
      execution_context_->AddConsoleMessage(
          MakeGarbageCollected<ConsoleMessage>(
              mojom::blink::ConsoleMessageSource::kJavaScript,
              mojom::blink::ConsoleMessageLevel::kInfo,
              "Created WebNN MLContext with pthreadpool threads count: " +
                  String::Number(pthreadpool_get_threads_count(
                      ml_context_xnnpack->Pthreadpool()))));
      ml_context = ml_context_xnnpack;
      resolver->Resolve(ml_context);
#endif
    } else {
      if (!webnn_client_) {
        webnn_client_ = WebnnClient::Create(execution_context_);
      }
      DCHECK_NE(webnn_client_, nullptr);
      ml_context = MakeGarbageCollected<WebnnContext>(
          WrapPersistent(script_state), WrapPersistent(resolver),
          option->powerPreference(), this, webnn_client_);
    }
  } else {
    // Create MLContext for Model Loader
    ml_context = MakeGarbageCollected<MLContext>(
        option->devicePreference(), option->powerPreference(),
        option->modelFormat(), option->numThreads(), this);
    resolver->Resolve(ml_context);
  }

  return promise;
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
