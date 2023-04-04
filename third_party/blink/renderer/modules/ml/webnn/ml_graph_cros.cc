// Copyright 2023 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_cros.h"

#include <numeric>

#include "base/numerics/checked_math.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise_resolver.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_compute_result.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"
#include "third_party/blink/renderer/modules/ml/ml.h"
#include "third_party/blink/renderer/modules/ml/ml_context.h"
#include "third_party/blink/renderer/modules/ml/ml_model_loader.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_builder.h"
#include "third_party/blink/renderer/platform/bindings/exception_state.h"
#include "third_party/flatbuffers/src/include/flatbuffers/flatbuffers.h"

namespace blink {

// static
void MLGraphCrOS::ValidateAndBuildAsync(MLContext* ml_context,
                                        const MLNamedOperands& named_outputs,
                                        ScriptPromiseResolver* resolver) {
  auto* script_state = resolver->GetScriptState();
  auto* execution_context = ExecutionContext::From(script_state);
  auto* graph =
      MakeGarbageCollected<MLGraphCrOS>(execution_context, ml_context);
  graph->BuildAsync(named_outputs, resolver);
}

MLGraphCrOS::MLGraphCrOS(ExecutionContext* execution_context,
                         MLContext* ml_context)
    : MLGraph(ml_context), remote_model_(execution_context) {}

MLGraphCrOS::~MLGraphCrOS() = default;

void MLGraphCrOS::Trace(Visitor* visitor) const {
  visitor->Trace(remote_model_);
  MLGraph::Trace(visitor);
}

void MLGraphCrOS::BuildAsyncImpl(const MLNamedOperands& outputs,
                                 ScriptPromiseResolver* resolver) {
  auto* script_state = resolver->GetScriptState();
  // TODO(crbug.com/1273291): Convert WebNN graph to tflite model instead of
  // the empty buffer.
  const String kEmptyBufferForTesting = "";
  auto* execution_context = ExecutionContext::From(script_state);
  auto* ml_model_loader = ml_context_->GetModelLoaderForWebNN(script_state);
  ml_model_loader->Load(
      script_state, kEmptyBufferForTesting.Span8(), resolver,
      WTF::BindOnce(&MLGraphCrOS::OnRemoteModelLoad, WrapPersistent(this),
                    WrapPersistent(execution_context),
                    WrapPersistent(resolver)));
}

void MLGraphCrOS::OnRemoteModelLoad(ExecutionContext* execution_context,
                                    ScriptPromiseResolver* resolver,
                                    LoadModelResult result,
                                    mojo::PendingRemote<Model> pending_remote,
                                    ModelInfoPtr tensor_info) {
  if (result != LoadModelResult::kOk) {
    resolver->Reject(MakeGarbageCollected<DOMException>(
        DOMExceptionCode::kUnknownError, "Unknown error."));
    return;
  }
  remote_model_.Bind(
      std::move(pending_remote),
      execution_context->GetTaskRunner(TaskType::kInternalDefault));

  // Stores input tensor information of loaded model to verify the input
  // data by user including name and byte length.
  input_tensor_name_to_info_ = std::move(tensor_info->input_tensor_info);
  // Stores output tensor information of loaded model to verify the output
  // data returned from `MLService` after computing.
  output_tensor_name_to_info_ = std::move(tensor_info->output_tensor_info);

  resolver->Resolve(this);
}

const TensorInfoMap& MLGraphCrOS::GetInputTensorInfoMapForTesting() const {
  return input_tensor_name_to_info_;
}

const TensorInfoMap& MLGraphCrOS::GetOutputTensorInfoMapForTesting() const {
  return output_tensor_name_to_info_;
}

MLGraph* MLGraphCrOS::BuildSyncImpl(const MLNamedOperands& named_outputs,
                                    ExceptionState& exception_state) {
  // TODO(crbug.com/1273291): Support sync build.
  exception_state.ThrowDOMException(DOMExceptionCode::kNotSupportedError,
                                    "Not implemented.");
  return nullptr;
}

void MLGraphCrOS::ComputeAsyncImpl(const MLNamedArrayBufferViews& inputs,
                                   const MLNamedArrayBufferViews& outputs,
                                   ScriptPromiseResolver* resolver) {
  // TODO(crbug.com/1273291): Support async compute.
  resolver->Reject(MakeGarbageCollected<DOMException>(
      DOMExceptionCode::kNotSupportedError, "Not implemented."));
}

void MLGraphCrOS::ComputeSyncImpl(const MLNamedArrayBufferViews& inputs,
                                  const MLNamedArrayBufferViews& outputs,
                                  ExceptionState& exception_state) {
  // TODO(crbug.com/1273291): Support sync compute.
  exception_state.ThrowDOMException(DOMExceptionCode::kNotSupportedError,
                                    "Not implemented.");
}

}  // namespace blink
