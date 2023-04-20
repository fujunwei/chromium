// Copyright 2023 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_cros.h"

#include "base/numerics/checked_math.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise_resolver.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_compute_result.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"
#include "third_party/blink/renderer/modules/ml/ml.h"
#include "third_party/blink/renderer/modules/ml/ml_context.h"
#include "third_party/blink/renderer/modules/ml/ml_model_loader.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_builder.h"
#include "third_party/blink/renderer/platform/bindings/exception_state.h"

namespace blink {

namespace {

flatbuffers::DetachedBuffer* g_flatbuffer_for_testing = nullptr;

using ml::model_loader::mojom::blink::ComputeResult;

bool ValidateModelLoadedTensorInfo(
    const HashMap<String, ml::model_loader::mojom::blink::TensorInfoPtr>&
        model_tensor_info,
    const HashMap<String, MLGraph::ResourceInfo>& graph_resources_info,
    String& error_message) {
  if (model_tensor_info.size() != graph_resources_info.size()) {
    error_message =
        "The number of model loaded tensor info doesn't match graph's "
        "expectation.";
    return false;
  }
  for (const auto& [name, mojo_tensor] : model_tensor_info) {
    if (!graph_resources_info.Contains(name)) {
      error_message = String::Format("The name \"%s\" isn't part of the graph.",
                                     name.Utf8().c_str());
      return false;
    }
    if (mojo_tensor->byte_size != graph_resources_info.at(name).byte_length) {
      error_message = String::Format(
          "The byte length of the model loaded tensor info with name \"%s\" "
          "doesn't match graph's expectation.",
          name.Utf8().c_str());
      return false;
    }
  }
  return true;
}

}  // namespace

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
  DOMArrayBuffer* buffer = nullptr;
  if (g_flatbuffer_for_testing) {
    buffer = DOMArrayBuffer::Create(g_flatbuffer_for_testing->data(),
                                    g_flatbuffer_for_testing->size());
  }
  // TODO(crbug.com/1273291): Convert WebNN graph to flatbuffer in the `buffer`.
  auto* script_state = resolver->GetScriptState();
  auto* execution_context = ExecutionContext::From(script_state);
  auto* ml_model_loader = ml_context_->GetModelLoaderForWebNN(script_state);
  ml_model_loader->Load(
      script_state, buffer,
      WTF::BindOnce(&MLGraphCrOS::OnRemoteModelLoad, WrapPersistent(this),
                    WrapPersistent(execution_context),
                    WrapPersistent(resolver)));
}

void MLGraphCrOS::OnRemoteModelLoad(
    ExecutionContext* execution_context,
    ScriptPromiseResolver* resolver,
    ml::model_loader::mojom::blink::LoadModelResult result,
    mojo::PendingRemote<ml::model_loader::mojom::blink::Model> pending_remote,
    ml::model_loader::mojom::blink::ModelInfoPtr tensor_info) {
  if (result != ml::model_loader::mojom::blink::LoadModelResult::kOk) {
    resolver->Reject(MakeGarbageCollected<DOMException>(
        DOMExceptionCode::kUnknownError, "Unknown error."));
    return;
  }
  // Verifies the inputs from model are expected for the WebNN graph.
  String error_message;
  if (!ValidateModelLoadedTensorInfo(tensor_info->input_tensor_info,
                                     input_resources_info_, error_message)) {
    resolver->Reject(MakeGarbageCollected<DOMException>(
        DOMExceptionCode::kDataError, "Invalid inputs: " + error_message));
    return;
  }
  // Verifies the outputs from model are expected for the WebNN graph.
  if (!ValidateModelLoadedTensorInfo(tensor_info->output_tensor_info,
                                     output_resources_info_, error_message)) {
    resolver->Reject(MakeGarbageCollected<DOMException>(
        DOMExceptionCode::kDataError, "Invalid outputs: " + error_message));
    return;
  }

  remote_model_.Bind(
      std::move(pending_remote),
      execution_context->GetTaskRunner(TaskType::kInternalDefault));

  resolver->Resolve(this);
}

// static
void MLGraphCrOS::SetFlatbufferForTesting(
    flatbuffers::DetachedBuffer* flatbuffer) {
  g_flatbuffer_for_testing = flatbuffer;
}

MLGraph* MLGraphCrOS::BuildSyncImpl(const MLNamedOperands& named_outputs,
                                    ExceptionState& exception_state) {
  // TODO(crbug.com/1273291): Support sync build that is only exposed to
  // dedicated worker.
  exception_state.ThrowDOMException(DOMExceptionCode::kNotSupportedError,
                                    "Not implemented.");
  return nullptr;
}

void MLGraphCrOS::ComputeAsyncImpl(const MLNamedArrayBufferViews& inputs,
                                   const MLNamedArrayBufferViews& outputs,
                                   ScriptPromiseResolver* resolver,
                                   ExceptionState& exception_state) {
  // The inputs has been verified in the basic class. so we can fill the buffer
  // directly with the input tensors.
  HashMap<String, Vector<uint8_t>> input_mojo;
  for (const auto& [name, array_buffer_view] : inputs) {
    wtf_size_t size =
        base::checked_cast<wtf_size_t>(array_buffer_view->byteLength());
    Vector<uint8_t> tensor(size);
    memcpy(tensor.data(), array_buffer_view->BaseAddress(), size);

    input_mojo.insert(name, std::move(tensor));
  }
  remote_model_->Compute(
      std::move(input_mojo),
      WTF::BindOnce(
          &MLGraphCrOS::OnComputeGraph, WrapPersistent(this),
          WrapPersistent(resolver),
          WrapPersistent(MakeGarbageCollected<MLNamedArrayBufferViews>(inputs)),
          WrapPersistent(
              MakeGarbageCollected<MLNamedArrayBufferViews>(outputs))));
}

void MLGraphCrOS::OnComputeGraph(
    ScriptPromiseResolver* resolver,
    const MLNamedArrayBufferViews* ml_inputs,
    const MLNamedArrayBufferViews* ml_outputs,
    ComputeResult mojo_result,
    const absl::optional<HashMap<String, Vector<uint8_t>>>& mojo_outputs) {
  if (mojo_result != ComputeResult::kOk || !mojo_outputs.has_value()) {
    resolver->Reject(MakeGarbageCollected<DOMException>(
        DOMExceptionCode::kOperationError,
        "Failed to obtain the computation result."));
    return;
  }

  for (const auto& [name, array_buffer_view] : *ml_outputs) {
    // Verifies the output because the ArrayBufferView can be detached before
    // invoking the callback of computing.
    if (array_buffer_view->IsDetached()) {
      resolver->Reject(MakeGarbageCollected<DOMException>(
          DOMExceptionCode::kUnknownError,
          "The array buffer view is detached: " + name));
      return;
    }

    // The verification before computing ensures the `ml_outputs` match graph's
    // expectation, so we only need to verify the `mojo_outputs` here.
    auto output_tensor_data = mojo_outputs.value().find(name);
    if (output_tensor_data == mojo_outputs.value().end()) {
      resolver->Reject(MakeGarbageCollected<DOMException>(
          DOMExceptionCode::kOperationError,
          "Failed to get result for the output " + name));
      return;
    }
    if (output_tensor_data->value.size() != array_buffer_view->byteLength()) {
      resolver->Reject(MakeGarbageCollected<DOMException>(
          DOMExceptionCode::kUnknownError,
          "The output tensor size does not match graph's expectation: " +
              name));
      return;
    }
    memcpy(array_buffer_view->BaseAddress(), output_tensor_data->value.data(),
           output_tensor_data->value.size());
  }
  auto* result = MLComputeResult::Create();
  result->setInputs(*ml_inputs);
  result->setOutputs(*ml_outputs);
  resolver->Resolve(result);
}

void MLGraphCrOS::ComputeSyncImpl(const MLNamedArrayBufferViews& inputs,
                                  const MLNamedArrayBufferViews& outputs,
                                  ExceptionState& exception_state) {
  // TODO(crbug.com/1273291): Support sync compute that is only exposed to
  // dedicated worker.
  exception_state.ThrowDOMException(DOMExceptionCode::kNotSupportedError,
                                    "Not implemented.");
}

}  // namespace blink
