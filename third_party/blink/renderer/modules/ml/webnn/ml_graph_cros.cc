// Copyright 2023 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_cros.h"

#include <algorithm>
#include <numeric>

#include "base/numerics/checked_math.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise_resolver.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_compute_result.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_data_type.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"
#include "third_party/blink/renderer/modules/ml/ml.h"
#include "third_party/blink/renderer/modules/ml/ml_context.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_builder.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_utils.h"
#include "third_party/blink/renderer/modules/ml/webnn/tflite_converter.h"
#include "third_party/blink/renderer/platform/bindings/exception_state.h"

namespace blink {

namespace {

using ml::model_loader::mojom::blink::CreateModelLoaderOptions;
using ml::model_loader::mojom::blink::DataType;
using ml::model_loader::mojom::blink::DevicePreference;
using ml::model_loader::mojom::blink::ModelFormat;

TfLiteConverter* BuildTFLiteModel(const MLNamedOperands& named_outputs,
                                  String& error_message) {
  auto* converter = MakeGarbageCollected<TfLiteConverter>();
  // Hold the output name map so that the name can be got quickly with operand
  // when building output tensor.
  OperandNameMap output_operand_name_map;
  for (const auto& [name, operand] : named_outputs) {
    output_operand_name_map.insert(operand, name);
  }

  // Map the operand to its index of tensor.
  OperandTensorIndexMap operand_tensor_index_map;
  auto* toposorted_operators = GetOperatorsInTopologicalOrder(named_outputs);
  // Visit the operators in topological order. For each operator,
  // 1, Build `tflite::Tensor` for its input and output operands if needed.
  // 2, Build `tflite::Operator` with the tensor index of inputs and outputs
  // operand.
  for (const auto& current_operator : *toposorted_operators) {
    for (const auto& operand : current_operator->Inputs()) {
      if (operand_tensor_index_map.Contains(operand.Get())) {
        // The tensor is already built for this operand, skip it.
        continue;
      }
      switch (operand->Kind()) {
        case MLOperand::OperandKind::kInput: {
          // Serialize tensor for input operand.
          if (!converter->SerializeInput(operand.Get(),
                                         operand_tensor_index_map)) {
            return nullptr;
          }
          break;
        }
        case MLOperand::OperandKind::kConstant: {
          // Serialize tensor for constant operand.
          if (!converter->SerializeConstant(operand.Get(),
                                            operand_tensor_index_map)) {
            return nullptr;
          }
          break;
        }
        case MLOperand::OperandKind::kOutput:
          // Because the operators are visited in topological order, if this
          // operand is an intermediate operand, it should already be defined as
          // an output operand of the dependent operator.
          NOTREACHED();
          break;
      }
    }

    for (const auto& operand : current_operator->Outputs()) {
      if (operand_tensor_index_map.Contains(operand.Get())) {
        // The tensor is already built for this operand, skip it.
        continue;
      }
      if (!converter->SerializeOutput(operand.Get(), output_operand_name_map,
                                      operand_tensor_index_map)) {
        return nullptr;
      }
    }

    // There are following steps to implement the `BuildOperator` function:
    // 1, Create `tflite::OperatorCode` with the kind of operator.
    // 2, Create `tflite::Tensor` for intermediate operand, or model output
    //    operand with output name.
    // 3, Create `tflite::Operator` with the tensor index of inputs and outputs
    //    operand.
    if (!converter->SerializeOperations(
            current_operator.Get(), operand_tensor_index_map, error_message)) {
      return nullptr;
    }
  }

  // Build the model.
  if (!converter->BuildModel()) {
    return nullptr;
  }
  return converter;
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
    : MLGraph(ml_context),
      remote_loader_(execution_context),
      remote_model_(execution_context) {}

MLGraphCrOS::~MLGraphCrOS() = default;

void MLGraphCrOS::Trace(Visitor* visitor) const {
  visitor->Trace(remote_loader_);
  visitor->Trace(remote_model_);
  MLGraph::Trace(visitor);
}

void MLGraphCrOS::BuildAsyncImpl(const MLNamedOperands& outputs,
                                 ScriptPromiseResolver* resolver) {
  auto options_mojo = CreateModelLoaderOptions::New();
  options_mojo->num_threads = ml_context_->GetNumThreads();
  // Hardcode the preference of creating ModelLoader mojo interface because
  // `MLService` only support TF-Lite model and CPU device at current stage.
  // TODO(crbug.com/1273291): Support power preference, other model format and
  // device preference.
  options_mojo->model_format = ModelFormat::kTfLite;
  options_mojo->device_preference = DevicePreference::kCpu;
  auto* named_outputs = MakeGarbageCollected<MLNamedOperands>(outputs);
  auto* script_state = resolver->GetScriptState();
  // Use `ModelLoader` mojo interface to load WebNN computational graphs that is
  // converted in FlatBuffer.
  ml_context_->GetML()->CreateModelLoader(
      script_state, std::move(options_mojo),
      WTF::BindOnce(&MLGraphCrOS::OnRemoteLoaderCreated, WrapPersistent(this),
                    WrapPersistent(script_state), WrapPersistent(resolver),
                    WrapPersistent(named_outputs)));
}

void MLGraphCrOS::OnRemoteLoaderCreated(
    ScriptState* script_state,
    ScriptPromiseResolver* resolver,
    const MLNamedOperands* named_outputs,
    CreateModelLoaderResult result,
    mojo::PendingRemote<ModelLoader> pending_remote) {
  switch (result) {
    case CreateModelLoaderResult::kUnknownError: {
      resolver->Reject(MakeGarbageCollected<DOMException>(
          DOMExceptionCode::kUnknownError, "Internal error."));
      return;
    }
    case CreateModelLoaderResult::kNotSupported: {
      resolver->Reject(MakeGarbageCollected<DOMException>(
          DOMExceptionCode::kNotSupportedError,
          "The context can not be supported."));
      return;
    }
    case CreateModelLoaderResult::kOk: {
      auto* execution_context = ExecutionContext::From(script_state);
      remote_loader_.Bind(
          std::move(pending_remote),
          execution_context->GetTaskRunner(TaskType::kInternalDefault));

      // TODO(crbug.com/1273291): Support other model format. ModelLoader mojo
      // interface only can load tflite model at current stage.
      // Transform WebNN Graphs to TF-Lite model in FlatBuffer with the schema.
      String error_message;
      TfLiteConverter* converter =
          BuildTFLiteModel(*named_outputs, error_message);
      if (!converter) {
        resolver->Reject(MakeGarbageCollected<DOMException>(
            DOMExceptionCode::kUnknownError, error_message));
        return;
      }
      auto& builder = converter->GetFlatBufferBuilder();
      // Load the FlatBuffer of WebNN Graph.
      remote_loader_->Load(
          base::make_span(
              static_cast<const uint8_t*>(builder.GetBufferPointer()),
              builder.GetSize()),
          WTF::BindOnce(&MLGraphCrOS::OnRemoteModelLoad, WrapPersistent(this),
                        WrapPersistent(execution_context),
                        WrapPersistent(resolver)));
      return;
    }
  }
}

void MLGraphCrOS::OnRemoteModelLoad(ExecutionContext* execution_context,
                                    ScriptPromiseResolver* resolver,
                                    LoadModelResult result,
                                    mojo::PendingRemote<Model> pending_remote,
                                    ModelInfoPtr tensor_info) {
  switch (result) {
    case LoadModelResult::kUnknownError:
      resolver->Reject(MakeGarbageCollected<DOMException>(
          DOMExceptionCode::kUnknownError, "Unknown error."));
      return;
    case LoadModelResult::kInvalidModel:
      resolver->Reject(MakeGarbageCollected<DOMException>(
          DOMExceptionCode::kDataError, "Invalid input model."));
      return;
    case LoadModelResult::kNotSupported:
      resolver->Reject(MakeGarbageCollected<DOMException>(
          DOMExceptionCode::kNotSupportedError, "Model can not be supported."));
      return;
    case LoadModelResult::kOk:
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
      return;
  }
}

MLGraph* MLGraphCrOS::BuildSyncImpl(const MLNamedOperands& named_outputs,
                                    ExceptionState& exception_state) {
  exception_state.ThrowDOMException(DOMExceptionCode::kNotSupportedError,
                                    "Not implemented.");
  return nullptr;
}

void MLGraphCrOS::ComputeAsyncImpl(const MLNamedArrayBufferViews& inputs,
                                   const MLNamedArrayBufferViews& outputs,
                                   ScriptPromiseResolver* resolver) {
  // First verifies the sizes of inputs.
  if (input_tensor_name_to_info_.size() != inputs.size()) {
    resolver->Reject(MakeGarbageCollected<DOMException>(
        DOMExceptionCode::kDataError,
        "The number of inputs doesn't match model's expectation."));
    return;
  }
  for (const auto& [name, array_buffer_view] : inputs) {
    auto iter = input_tensor_name_to_info_.find(name);
    if (iter == input_tensor_name_to_info_.end()) {
      resolver->Reject(MakeGarbageCollected<DOMException>(
          DOMExceptionCode::kDataError, "There is unknown input: " + name));
      return;
    }
    if (iter->value->byte_size != array_buffer_view->byteLength()) {
      resolver->Reject(MakeGarbageCollected<DOMException>(
          DOMExceptionCode::kUnknownError, "Wrong input size."));
      return;
    }
  }

  // Fills the buffer with input tensors.
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
          &MLGraphCrOS::OnComputeResult, WrapPersistent(this),
          WrapPersistent(resolver),
          WrapPersistent(MakeGarbageCollected<MLNamedArrayBufferViews>(inputs)),
          WrapPersistent(
              MakeGarbageCollected<MLNamedArrayBufferViews>(outputs))));
}

void MLGraphCrOS::OnComputeResult(
    ScriptPromiseResolver* resolver,
    const MLNamedArrayBufferViews* named_inputs,
    const MLNamedArrayBufferViews* named_outputs,
    ComputeResult result,
    const absl::optional<HashMap<String, Vector<uint8_t>>>& outputs) {
  if (result != ComputeResult::kOk || !outputs.has_value()) {
    resolver->Reject(MakeGarbageCollected<DOMException>(
        DOMExceptionCode::kOperationError,
        "Failed to obtain the computation result."));
    return;
  }

  if (outputs.value().size() != output_tensor_name_to_info_.size()) {
    resolver->Reject(MakeGarbageCollected<DOMException>(
        DOMExceptionCode::kUnknownError,
        "The number of output tensors of computation doesn't match the model's "
        "expectation."));
    return;
  }

  for (const auto& name_tensor : outputs.value()) {
    auto iter = output_tensor_name_to_info_.find(name_tensor.key);
    if (iter == output_tensor_name_to_info_.end()) {
      resolver->Reject(MakeGarbageCollected<DOMException>(
          DOMExceptionCode::kUnknownError,
          "There is an unknown output tensor in the computation result: " +
              name_tensor.key));
      return;
    }
    if (name_tensor.value.size() != iter->value->byte_size) {
      resolver->Reject(MakeGarbageCollected<DOMException>(
          DOMExceptionCode::kUnknownError,
          "The output tensor size does not match model's expectation: " +
              name_tensor.key));
      return;
    }
  }

  for (const auto& [name, array_buffer_view] : *named_outputs) {
    auto iter = outputs.value().find(name);
    if (iter == outputs.value().end()) {
      resolver->Reject(MakeGarbageCollected<DOMException>(
          DOMExceptionCode::kOperationError,
          "Failed to get result for the output " + name));
      return;
    }
    memcpy(array_buffer_view->BaseAddress(), iter->value.data(),
           iter->value.size());
  }
  auto* ml_result = MLComputeResult::Create();
  ml_result->setInputs(*named_inputs);
  ml_result->setOutputs(*named_outputs);
  resolver->Resolve(ml_result);

  resolver->Resolve();
}

void MLGraphCrOS::ComputeSyncImpl(const MLNamedArrayBufferViews& inputs,
                                  const MLNamedArrayBufferViews& outputs,
                                  ExceptionState& exception_state) {
  exception_state.ThrowDOMException(DOMExceptionCode::kNotSupportedError,
                                    "Not implemented.");
}

}  // namespace blink
