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
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_builder.h"
#include "third_party/blink/renderer/platform/bindings/exception_state.h"
#include "third_party/flatbuffers/src/include/flatbuffers/flatbuffers.h"
#include "third_party/tflite/src/tensorflow/lite/schema/schema_generated.h"

namespace blink {

namespace {

using ml::model_loader::mojom::blink::CreateModelLoaderOptions;
using ml::model_loader::mojom::blink::DevicePreference;
using ml::model_loader::mojom::blink::ModelFormat;

// The version number of the Schema. Ideally all changes will be backward
// compatible. If that ever changes, we must ensure that version is the first
// entry in the new tflite root so that we can see that version is not 1.
#define TFLITE_SCHEMA_VERSION (3)

// Build the tflite model mapping with WebNN graph in the following topology:
//       [input] [constant]
//           \   /
//            add
//             |
//          [output]
flatbuffers::FlatBufferBuilder BuildTfLiteModelForTesting() {
  // Tflite model parameters information.
  const char* kModelDescription = "ElementWise binary model for testing";
  const tflite::TensorType type = tflite::TensorType_FLOAT32;
  const Vector<int32_t> dimensions = {2};
  const Vector<float> weights = {3.0, 4.0};

  flatbuffers::FlatBufferBuilder builder;
  // It is required that the first entry in the buffers of model is always an
  // empty buffer. This is so that the default buffer index of zero in Tensor
  // will always refer to a valid empty buffer.
  Vector<flatbuffers::Offset<tflite::Buffer>> buffers = {
      tflite::CreateBuffer(builder, builder.CreateVector({})),
  };
  // Create tflite |Buffer| for constant tensor.
  buffers.push_back(tflite::CreateBuffer(
      builder,
      builder.CreateVector(reinterpret_cast<const uint8_t*>(weights.data()),
                           sizeof(float) * weights.size())));

  // A list of all tflite |Tensor| used in this model.
  Vector<flatbuffers::Offset<tflite::Tensor>> tensors;
  // Create tflite |Tensor| for constant tensor.
  uint32_t lhs_buffer_index = 0;
  tensors.emplace_back(tflite::CreateTensor(
      builder, builder.CreateVector<int32_t>(dimensions), type,
      lhs_buffer_index, builder.CreateString("input")));
  // Create tflite |Tensor| for input tensor.
  uint32_t rhs_buffer_index = 1;
  tensors.emplace_back(
      tflite::CreateTensor(builder, builder.CreateVector<int32_t>(dimensions),
                           type, rhs_buffer_index));
  // Create tflite |Tensor| for output tensor.
  uint32_t output_buffer_index = 0;
  tensors.emplace_back(tflite::CreateTensor(
      builder, builder.CreateVector<int32_t>(dimensions), type,
      output_buffer_index, builder.CreateString("output")));

  // A list of all tflite |Operator| used in this model.
  Vector<flatbuffers::Offset<tflite::Operator>> operators;
  int32_t lhs_tensor_index = 0, rhs_tensor_index = 1, output_tensor_index = 2;
  Vector<int32_t> op_inputs = {lhs_tensor_index, rhs_tensor_index};
  Vector<int32_t> op_outputs = {output_tensor_index};
  operators.emplace_back(tflite::CreateOperator(
      builder, 0, builder.CreateVector<int32_t>(op_inputs),
      builder.CreateVector<int32_t>(op_outputs)));

  // Create subgraph in the model.
  Vector<int32_t> subgraph_inputs = {lhs_tensor_index};
  Vector<int32_t> subgraph_outputs = {output_tensor_index};
  flatbuffers::Offset<tflite::SubGraph> subgraph = tflite::CreateSubGraph(
      builder, builder.CreateVector(tensors.data(), tensors.size()),
      builder.CreateVector<int32_t>(subgraph_inputs),
      builder.CreateVector<int32_t>(subgraph_outputs),
      builder.CreateVector(operators.data(), operators.size()));

  flatbuffers::Offset<flatbuffers::String> description =
      builder.CreateString(kModelDescription);

  Vector<flatbuffers::Offset<tflite::OperatorCode>> operator_codes = {
      {tflite::CreateOperatorCode(builder, tflite::BuiltinOperator_ADD)}};
  flatbuffers::Offset<tflite::Model> model_buffer = tflite::CreateModel(
      builder, TFLITE_SCHEMA_VERSION,
      builder.CreateVector(operator_codes.data(), operator_codes.size()),
      builder.CreateVector(&subgraph, 1), description,
      builder.CreateVector(buffers.data(), buffers.size()));

  tflite::FinishModelBuffer(builder, model_buffer);

  return builder;
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
          "The input configuration can not be supported."));
      return;
    }
    case CreateModelLoaderResult::kOk: {
      auto* execution_context = ExecutionContext::From(script_state);
      remote_loader_.Bind(
          std::move(pending_remote),
          execution_context->GetTaskRunner(TaskType::kInternalDefault));

      auto builder = BuildTfLiteModelForTesting();
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
