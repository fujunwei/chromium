// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_cros.h"

#include <algorithm>
#include <fstream>
#include <numeric>

#include "base/numerics/checked_math.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise_resolver.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_data_type.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"
#include "third_party/blink/renderer/modules/ml/ml.h"
#include "third_party/blink/renderer/modules/ml/ml_context.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_builder.h"
#include "third_party/blink/renderer/modules/ml/webnn/tf_lite_model_info.h"
#include "third_party/blink/renderer/platform/bindings/exception_state.h"

namespace blink {

namespace {

using ml::model_loader::mojom::blink::CreateModelLoaderOptions;
using ml::model_loader::mojom::blink::DataType;
using ml::model_loader::mojom::blink::DevicePreference;
using ml::model_loader::mojom::blink::ModelFormat;

// The version number of the Schema. Ideally all changes will be backward
// compatible. If that ever changes, we must ensure that version is the first
// entry in the new tflite root so that we can see that version is not 1.
#define TFLITE_SCHEMA_VERSION (3)

}  // namespace

// static
void MLGraphCrOS::ValidateAndBuildAsync(MLContext* ml_context,
                                        const MLNamedOperands& named_outputs,
                                        ScriptPromiseResolver* resolver,
                                        ExceptionState& exception_state) {
  auto* script_state = resolver->GetScriptState();
  auto* execution_context = ExecutionContext::From(script_state);
  auto* graph =
      MakeGarbageCollected<MLGraphCrOS>(execution_context, ml_context);
  graph->BuildAsync(named_outputs, resolver, exception_state);
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

bool Input1Static() {
  return true;
}

bool Input2Static() {
  return true;
}

std::vector<int32_t> Input1Shape() {
  return std::vector<int32_t>({1, 2, 2, 3});
}

std::vector<int32_t> Input2Shape() {
  return std::vector<int32_t>({1, 2, 2, 3});
}

std::vector<int32_t> OutputShape() {
  return std::vector<int32_t>({1, 2, 2, 3});
}

int32_t ComputeSize(const std::vector<int32_t>& shape) {
  return std::accumulate(shape.cbegin(), shape.cend(), 1,
                         std::multiplies<int32_t>());
}

std::vector<const uint8_t> CreateTfLiteModel(
    tflite::BuiltinOperator binary_op) {
  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<tflite::OperatorCode>> operator_codes{
      {tflite::CreateOperatorCode(builder, binary_op)}};

  std::vector<flatbuffers::Offset<tflite::Buffer>> buffers{{
      tflite::CreateBuffer(builder, builder.CreateVector({})),
  }};

  uint32_t input1_buffer = 0;
  if (Input1Static()) {
    std::vector<float> input1_data(ComputeSize(Input1Shape()), 2);
    // std::generate(input1_data.begin(), input1_data.end(), input1_rng);
    input1_buffer = static_cast<uint32_t>(buffers.size());

    buffers.push_back(tflite::CreateBuffer(
        builder, builder.CreateVector(
                     reinterpret_cast<const uint8_t*>(input1_data.data()),
                     sizeof(float) * input1_data.size())));
  }

  uint32_t input2_buffer = 0;
  if (Input2Static()) {
    std::vector<float> input2_data(ComputeSize(Input2Shape()), 3);
    // std::generate(input2_data.begin(), input2_data.end(), input2_rng);

    input2_buffer = static_cast<uint32_t>(buffers.size());

    buffers.push_back(tflite::CreateBuffer(
        builder, builder.CreateVector(
                     reinterpret_cast<const uint8_t*>(input2_data.data()),
                     sizeof(float) * input2_data.size())));
  }

  const std::vector<int32_t> output_shape = OutputShape();
  std::vector<flatbuffers::Offset<tflite::Tensor>> tensors;
  std::vector<flatbuffers::Offset<tflite::Operator>> operators;
  tensors.emplace_back(tflite::CreateTensor(
      builder, builder.CreateVector<int32_t>(Input1Shape()),
      tflite::TensorType_FLOAT32, input1_buffer));
  tensors.emplace_back(tflite::CreateTensor(
      builder, builder.CreateVector<int32_t>(Input2Shape()),
      tflite::TensorType_FLOAT32, input2_buffer));
  tensors.emplace_back(
      tflite::CreateTensor(builder, builder.CreateVector<int32_t>(output_shape),
                           tflite::TensorType_FLOAT32));

  // TF-Lite support activation in elementwise binary operations, but WebNN Spec
  // doesn't define the feature, so the options of elementwise binary doesn't
  // need to be configured.
  tflite::BuiltinOptions builtin_options_type = tflite::BuiltinOptions_NONE;
  flatbuffers::Offset<void> builtin_options = 0;

  const std::array<int32_t, 2> op_inputs{
      {static_cast<int>(tensors.size()) - 3,
       static_cast<int>(tensors.size()) - 2}};
  const std::array<int32_t, 1> op_outputs{
      {static_cast<int>(tensors.size()) - 1}};
  operators.emplace_back(tflite::CreateOperator(
      builder, /*opcode_index=*/0, builder.CreateVector<int32_t>(op_inputs),
      builder.CreateVector<int32_t>(op_outputs), builtin_options_type,
      builtin_options));

  std::vector<int32_t> subgraph_inputs;
  if (!Input1Static()) {
    subgraph_inputs.push_back(static_cast<int32_t>(tensors.size() - 3));
  }
  if (!Input2Static()) {
    subgraph_inputs.push_back(static_cast<int32_t>(tensors.size() - 2));
  }
  const std::array<int32_t, 1> subgraph_outputs{
      {static_cast<int>(tensors.size()) - 1}};
  flatbuffers::Offset<tflite::SubGraph> subgraph =
      tflite::CreateSubGraph(builder, builder.CreateVector(tensors),
                             builder.CreateVector<int32_t>(subgraph_inputs),
                             builder.CreateVector<int32_t>(subgraph_outputs),
                             builder.CreateVector(operators));

  flatbuffers::Offset<flatbuffers::String> description =
      builder.CreateString("Binary model");

  flatbuffers::Offset<tflite::Model> model_buffer = tflite::CreateModel(
      builder, TFLITE_SCHEMA_VERSION, builder.CreateVector(operator_codes),
      builder.CreateVector(&subgraph, 1), description,
      builder.CreateVector(buffers));

  tflite::FinishModelBuffer(builder, model_buffer);

  return std::vector<const uint8_t>(
      builder.GetBufferPointer(),
      builder.GetBufferPointer() + builder.GetSize());
}

void MLGraphCrOS::BuildAsyncImpl(const MLNamedOperands& outputs,
                                 ScriptPromiseResolver* resolver,
                                 ExceptionState& exception_state) {
  LOG(ERROR) << "==============BuildAsyncImpl";
  auto options_mojo = CreateModelLoaderOptions::New();
  options_mojo->num_threads = ml_context_->GetNumThreads();
  options_mojo->model_format = ml_context_->GetModelFormatMojoType();
  options_mojo->device_preference = ml_context_->GetDevicePreferenceMojoType();
  // TODO(crbug.com/1273291): Add power preference for power consumption.
  auto* named_outputs = MakeGarbageCollected<MLNamedOperands>(outputs);
  auto* script_state = resolver->GetScriptState();
  ml_context_->GetML()->CreateModelLoader(
      script_state, exception_state, std::move(options_mojo),
      WTF::Bind(&MLGraphCrOS::OnRemoteLoaderCreated, WrapPersistent(this),
                WrapPersistent(script_state), WrapPersistent(resolver),
                WrapPersistent(named_outputs)));

  // Test the tf-lite model converted from WebNN Graph
  // std::vector<const uint8_t> buffer =
  //     CreateTfLiteModel(tflite::BuiltinOperator_ADD);
  // ====================
  // HeapVector<Member<const MLOperand>> inputs;
  // HeapVector<Member<const MLOperand>> constants;
  // HeapVector<Member<const MLOperator>> sorted_operators;
  // MLGraphBuilder::SortOperators(*named_outputs, inputs, constants,
  //                               sorted_operators);

  // LOG(ERROR) << "========== 1";
  // auto* tf_lite_model = MakeGarbageCollected<TFLiteModelInfo>();
  // LOG(ERROR) << "========== 1.1";
  // std::vector<int32_t> subgraph_inputs;
  // for (const auto& input : inputs) {
  //   int32_t index = tf_lite_model->BuildTensor(input.Get());
  //   subgraph_inputs.push_back(index);
  // }

  // LOG(ERROR) << "========== 2";
  // for (const auto& constant : constants) {
  //   tf_lite_model->BuildBuffer(constant.Get());
  // }

  // for (const auto& op : sorted_operators) {
  //   tf_lite_model->BuildOperator(op.Get());
  // }

  // LOG(ERROR) << "========== 3";

  // std::vector<int32_t> subgraph_outputs;
  // for (const auto& [_, operand] : *named_outputs) {
  //   // Add the output operand to model.
  //   subgraph_outputs.push_back(tf_lite_model->GetTensorIndex(operand));
  // }

  // LOG(ERROR) << "========== 4";

  // // Build the model.
  // tf_lite_model->BuildModel(subgraph_inputs, subgraph_outputs);

  // // std::vector<const uint8_t> buffer =
  // //     CreateTfLiteModel(tflite::BuiltinOperator_ADD);
  // auto& builder = tf_lite_model->GetFlatBufferBuilder();
  // // =====================
  // std::ofstream file("/home/junwei/workspace/webnn/ops_test/add.tflite",
  //                    std::ios::binary);
  // if (!file.is_open()) {
  //   return;
  // }
  // file << std::string(builder.GetBufferPointer(),
  //                     builder.GetBufferPointer() + builder.GetSize());
  // file.close();
  // LOG(ERROR) << "==========Write model to the file";
}

void MLGraphCrOS::OnRemoteLoaderCreated(
    ScriptState* script_state,
    ScriptPromiseResolver* resolver,
    const MLNamedOperands* named_outputs,
    CreateModelLoaderResult result,
    mojo::PendingRemote<ModelLoader> pending_remote) {
  LOG(ERROR) << "==========OnRemoteLoaderCreated";

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
      LOG(ERROR) << "==========OnRemoteLoaderCreated kOk";
      auto* execution_context = ExecutionContext::From(script_state);

      remote_loader_.Bind(
          std::move(pending_remote),
          execution_context->GetTaskRunner(TaskType::kInternalDefault));

      // TODO::Remove the to the topological if TF-Lite doesn't need it.
      // auto* toposorted_operators =
      //     GetOperatorsInTopologicalOrder(named_outputs);
      // DCHECK(toposorted_operators != nullptr);

      HeapVector<Member<const MLOperand>> inputs;
      HeapVector<Member<const MLOperand>> constants;
      HeapVector<Member<const MLOperator>> sorted_operators;
      MLGraphBuilder::SortOperators(*named_outputs, inputs, constants,
                                    sorted_operators);

      auto* tf_lite_model = MakeGarbageCollected<TFLiteModelInfo>();
      std::vector<int32_t> subgraph_inputs;
      for (const auto& input : inputs) {
        int32_t index = tf_lite_model->BuildTensor(input.Get());
        subgraph_inputs.push_back(index);
      }

      for (const auto& constant : constants) {
        tf_lite_model->BuildBuffer(constant.Get());
      }

      for (const auto& op : sorted_operators) {
        tf_lite_model->BuildOperator(op.Get());
      }

      std::vector<int32_t> subgraph_outputs;
      for (const auto& [_, operand] : *named_outputs) {
        // Add the output operand to model.
        subgraph_outputs.push_back(tf_lite_model->GetTensorIndex(operand));
      }
      // Build the model.
      tf_lite_model->BuildModel(subgraph_inputs, subgraph_outputs);

      // std::vector<const uint8_t> buffer =
      //     CreateTfLiteModel(tflite::BuiltinOperator_ADD);
      auto& builder = tf_lite_model->GetFlatBufferBuilder();
      remote_loader_->Load(
          base::make_span(
              static_cast<const uint8_t*>(builder.GetBufferPointer()),
              builder.GetSize()),
          WTF::Bind(&MLGraphCrOS::OnRemoteModelLoad, WrapPersistent(this),
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
      // if (!CheckIOTensorByteSize(tf_lite_model)) {
      //   resolver->Reject(MakeGarbageCollected<DOMException>(
      //       DOMExceptionCode::kDataError,
      //       "Invalid IO tensor buffer byte size."));
      //   pending_remote.reset();
      //   return;
      // }

      remote_model_.Bind(
          std::move(pending_remote),
          execution_context->GetTaskRunner(TaskType::kInternalDefault));
      // Stores the model info.
      input_tensor_name_to_info_ = std::move(tensor_info->input_tensor_info);
      for (const auto& [name, _] : input_tensor_name_to_info_) {
        LOG(ERROR) << "======model input name " << name;
      }
      output_tensor_name_to_info_ = std::move(tensor_info->output_tensor_info);
      for (const auto& [name, _] : output_tensor_name_to_info_) {
        LOG(ERROR) << "======model input name " << name;
      }
      resolver->Resolve(this);

      LOG(ERROR) << "==============Load model sucessfully";
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
  for (const auto& name_tensor : inputs) {
    auto iter = input_tensor_name_to_info_.find(name_tensor.first);
    if (iter == input_tensor_name_to_info_.end()) {
      resolver->Reject(MakeGarbageCollected<DOMException>(
          DOMExceptionCode::kDataError,
          "There is unknown input: " + name_tensor.first));
      return;
    }
    if (iter->value->byte_size !=
        name_tensor.second->GetAsArrayBufferView()->byteLength()) {
      resolver->Reject(MakeGarbageCollected<DOMException>(
          DOMExceptionCode::kUnknownError, "Wrong input size."));
      return;
    }
  }
  // Fills the buffer with input tensors.
  HashMap<String, Vector<uint8_t>> input_mojo;
  for (const auto& input : inputs) {
    const auto& [name, array_buffer_or_view] = input;
    LOG(ERROR) << "========the input name configured by user " << name;
    auto* array_buffer_view =
        array_buffer_or_view->GetAsArrayBufferView().Get();
    wtf_size_t size =
        base::checked_cast<wtf_size_t>(array_buffer_view->byteLength());
    Vector<uint8_t> tensor(size);
    memcpy(tensor.data(), array_buffer_view->BaseAddress(), size);

    input_mojo.insert(name, std::move(tensor));
  }
  auto* named_outputs = MakeGarbageCollected<MLNamedArrayBufferViews>(outputs);
  remote_model_->Compute(
      std::move(input_mojo),
      WTF::Bind(&MLGraphCrOS::OnComputeResult, WrapPersistent(this),
                WrapPersistent(resolver), WrapPersistent(named_outputs)));
}

void MLGraphCrOS::OnComputeResult(
    ScriptPromiseResolver* resolver,
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
        "The number of output tensors of computation does't match the model's "
        "expectation."));
    return;
  }

  for (const auto& name_tensor : outputs.value()) {
    LOG(ERROR) << "===========output name from MLService " << name_tensor.key;
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

  for (const auto& output : *named_outputs) {
    const auto& [name, array_buffer_or_view] = output;
    auto* array_buffer_view =
        array_buffer_or_view->GetAsArrayBufferView().Get();

    auto iter = outputs.value().find(output.first);
    if (iter == outputs.value().end()) {
      resolver->Reject(MakeGarbageCollected<DOMException>(
          DOMExceptionCode::kOperationError,
          "Failed to get result for the output " + output.first));
      return;
    }
    memcpy(array_buffer_view->BaseAddress(), iter->value.data(),
           iter->value.size());
  }

  resolver->Resolve();
}

}  // namespace blink
