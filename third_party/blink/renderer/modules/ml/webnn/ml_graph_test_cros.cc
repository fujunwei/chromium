// Copyright 2023 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "components/ml/mojom/ml_service.mojom-blink.h"
#include "components/ml/mojom/web_platform_model.mojom-blink.h"
#include "mojo/public/cpp/bindings/pending_remote.h"
#include "mojo/public/cpp/bindings/receiver.h"
#include "mojo/public/cpp/bindings/self_owned_receiver.h"
#include "third_party/blink/renderer/bindings/core/v8/native_value_traits_impl.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise_resolver.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise_tester.h"
#include "third_party/blink/renderer/bindings/core/v8/v8_dom_exception.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_conv_2d_options.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"
#include "third_party/blink/renderer/core/execution_context/execution_context.h"
#include "third_party/blink/renderer/modules/ml/ml.h"
#include "third_party/blink/renderer/modules/ml/ml_context.h"
#include "third_party/blink/renderer/modules/ml/ml_model_loader_test_util.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_builder.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_builder_test.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_builder_utils.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_cros.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_test_base.h"
#include "third_party/blink/renderer/platform/testing/unit_test_helpers.h"
#include "third_party/flatbuffers/src/include/flatbuffers/flatbuffers.h"
#include "third_party/tflite/src/tensorflow/lite/kernels/builtin_op_kernels.h"
#include "third_party/tflite/src/tensorflow/lite/model.h"
#include "third_party/tflite/src/tensorflow/lite/mutable_op_resolver.h"
#include "third_party/tflite/src/tensorflow/lite/schema/schema_generated.h"

namespace blink {

namespace blink_mojom = ml::model_loader::mojom::blink;

namespace {

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
flatbuffers::DetachedBuffer BuildTfLiteModelForTesting() {
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

  return builder.Release();
}

blink_mojom::TensorInfoPtr ConvertToMojom(const TfLiteTensor* tensor) {
  auto tensor_info = blink_mojom::TensorInfo::New();
  tensor_info->byte_size = base::checked_cast<uint32_t>(tensor->bytes);
  WTF::Vector<uint32_t> dims;
  dims.reserve(tensor->dims->size);
  for (int32_t i = 0; i < tensor->dims->size; ++i) {
    dims.push_back(tensor->dims->data[i]);
  }
  tensor_info->dimensions = std::move(dims);
  return tensor_info;
}

// This class maintains all the currently supported TFLite
// operations for the Chromium build of TFLite and registers them for use.
class TFLiteOpResolver : public tflite::MutableOpResolver {
 public:
  TFLiteOpResolver() {
    AddBuiltin(tflite::BuiltinOperator_ADD,
               tflite::ops::builtin::Register_ADD(),
               /* min_version = */ 1,
               /* max_version = */ 2);
  }
};

class TFLiteRuntime {
 public:
  TFLiteRuntime() = default;
  TFLiteRuntime(const TFLiteRuntime&) = delete;
  TFLiteRuntime(TFLiteRuntime&&) = delete;
  ~TFLiteRuntime() = default;

  TfLiteStatus Load(mojo_base::BigBuffer& buffer,
                    blink_mojom::ModelInfoPtr& info) {
    bool empty_buffer = buffer.size() == 0 ? true : false;
    auto flat_buffer_for_testing = empty_buffer ? BuildTfLiteModelForTesting()
                                                : flatbuffers::DetachedBuffer();
    const tflite::Model* model = tflite::GetModel(
        empty_buffer ? flat_buffer_for_testing.data() : buffer.data());
    EXPECT_NE(model, nullptr);
    TFLiteOpResolver op_resolver;
    EXPECT_EQ(tflite::InterpreterBuilder(model, op_resolver)(&interpreter_),
              kTfLiteOk);
    EXPECT_NE(interpreter_, nullptr);
    EXPECT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

    for (auto index : interpreter_->inputs()) {
      auto* tensor = interpreter_->tensor(index);
      info->input_tensor_info.insert(WTF::String(tensor->name),
                                     ConvertToMojom(tensor));
    }

    for (auto index : interpreter_->outputs()) {
      auto* tensor = interpreter_->tensor(index);
      info->output_tensor_info.insert(WTF::String(tensor->name),
                                      ConvertToMojom(tensor));
    }
    return kTfLiteOk;
  }

 private:
  std::unique_ptr<tflite::Interpreter> interpreter_;
};

}  // namespace

class FakeMLModelWithTfLite : public FakeMLModel {
 public:
  FakeMLModelWithTfLite() = default;
  FakeMLModelWithTfLite(const FakeMLModelWithTfLite&) = delete;
  FakeMLModelWithTfLite(FakeMLModelWithTfLite&&) = delete;
  ~FakeMLModelWithTfLite() override = default;

  FakeMLModelLoader::LoadFn CreateFromThis() {
    return WTF::BindOnce(&FakeMLModelWithTfLite::OnCreateModel,
                         WTF::Unretained(this));
  }

 private:
  void OnCreateModel(mojo_base::BigBuffer buffer,
                     blink_mojom::ModelLoader::LoadCallback callback) {
    std::unique_ptr<TFLiteRuntime> runtime = std::make_unique<TFLiteRuntime>();
    blink_mojom::ModelInfoPtr info = blink_mojom::ModelInfo::New();
    EXPECT_EQ(runtime->Load(buffer, info), kTfLiteOk);
    FakeMLModel::SetModelInfo(std::move(info));
    FakeMLModel::OnCreateModel(std::move(buffer), std::move(callback));
  }
};

class MLGraphTestCrOS : public MLGraphTestBase {
 public:
  void SetUp() override {
    // Bind the receiver of `ModelLoader` mojo interface with
    // `FakeMLModelLoader` when calling MLService::CreateModelLoader() method.
    service_.SetCreateModelLoader(loader_.CreateFromThis());
    // Bind the receiver of `Model` mojo interface with `FakeMLModel` when
    // calling ModelLoader::Load() method.
    loader_.SetLoad(model_.CreateFromThis());
  }

 protected:
  FakeMLService service_;
  FakeMLModelLoader loader_;
  FakeMLModelWithTfLite model_;
};

TEST_P(MLGraphTestCrOS, BuildGraphWithTfliteModel) {
  V8TestingScope scope;
  // Setup binder for MLService
  ScopedSetMLServiceBinder scoped_setup_binder(&service_, scope);
  // Test building graph for the operands in the following topology:
  //       [input] [constant]
  //           \   /
  //            add
  //             |
  //          [output]
  const V8MLOperandType::Enum type = V8MLOperandType::Enum::kFloat32;
  const Vector<uint32_t> dimensions = {2};
  const Vector<float> weights = {3.0, 4.0};

  auto* builder = CreateMLGraphBuilder(scope.GetExecutionContext());
  auto* input =
      BuildInput(builder, "input", dimensions, type, scope.GetExceptionState());
  auto* constant = BuildConstant(builder, dimensions, type, weights,
                                 scope.GetExceptionState());
  auto* output = BuildElementWiseBinary(
      scope, builder, ElementWiseBinaryKind::kAdd, input, constant);
  auto [graph, exception] = BuildGraph(scope, builder, {{"output", output}});
  EXPECT_NE(graph, nullptr);
  MLGraphCrOS* cros_graph = static_cast<MLGraphCrOS*>(graph.Get());
  const auto& input_tensor_info = cros_graph->GetInputTensorInfoMapForTesting();
  EXPECT_EQ(input_tensor_info.size(), 1u);
  EXPECT_EQ(input_tensor_info.Contains("input"), true);
  EXPECT_EQ(input_tensor_info.find("input")->value->dimensions, dimensions);
  const auto& output_tensor_info =
      cros_graph->GetOutputTensorInfoMapForTesting();
  EXPECT_EQ(output_tensor_info.size(), 1u);
  EXPECT_EQ(output_tensor_info.Contains("output"), true);
  EXPECT_EQ(output_tensor_info.find("output")->value->dimensions, dimensions);
}

INSTANTIATE_TEST_SUITE_P(
    All,
    MLGraphTestCrOS,
    testing::Combine(::testing::Values(BackendType::kModelLoader),
                     ::testing::Values(ExecutionMode::kAsync)),
    TestVarietyToString);

}  // namespace blink
