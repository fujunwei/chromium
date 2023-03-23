// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_test_cros.h"

#include "third_party/blink/renderer/bindings/core/v8/native_value_traits_impl.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise_resolver.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise_tester.h"
#include "third_party/blink/renderer/bindings/core/v8/v8_dom_exception.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_conv_2d_options.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"
#include "third_party/blink/renderer/core/execution_context/execution_context.h"
#include "third_party/blink/renderer/modules/ml/ml.h"
#include "third_party/blink/renderer/modules/ml/ml_context.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_builder.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_builder_test.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_builder_utils.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_cros.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_test_base.h"
#include "third_party/blink/renderer/platform/testing/unit_test_helpers.h"
#include "third_party/tflite/src/tensorflow/lite/kernels/builtin_op_kernels.h"
#include "third_party/tflite/src/tensorflow/lite/model.h"
#include "third_party/tflite/src/tensorflow/lite/mutable_op_resolver.h"

namespace blink {

namespace blink_mojom = ml::model_loader::mojom::blink;

namespace {

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
  explicit TFLiteRuntime(mojo_base::BigBuffer buffer)
      : buffer_(std::move(buffer)) {}
  TFLiteRuntime(const TFLiteRuntime&) = delete;
  TFLiteRuntime(TFLiteRuntime&&) = delete;
  ~TFLiteRuntime() = default;

  TfLiteStatus Load(blink_mojom::ModelInfoPtr& info) {
    const tflite::Model* model = tflite::GetModel(buffer_.data());
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
  mojo_base::BigBuffer buffer_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
};

}  // namespace

class FakeMLModel : public blink_mojom::Model {
 public:
  explicit FakeMLModel(std::unique_ptr<TFLiteRuntime> runtime)
      : runtime_(std::move(runtime)) {}
  FakeMLModel(const FakeMLModel&) = delete;
  FakeMLModel(FakeMLModel&&) = delete;
  ~FakeMLModel() override = default;

 private:
  // Override methods from blink_mojom::Model.
  void Compute(const WTF::HashMap<WTF::String, WTF::Vector<uint8_t>>& input,
               blink_mojom::Model::ComputeCallback callback) override {
    NOTIMPLEMENTED();
  }

  std::unique_ptr<TFLiteRuntime> runtime_;
};

class FakeMLModelLoader : public blink_mojom::ModelLoader {
 public:
  FakeMLModelLoader() = default;
  FakeMLModelLoader(const FakeMLModelLoader&) = delete;
  FakeMLModelLoader(FakeMLModelLoader&&) = delete;
  ~FakeMLModelLoader() override = default;

 private:
  // Override methods from blink_mojom::ModelLoader.
  void Load(mojo_base::BigBuffer buffer,
            blink_mojom::ModelLoader::LoadCallback callback) override {
    std::unique_ptr<TFLiteRuntime> runtime =
        std::make_unique<TFLiteRuntime>(std::move(buffer));
    blink_mojom::ModelInfoPtr info = blink_mojom::ModelInfo::New();
    EXPECT_EQ(runtime->Load(info), kTfLiteOk);
    mojo::PendingRemote<blink_mojom::Model> blink_remote;
    // The receiver bind to FakeMLModel.
    mojo::MakeSelfOwnedReceiver<blink_mojom::Model>(
        std::make_unique<FakeMLModel>(std::move(runtime)),
        blink_remote.InitWithNewPipeAndPassReceiver());
    std::move(callback).Run(blink_mojom::LoadModelResult::kOk,
                            std::move(blink_remote), std::move(info));
  }
};

// A fake MLService that intercepts Blink's browser interface request to the
// ml.model_loader.MLService interface.
class FakeMLService : public blink_mojom::MLService {
 public:
  FakeMLService() = default;
  FakeMLService(const FakeMLService&) = delete;
  FakeMLService(FakeMLService&&) = delete;
  ~FakeMLService() override = default;

  void BindFakeService(mojo::ScopedMessagePipeHandle pipe) {
    receiver_.reset();
    receiver_.Bind(
        mojo::PendingReceiver<blink_mojom::MLService>(std::move(pipe)));
  }

 private:
  // Override methods from ml::blink_mojom::MLService.
  void CreateModelLoader(blink_mojom::CreateModelLoaderOptionsPtr options,
                         CreateModelLoaderCallback callback) override {
    mojo::PendingRemote<blink_mojom::ModelLoader> blink_remote;
    // The receiver bind to FakeMLModelLoader.
    mojo::MakeSelfOwnedReceiver<blink_mojom::ModelLoader>(
        std::make_unique<FakeMLModelLoader>(),
        blink_remote.InitWithNewPipeAndPassReceiver());

    std::move(callback).Run(blink_mojom::CreateModelLoaderResult::kOk,
                            std::move(blink_remote));
  }

  mojo::Receiver<blink_mojom::MLService> receiver_{this};
};

ScopedMLServiceBinder::ScopedMLServiceBinder(const V8TestingScope& scope)
    : ml_service_(std::make_unique<FakeMLService>()),
      interface_broker_(
          scope.GetExecutionContext()->GetBrowserInterfaceBroker()) {
  interface_broker_.SetBinderForTesting(
      blink_mojom::MLService::Name_,
      WTF::BindRepeating(&FakeMLService::BindFakeService,
                         // Safe to WTF::Unretained, we unregister the
                         // binder when the test finishes.
                         WTF::Unretained(ml_service_.get())));
}

ScopedMLServiceBinder::~ScopedMLServiceBinder() {
  interface_broker_.SetBinderForTesting(blink_mojom::MLService::Name_,
                                        base::NullCallback());
}

class MLGraphTestCrOS : public MLGraphTestBase {};

TEST_P(MLGraphTestCrOS, BuildGraphWithTfliteModel) {
  V8TestingScope scope;
  // Setup binder for MLService
  ScopedMLServiceBinder scoped_setup_binder(scope);
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
