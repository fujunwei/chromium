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
  return tensor_info;
}

// This class maintains all the currently supported TFLite
// operations for the Chromium build of TFLite and registers them for use.
class TFLiteOpResolver : public tflite::MutableOpResolver {
 public:
  TFLiteOpResolver() {
    AddBuiltin(tflite::BuiltinOperator_RELU,
               tflite::ops::builtin::Register_RELU(), /* min_version = */ 1,
               /* max_version = */ 2);
    AddBuiltin(tflite::BuiltinOperator_ADD,
               tflite::ops::builtin::Register_ADD(),
               /* min_version = */ 1,
               /* max_version = */ 2);
    AddBuiltin(tflite::BuiltinOperator_SUB,
               tflite::ops::builtin::Register_SUB(),
               /* min_version = */ 1,
               /* max_version = */ 3);
    AddBuiltin(tflite::BuiltinOperator_MUL,
               tflite::ops::builtin::Register_MUL(),
               /* min_version = */ 1,
               /* max_version = */ 4);
    AddBuiltin(tflite::BuiltinOperator_DIV,
               tflite::ops::builtin::Register_DIV(),
               /* min_version */ 1,
               /* max_version */ 2);
    AddBuiltin(tflite::BuiltinOperator_MAXIMUM,
               tflite::ops::builtin::Register_MAXIMUM(),
               /* min_version = */ 1,
               /* max_version = */ 4);
    AddBuiltin(tflite::BuiltinOperator_MINIMUM,
               tflite::ops::builtin::Register_MINIMUM(),
               /* min_version = */ 1,
               /* max_version = */ 4);
    AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
               tflite::ops::builtin::Register_SOFTMAX(),
               /* min_version = */ 1,
               /* max_version = */ 3);
    AddBuiltin(tflite::BuiltinOperator_RESHAPE,
               tflite::ops::builtin::Register_RESHAPE());

    AddBuiltin(tflite::BuiltinOperator_AVERAGE_POOL_2D,
               tflite::ops::builtin::Register_AVERAGE_POOL_2D(),
               /* min_version */ 1,
               /* max_version */ 3);
    AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,
               tflite::ops::builtin::Register_MAX_POOL_2D(),
               /* min_version */ 1,
               /* max_version */ 3);
    AddBuiltin(tflite::BuiltinOperator_L2_POOL_2D,
               tflite::ops::builtin::Register_L2_POOL_2D());
    AddBuiltin(tflite::BuiltinOperator_CONV_2D,
               tflite::ops::builtin::Register_CONV_2D(),
               /* min_version = */ 1,
               /* max_version = */ 4);
    AddBuiltin(tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
               tflite::ops::builtin::Register_DEPTHWISE_CONV_2D(),
               /* min_version = */ 1,
               /* max_version = */ 5);
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

  TfLiteStatus Compute(
      const WTF::HashMap<WTF::String, WTF::Vector<uint8_t>>& named_input,
      WTF::HashMap<WTF::String, WTF::Vector<uint8_t>>& named_output) {
    for (auto index : interpreter_->inputs()) {
      auto* tensor = interpreter_->tensor(index);
      Vector<uint8_t> input_data = named_input.at(WTF::String(tensor->name));
      memcpy(tensor->data.raw, input_data.data(), tensor->bytes);
    }

    // Compute the graph.
    EXPECT_EQ(interpreter_->Invoke(), kTfLiteOk);

    for (auto index : interpreter_->outputs()) {
      auto* tensor = interpreter_->tensor(index);
      WTF::Vector<uint8_t> output_data(static_cast<wtf_size_t>(tensor->bytes));
      memcpy(output_data.data(), tensor->data.raw, tensor->bytes);
      named_output.insert(WTF::String(tensor->name), std::move(output_data));
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
    WTF::HashMap<WTF::String, WTF::Vector<uint8_t>> named_output;
    EXPECT_EQ(runtime_->Compute(input, named_output), kTfLiteOk);
    std::move(callback).Run(blink_mojom::ComputeResult::kOk, named_output);
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
void FakeMLService::BindFakeService(mojo::ScopedMessagePipeHandle pipe) {
  receiver_.reset();
  receiver_.Bind(
      mojo::PendingReceiver<blink_mojom::MLService>(std::move(pipe)));
}

void FakeMLService::CreateModelLoader(
    blink_mojom::CreateModelLoaderOptionsPtr options,
    CreateModelLoaderCallback callback) {
  mojo::PendingRemote<blink_mojom::ModelLoader> blink_remote;
  // The receiver bind to FakeMLModelLoader.
  mojo::MakeSelfOwnedReceiver<blink_mojom::ModelLoader>(
      std::make_unique<FakeMLModelLoader>(),
      blink_remote.InitWithNewPipeAndPassReceiver());

  std::move(callback).Run(blink_mojom::CreateModelLoaderResult::kOk,
                          std::move(blink_remote));
}

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

}  // namespace blink
