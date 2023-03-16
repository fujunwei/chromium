// Copyright 2023 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_mojo.h"
#include "base/run_loop.h"
#include "base/test/bind.h"
#include "base/test/scoped_feature_list.h"
#include "mojo/public/cpp/bindings/pending_receiver.h"
#include "mojo/public/cpp/bindings/receiver.h"
#include "mojo/public/cpp/bindings/self_owned_receiver.h"
#include "services/webnn/public/mojom/webnn_graph.mojom-blink.h"
#include "services/webnn/public/mojom/webnn_service.mojom-blink.h"
#include "testing/gtest/include/gtest/gtest.h"
#include "third_party/blink/public/common/browser_interface_broker_proxy.h"
#include "third_party/blink/public/common/features.h"
#include "third_party/blink/public/platform/platform.h"
#include "third_party/blink/renderer/bindings/core/v8/native_value_traits_impl.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise_resolver.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise_tester.h"
#include "third_party/blink/renderer/bindings/core/v8/v8_binding_for_testing.h"
#include "third_party/blink/renderer/bindings/core/v8/v8_dom_exception.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_context_options.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_builder_test.h"

namespace blink {

namespace blink_mojom = webnn::mojom::blink;

class FakeWebnnGraph : public blink_mojom::WebnnGraph {
 public:
  FakeWebnnGraph() = default;
  ~FakeWebnnGraph() override = default;

 private:
  // Override methods from webnn::mojom::WebnnGraph.
  // TODO(crbug.com/1273291): Add build and compute methods.
};

class FakeWebnnContext : public blink_mojom::WebnnContext {
 public:
  FakeWebnnContext() = default;
  ~FakeWebnnContext() override = default;

 private:
  // Override methods from webnn::mojom::WebnnContext.
  void CreateGraph(CreateGraphCallback callback) override {
    mojo::PendingRemote<blink_mojom::WebnnGraph> blink_remote;
    // The receiver bind to FakeWebnnGraph.
    mojo::MakeSelfOwnedReceiver<blink_mojom::WebnnGraph>(
        std::make_unique<FakeWebnnGraph>(),
        blink_remote.InitWithNewPipeAndPassReceiver());

    std::move(callback).Run(std::move(blink_remote));
  }
};

class FakeWebnnContextProvider : public blink_mojom::WebnnContextProvider {
 public:
  FakeWebnnContextProvider() : receiver_(this) {}
  ~FakeWebnnContextProvider() override = default;

  void BindRequest(mojo::ScopedMessagePipeHandle handle) {
    DCHECK(!receiver_.is_bound());
    receiver_.Bind(mojo::PendingReceiver<blink_mojom::WebnnContextProvider>(
        std::move(handle)));
    receiver_.set_disconnect_handler(WTF::BindOnce(
        &FakeWebnnContextProvider::OnConnectionError, WTF::Unretained(this)));
  }

  bool IsBound() const { return receiver_.is_bound(); }

  void OnConnectionError() { receiver_.reset(); }

 private:
  // Override methods from webnn::mojom::WebnnContextProvider.
  void CreateWebnnContext(blink_mojom::CreateContextOptionsPtr options,
                          CreateWebnnContextCallback callback) override {
    mojo::PendingRemote<blink_mojom::WebnnContext> blink_remote;
    // The receiver bind to FakeWebnnContext.
    mojo::MakeSelfOwnedReceiver<blink_mojom::WebnnContext>(
        std::make_unique<FakeWebnnContext>(),
        blink_remote.InitWithNewPipeAndPassReceiver());

    std::move(callback).Run(blink_mojom::CreateContextResult::kOk,
                            std::move(blink_remote));
  }

  mojo::Receiver<blink_mojom::WebnnContextProvider> receiver_;
};

class ScopedWebnnServiceBinder {
 public:
  explicit ScopedWebnnServiceBinder(V8TestingScope& scope)
      : fake_webnn_context_provider_(
            std::make_unique<FakeWebnnContextProvider>()),
        interface_broker_(
            scope.GetExecutionContext()->GetBrowserInterfaceBroker()) {
    interface_broker_.SetBinderForTesting(
        blink_mojom::WebnnContextProvider::Name_,
        WTF::BindRepeating(
            &FakeWebnnContextProvider::BindRequest,
            WTF::Unretained(fake_webnn_context_provider_.get())));
  }

  ~ScopedWebnnServiceBinder() {
    interface_broker_.SetBinderForTesting(
        blink_mojom::WebnnContextProvider::Name_, base::NullCallback());
  }

  bool IsWebnnContextBound() const {
    return fake_webnn_context_provider_->IsBound();
  }

 private:
  std::unique_ptr<FakeWebnnContextProvider> fake_webnn_context_provider_;
  const BrowserInterfaceBrokerProxy& interface_broker_;
};

class MLGraphMojoTest : public testing::Test {};

MLGraphMojo* ToMLGraphMojo(V8TestingScope* scope, ScriptValue value) {
  return NativeValueTraits<MLGraphMojo>::NativeValue(
      scope->GetIsolate(), value.V8Value(), scope->GetExceptionState());
}

// Build a simple MLGraph asynchronously with only one relu operator.
ScriptPromise BuildSimpleGraph(V8TestingScope& scope,
                               MLContextOptions* context_options) {
  auto* builder =
      CreateMLGraphBuilder(scope.GetExecutionContext(), context_options);
  auto* input =
      BuildInput(builder, "input", {3, 4, 5}, V8MLOperandType::Enum::kFloat32,
                 scope.GetExceptionState());
  auto* output = builder->relu(input, scope.GetExceptionState());
  EXPECT_NE(output, nullptr);
  return builder->build(scope.GetScriptState(), {{"output", output}},
                        scope.GetExceptionState());
}

TEST_F(MLGraphMojoTest, CreateWebnnGraphTest) {
  V8TestingScope scope;
  // Bind fake Webnn Context in the service for testing.
  ScopedWebnnServiceBinder scoped_setup_binder(scope);

  auto* script_state = scope.GetScriptState();
  auto* options = MLContextOptions::Create();
  // Create Webnn Context with GPU device preference.
  options->setDevicePreference(V8MLDevicePreference::Enum::kGpu);

  {
    // Test disabling WebNN Service by default. The promise should be rejected
    // since the WebNN Service is disabled.
    ScriptPromiseTester tester(script_state, BuildSimpleGraph(scope, options));
    tester.WaitUntilSettled();
    EXPECT_TRUE(tester.IsRejected());
    auto* exception = V8DOMException::ToImplWithTypeCheck(
        scope.GetIsolate(), tester.Value().V8Value());
    EXPECT_NE(exception, nullptr);
    EXPECT_EQ(exception->name(), "NotSupportedError");
    EXPECT_EQ(exception->message(), "Not implemented");
    EXPECT_FALSE(scoped_setup_binder.IsWebnnContextBound());
  }

  {
    // Test enabling WebNN Service in feature list. The promise should be
    // resoveld with an MLGraphMojo object.
    base::test::ScopedFeatureList scoped_feature_list;
    scoped_feature_list.InitAndEnableFeature(
        blink::features::kEnableMachineLearningNeuralNetworkService);

    ScriptPromiseTester tester(script_state, BuildSimpleGraph(scope, options));
    tester.WaitUntilSettled();
    EXPECT_TRUE(tester.IsFulfilled());
    auto* mojo_graph = ToMLGraphMojo(&scope, tester.Value());
    EXPECT_NE(mojo_graph, nullptr);
    EXPECT_TRUE(scoped_setup_binder.IsWebnnContextBound());
  }
}

}  // namespace blink
