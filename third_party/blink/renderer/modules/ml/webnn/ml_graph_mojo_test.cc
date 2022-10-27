// Copyright 2019 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_mojo.h"
#include "base/run_loop.h"
#include "base/test/bind.h"
#include "base/test/scoped_feature_list.h"
#include "components/ml/mojom/webnn_graph.mojom-blink.h"
#include "components/ml/mojom/webnn_service.mojom-blink.h"
#include "mojo/public/cpp/bindings/pending_receiver.h"
#include "mojo/public/cpp/bindings/receiver.h"
#include "mojo/public/cpp/bindings/self_owned_receiver.h"
#include "testing/gmock/include/gmock/gmock.h"
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
#include "third_party/blink/renderer/modules/ml/ml.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_builder_test.h"
#include "third_party/blink/renderer/platform/testing/unit_test_helpers.h"

namespace blink {

class FakeWebnnGraph : public ml::webnn::mojom::blink::WebnnGraph {
 public:
  FakeWebnnGraph() = default;
  ~FakeWebnnGraph() override = default;

 private:
  // Override methods from ml::webnn::mojom::WebnnGraph.
};

class FakeWebnnContext : public ml::webnn::mojom::blink::WebnnContext {
 public:
  FakeWebnnContext() : receiver_(this) {}
  ~FakeWebnnContext() override = default;

  void BindRequest(mojo::ScopedMessagePipeHandle handle) {
    DCHECK(!receiver_.is_bound());
    receiver_.Bind(mojo::PendingReceiver<ml::webnn::mojom::blink::WebnnContext>(
        std::move(handle)));
    receiver_.set_disconnect_handler(WTF::BindOnce(
        &FakeWebnnContext::OnConnectionError, WTF::Unretained(this)));
  }

  bool IsBound() const { return receiver_.is_bound(); }

  void OnConnectionError() { receiver_.reset(); }

 private:
  // Override methods from ml::webnn::mojom::WebnnContext.
  void CreateGraph(ml::webnn::mojom::blink::CreateGraphOptionsPtr options,
                   CreateGraphCallback callback) override {
    mojo::PendingRemote<ml::webnn::mojom::blink::WebnnGraph> blink_remote;
    // The receiver bind to FakeWebnnGraph.
    mojo::MakeSelfOwnedReceiver<ml::webnn::mojom::blink::WebnnGraph>(
        std::make_unique<FakeWebnnGraph>(),
        blink_remote.InitWithNewPipeAndPassReceiver());

    std::move(callback).Run(ml::webnn::mojom::blink::CreateGraphResult::kOk,
                            std::move(blink_remote));
  }

  mojo::Receiver<ml::webnn::mojom::blink::WebnnContext> receiver_;
};

class FakeWebnnService {
 public:
  FakeWebnnService() = default;
  ~FakeWebnnService() = default;

  void BindFakeWebnnContext(ExecutionContext* execution_context) {
    fake_webnn_context_.reset();
    fake_webnn_context_ = std::make_unique<FakeWebnnContext>();
    execution_context->GetBrowserInterfaceBroker().SetBinderForTesting(
        ml::webnn::mojom::blink::WebnnContext::Name_,
        WTF::BindRepeating(&FakeWebnnContext::BindRequest,
                           WTF::Unretained(fake_webnn_context_.get())));
  }

  bool IsWebnnContextBound() const { return fake_webnn_context_->IsBound(); }

 private:
  std::unique_ptr<FakeWebnnContext> fake_webnn_context_;
};

// Overrides requests for Webnn mojo requests with FakeWebnnService instances.
class MLGraphMojoTest : public testing::Test {
 public:
  MLGraphMojoTest() { webnn_service_ = std::make_unique<FakeWebnnService>(); }

  FakeWebnnService* WebnnService() { return webnn_service_.get(); }

 private:
  std::unique_ptr<FakeWebnnService> webnn_service_;
};

MLGraphMojo* ToMLGraphMojo(V8TestingScope* scope, ScriptValue value) {
  return NativeValueTraits<MLGraphMojo>::NativeValue(
      scope->GetIsolate(), value.V8Value(), scope->GetExceptionState());
}

TEST_F(MLGraphMojoTest, CreateWebnnGraphTest) {
  V8TestingScope scope;
  // Bind fake Webnn Context in the service for testing.
  WebnnService()->BindFakeWebnnContext(scope.GetExecutionContext());

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
    EXPECT_FALSE(WebnnService()->IsWebnnContextBound());
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
    EXPECT_TRUE(WebnnService()->IsWebnnContextBound());
  }
}

}  // namespace blink
