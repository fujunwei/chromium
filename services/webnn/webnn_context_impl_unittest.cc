// Copyright 2023 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "base/run_loop.h"
#include "base/test/task_environment.h"
#include "mojo/public/cpp/bindings/pending_receiver.h"
#include "mojo/public/cpp/bindings/remote.h"
#include "mojo/public/cpp/bindings/self_owned_receiver.h"
#include "services/webnn/public/mojom/webnn_graph.mojom.h"
#include "services/webnn/public/mojom/webnn_service.mojom.h"
#include "testing/gtest/include/gtest/gtest.h"

namespace webnn {

namespace {

void CreateGraphCallback(base::OnceClosure quit_closure,
                         mojom::CreateGraphResult result,
                         mojo::PendingRemote<mojom::WebnnGraph> remote) {
  EXPECT_EQ(result, mojom::CreateGraphResult::kOk);
  EXPECT_TRUE(remote.is_valid());
  std::move(quit_closure).Run();
}

class FakeWebnnGraph : public mojom::WebnnGraph {
 public:
  FakeWebnnGraph() = default;
  ~FakeWebnnGraph() override = default;

  FakeWebnnGraph(const FakeWebnnGraph&) = delete;
  FakeWebnnGraph& operator=(const FakeWebnnGraph&) = delete;

 private:
  // mojom::WebnnGraph
};

class FakeWebnnContext final : public mojom::WebnnContext {
 public:
  static void Create(mojo::PendingReceiver<mojom::WebnnContext> receiver) {
    mojo::MakeSelfOwnedReceiver<mojom::WebnnContext>(
        base::WrapUnique(new FakeWebnnContext()), std::move(receiver));
  }

  ~FakeWebnnContext() override = default;

  FakeWebnnContext(const FakeWebnnContext&) = delete;
  FakeWebnnContext& operator=(const FakeWebnnContext&) = delete;

 protected:
  FakeWebnnContext() = default;

 private:
  // mojom::WebnnContext
  void CreateGraph(mojom::CreateGraphOptionsPtr options,
                   CreateGraphCallback callback) override {
    // The remote sent to the renderer.
    mojo::PendingRemote<mojom::WebnnGraph> blink_remote;
    // The receiver bind to FakeWebnnGraph.
    mojo::MakeSelfOwnedReceiver<mojom::WebnnGraph>(
        base::WrapUnique(new FakeWebnnGraph()),
        blink_remote.InitWithNewPipeAndPassReceiver());

    std::move(callback).Run(mojom::CreateGraphResult::kOk,
                            std::move(blink_remote));
  }
};

}  // namespace

class WebnnContextImplTest : public testing::Test {
 public:
  WebnnContextImplTest(const WebnnContextImplTest&) = delete;
  WebnnContextImplTest& operator=(const WebnnContextImplTest&) = delete;

 protected:
  WebnnContextImplTest() = default;
  ~WebnnContextImplTest() override = default;

 private:
  base::test::TaskEnvironment task_environment_;
};

TEST_F(WebnnContextImplTest, CreateWebnnGraphTest) {
  mojo::Remote<mojom::WebnnContext> webnn_context;
  FakeWebnnContext::Create(webnn_context.BindNewPipeAndPassReceiver());

  base::RunLoop run_loop;
  auto options = mojom::CreateGraphOptions::New();
  webnn_context->CreateGraph(
      std::move(options),
      base::BindOnce(&CreateGraphCallback, run_loop.QuitClosure()));
  run_loop.Run();
}

}  // namespace webnn
