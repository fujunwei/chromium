// Copyright 2023 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "base/run_loop.h"
#include "base/test/bind.h"
#include "base/test/task_environment.h"
#include "mojo/public/cpp/bindings/pending_receiver.h"
#include "mojo/public/cpp/bindings/remote.h"
#include "mojo/public/cpp/bindings/self_owned_receiver.h"
#include "services/webnn/public/mojom/webnn_graph.mojom.h"
#include "services/webnn/public/mojom/webnn_service.mojom.h"
#include "services/webnn/webnn_context_provider_impl_win.h"
#include "testing/gtest/include/gtest/gtest.h"

namespace webnn {

class WebnnContextImplWinTest : public testing::Test {
 public:
  WebnnContextImplWinTest(const WebnnContextImplWinTest&) = delete;
  WebnnContextImplWinTest& operator=(const WebnnContextImplWinTest&) = delete;

 protected:
  WebnnContextImplWinTest() = default;
  ~WebnnContextImplWinTest() override = default;

 private:
  base::test::TaskEnvironment task_environment_;
};

TEST_F(WebnnContextImplWinTest, CreateWebnnGraphTest) {
  mojo::Remote<mojom::WebnnContextProvider> provider_remote;
  mojo::Remote<mojom::WebnnContext> webnn_context_remote;

  WebnnContextProviderImplWin::Create(
      provider_remote.BindNewPipeAndPassReceiver());

  bool is_callback_called = false;
  base::RunLoop run_loop_create_context;
  auto options = mojom::CreateContextOptions::New();
  provider_remote->CreateWebnnContext(
      std::move(options),
      base::BindLambdaForTesting(
          [&](mojom::CreateContextResult result,
              mojo::PendingRemote<mojom::WebnnContext> remote) {
            EXPECT_EQ(result, mojom::CreateContextResult::kOk);
            webnn_context_remote.Bind(std::move(remote));
            is_callback_called = true;
            run_loop_create_context.Quit();
          }));
  run_loop_create_context.Run();
  EXPECT_TRUE(is_callback_called);

  base::RunLoop run_loop_create_graph;
  is_callback_called = false;
  webnn_context_remote->CreateGraph(base::BindLambdaForTesting(
      [&](mojo::PendingRemote<mojom::WebnnGraph> remote) {
        EXPECT_TRUE(remote.is_valid());
        is_callback_called = true;
        run_loop_create_graph.Quit();
      }));
  run_loop_create_graph.Run();
  EXPECT_TRUE(is_callback_called);
}

}  // namespace webnn
