// Copyright 2023 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "services/webnn/webnn_context_impl_win.h"

#include "base/memory/ptr_util.h"
#include "mojo/public/cpp/bindings/self_owned_receiver.h"
#include "services/webnn/webnn_graph_impl_win.h"

namespace webnn {

// static
void WebNNContextImplWin::Create(
    mojo::PendingReceiver<mojom::WebNNContext> receiver) {
  mojo::MakeSelfOwnedReceiver<mojom::WebNNContext>(
      base::WrapUnique(new WebNNContextImplWin()), std::move(receiver));
}

WebNNContextImplWin::~WebNNContextImplWin() = default;

WebNNContextImplWin::WebNNContextImplWin() = default;

void WebNNContextImplWin::CreateGraph(
    mojom::WebNNContext::CreateGraphCallback callback) {
  // The remote sent to the renderer.
  mojo::PendingRemote<mojom::WebNNGraph> blink_remote;
  // The receiver bound to WebNNGraphImplWin.
  WebNNGraphImplWin::Create(blink_remote.InitWithNewPipeAndPassReceiver());

  std::move(callback).Run(std::move(blink_remote));
}

}  // namespace webnn
