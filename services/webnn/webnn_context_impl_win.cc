// Copyright 2023 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "services/webnn/webnn_context_impl_win.h"

#include "base/memory/ptr_util.h"
#include "mojo/public/cpp/bindings/self_owned_receiver.h"
#include "services/webnn/webnn_graph_impl_win.h"

namespace webnn {

// static
void WebnnContextImplWin::Create(
    mojo::PendingReceiver<mojom::WebnnContext> receiver) {
  mojo::MakeSelfOwnedReceiver<mojom::WebnnContext>(
      base::WrapUnique(new WebnnContextImplWin()), std::move(receiver));
}

WebnnContextImplWin::~WebnnContextImplWin() = default;

WebnnContextImplWin::WebnnContextImplWin() = default;

void WebnnContextImplWin::CreateGraph(
    mojom::WebnnContext::CreateGraphCallback callback) {
  // The remote sent to the renderer.
  mojo::PendingRemote<mojom::WebnnGraph> blink_remote;
  // The receiver bind to WebnnGraphImplWin.
  WebnnGraphImplWin::Create(blink_remote.InitWithNewPipeAndPassReceiver());

  std::move(callback).Run(std::move(blink_remote));
}

}  // namespace webnn
