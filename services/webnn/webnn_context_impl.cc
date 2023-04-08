// Copyright 2023 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "services/webnn/webnn_context_impl.h"

#include "base/memory/ptr_util.h"
#include "mojo/public/cpp/bindings/self_owned_receiver.h"
#include "services/webnn/webnn_graph_impl.h"

namespace webnn {

// static
void WebNNContextImpl::Create(
    mojo::PendingReceiver<mojom::WebNNContext> receiver) {
  mojo::MakeSelfOwnedReceiver<mojom::WebNNContext>(
      base::WrapUnique(new WebNNContextImpl()), std::move(receiver));
}

WebNNContextImpl::~WebNNContextImpl() = default;

WebNNContextImpl::WebNNContextImpl() = default;

void WebNNContextImpl::CreateGraph(
    mojom::WebNNContext::CreateGraphCallback callback) {
  // The remote sent to the renderer.
  mojo::PendingRemote<mojom::WebNNGraph> blink_remote;
  // The receiver bound to WebNNGraphImpl.
  WebNNGraphImpl::Create(blink_remote.InitWithNewPipeAndPassReceiver());

  std::move(callback).Run(std::move(blink_remote));
}

}  // namespace webnn
