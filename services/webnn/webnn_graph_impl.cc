// Copyright 2023 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "services/webnn/webnn_graph_impl.h"

#include "base/memory/ptr_util.h"
#include "mojo/public/cpp/bindings/self_owned_receiver.h"

namespace webnn {

// static
void WebNNGraphImpl::Create(mojo::PendingReceiver<mojom::WebNNGraph> receiver) {
  mojo::MakeSelfOwnedReceiver<mojom::WebNNGraph>(
      base::WrapUnique(new WebNNGraphImpl()), std::move(receiver));
}

WebNNGraphImpl::~WebNNGraphImpl() = default;

WebNNGraphImpl::WebNNGraphImpl() = default;

}  // namespace webnn
