// Copyright 2023 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "services/webnn/webnn_graph_impl_win.h"

#include "base/memory/ptr_util.h"
#include "mojo/public/cpp/bindings/self_owned_receiver.h"

namespace webnn {

// static
void WebNNGraphImplWin::Create(
    mojo::PendingReceiver<mojom::WebNNGraph> receiver) {
  mojo::MakeSelfOwnedReceiver<mojom::WebNNGraph>(
      base::WrapUnique(new WebNNGraphImplWin()), std::move(receiver));
}

WebNNGraphImplWin::~WebNNGraphImplWin() = default;

WebNNGraphImplWin::WebNNGraphImplWin() = default;

}  // namespace webnn
