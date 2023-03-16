// Copyright 2023 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "services/webnn/webnn_graph_impl_win.h"

#include "base/memory/ptr_util.h"
#include "mojo/public/cpp/bindings/self_owned_receiver.h"

namespace webnn {

// static
void WebnnGraphImplWin::Create(
    mojo::PendingReceiver<mojom::WebnnGraph> receiver) {
  mojo::MakeSelfOwnedReceiver<mojom::WebnnGraph>(
      base::WrapUnique(new WebnnGraphImplWin()), std::move(receiver));
}

WebnnGraphImplWin::~WebnnGraphImplWin() = default;

WebnnGraphImplWin::WebnnGraphImplWin() = default;

}  // namespace webnn
