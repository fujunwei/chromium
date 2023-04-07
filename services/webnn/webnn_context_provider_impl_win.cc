// Copyright 2023 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "services/webnn/webnn_context_provider_impl_win.h"

#include "base/memory/ptr_util.h"
#include "mojo/public/cpp/bindings/self_owned_receiver.h"
#include "services/webnn/webnn_context_impl_win.h"

namespace webnn {

namespace {

using mojom::CreateContextOptionsPtr;
using mojom::WebNNContextProvider;

}  // namespace

// static
void WebNNContextProviderImplWin::Create(
    mojo::PendingReceiver<WebNNContextProvider> receiver) {
  mojo::MakeSelfOwnedReceiver<WebNNContextProvider>(
      base::WrapUnique(new WebNNContextProviderImplWin()), std::move(receiver));
}

WebNNContextProviderImplWin::~WebNNContextProviderImplWin() = default;

WebNNContextProviderImplWin::WebNNContextProviderImplWin() = default;

void WebNNContextProviderImplWin::CreateWebNNContext(
    CreateContextOptionsPtr options,
    WebNNContextProvider::CreateWebNNContextCallback callback) {
  // The remote sent to the renderer.
  mojo::PendingRemote<mojom::WebNNContext> blink_remote;
  // The receiver bind to WebNNContextImplWin.
  WebNNContextImplWin::Create(blink_remote.InitWithNewPipeAndPassReceiver());

  std::move(callback).Run(mojom::CreateContextResult::kOk,
                          std::move(blink_remote));
}

}  // namespace webnn
