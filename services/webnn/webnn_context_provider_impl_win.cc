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
using mojom::WebnnContextProvider;

}  // namespace

// static
void WebnnContextProviderImplWin::Create(
    mojo::PendingReceiver<WebnnContextProvider> receiver) {
  mojo::MakeSelfOwnedReceiver<WebnnContextProvider>(
      base::WrapUnique(new WebnnContextProviderImplWin()), std::move(receiver));
}

WebnnContextProviderImplWin::~WebnnContextProviderImplWin() = default;

WebnnContextProviderImplWin::WebnnContextProviderImplWin() = default;

void WebnnContextProviderImplWin::CreateWebnnContext(
    CreateContextOptionsPtr options,
    WebnnContextProvider::CreateWebnnContextCallback callback) {
  // The remote sent to the renderer.
  mojo::PendingRemote<mojom::WebnnContext> blink_remote;
  // The receiver bind to WebnnContextImplWin.
  WebnnContextImplWin::Create(blink_remote.InitWithNewPipeAndPassReceiver());
  // mojo::MakeSelfOwnedReceiver<WebnnContext>(
  //     base::WrapUnique(new WebnnContextImplWin()),
  //     blink_remote.InitWithNewPipeAndPassReceiver());

  std::move(callback).Run(mojom::CreateContextResult::kOk, std::move(blink_remote));
}

}  // namespace webnn
