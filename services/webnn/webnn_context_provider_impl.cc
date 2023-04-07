// Copyright 2023 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "services/webnn/webnn_context_provider_impl.h"

#include "base/memory/ptr_util.h"
#include "mojo/public/cpp/bindings/self_owned_receiver.h"

namespace webnn {

namespace {

using webnn::mojom::CreateContextOptionsPtr;
using webnn::mojom::WebNNContextProvider;

}  // namespace

// static
void WebNNContextProviderImpl::Create(
    mojo::PendingReceiver<WebNNContextProvider> receiver) {
  mojo::MakeSelfOwnedReceiver<WebNNContextProvider>(
      base::WrapUnique(new WebNNContextProviderImpl()), std::move(receiver));
}

WebNNContextProviderImpl::~WebNNContextProviderImpl() = default;

WebNNContextProviderImpl::WebNNContextProviderImpl() = default;

void WebNNContextProviderImpl::CreateWebNNContext(
    CreateContextOptionsPtr options,
    WebNNContextProvider::CreateWebNNContextCallback callback) {
  // TODO(crbug.com/1273291): Supporting WebNN Service on the platform.
  std::move(callback).Run(mojom::CreateContextResult::kNotSupported,
                          mojo::NullRemote());
  DLOG(ERROR) << "Platform not supported for WebNN Service.";
}

}  // namespace webnn
