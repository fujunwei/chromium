// Copyright 2023 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "services/webnn/webnn_context_provider_impl.h"

#include "base/memory/ptr_util.h"
#include "mojo/public/cpp/bindings/self_owned_receiver.h"

namespace webnn {

namespace {

using webnn::mojom::CreateContextOptionsPtr;
using webnn::mojom::WebnnContextProvider;

}  // namespace

// static
void WebnnContextProviderImpl::Create(
    mojo::PendingReceiver<WebnnContextProvider> receiver) {
  mojo::MakeSelfOwnedReceiver<WebnnContextProvider>(
      base::WrapUnique(new WebnnContextProviderImpl()), std::move(receiver));
}

WebnnContextProviderImpl::~WebnnContextProviderImpl() = default;

WebnnContextProviderImpl::WebnnContextProviderImpl() = default;

void WebnnContextProviderImpl::CreateWebnnContext(
    CreateContextOptionsPtr options,
    WebnnContextProvider::CreateWebnnContextCallback callback) {
  // TODO(crbug.com/1273291): Supporting Webnn Service on the platform.
  std::move(callback).Run(mojom::CreateContextResult::kNotSupported,
                          mojo::NullRemote());
  DLOG(ERROR) << "Platform not supported for Webnn Service.";
}

}  // namespace webnn
