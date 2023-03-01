// Copyright 2023 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "services/webnn/webnn_context_impl.h"

#include "base/memory/ptr_util.h"
#include "mojo/public/cpp/bindings/self_owned_receiver.h"

namespace webnn {

// static
void WebnnContextImpl::Create(
    mojo::PendingReceiver<mojom::WebnnContext> receiver) {
  mojo::MakeSelfOwnedReceiver<mojom::WebnnContext>(
      base::WrapUnique(new WebnnContextImpl()), std::move(receiver));
}

WebnnContextImpl::~WebnnContextImpl() = default;

WebnnContextImpl::WebnnContextImpl() = default;

void WebnnContextImpl::CreateGraph(
    mojom::CreateGraphOptionsPtr options,
    mojom::WebnnContext::CreateGraphCallback callback) {
  // TODO(crbug.com/1273291): Supporting Webnn Service on the platform.
  std::move(callback).Run(mojom::CreateGraphResult::kNotSupported,
                          mojo::NullRemote());
  DLOG(ERROR) << "Platform not supported for Webnn Service.";
}

}  // namespace webnn
