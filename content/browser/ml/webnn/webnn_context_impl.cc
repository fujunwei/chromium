// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/browser/ml/webnn/webnn_context_impl.h"

#include "base/memory/ptr_util.h"
#include "mojo/public/cpp/bindings/self_owned_receiver.h"

namespace content::webnn {

namespace {

using ml::webnn::mojom::CreateGraphOptionsPtr;
using ml::webnn::mojom::CreateGraphResult;
using ml::webnn::mojom::WebnnContext;

}  // namespace

// static
void WebnnContextImpl::Create(mojo::PendingReceiver<WebnnContext> receiver) {
  mojo::MakeSelfOwnedReceiver<WebnnContext>(
      base::WrapUnique(new WebnnContextImpl()), std::move(receiver));
}

WebnnContextImpl::~WebnnContextImpl() = default;

WebnnContextImpl::WebnnContextImpl() = default;

void WebnnContextImpl::CreateGraph(CreateGraphOptionsPtr options,
                                   WebnnContext::CreateGraphCallback callback) {
  // TODO(crbug.com/1273291): Supporting Webnn Service on the platform.
  std::move(callback).Run(CreateGraphResult::kNotSupported, mojo::NullRemote());
  DLOG(ERROR) << "Platform not supported for Webnn Service.";
}

}  // namespace content::webnn
