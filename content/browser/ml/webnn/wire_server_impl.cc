// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/browser/ml/webnn/wire_server_impl.h"

#include "base/memory/ptr_util.h"
#include "mojo/public/cpp/bindings/self_owned_receiver.h"

namespace content {

namespace {

using ml::webnn::mojom::ContextOptionsPtr;
using ml::webnn::mojom::WireServer;

}  // namespace

namespace webnn {

// static
void WireServerImpl::Create(mojo::PendingReceiver<WireServer> receiver) {
  mojo::MakeSelfOwnedReceiver<WireServer>(
      base::WrapUnique(new WireServerImpl()), std::move(receiver));
}

WireServerImpl::~WireServerImpl() = default;

WireServerImpl::WireServerImpl() = default;

void WireServerImpl::CreateContext(uint32_t self_id,
                                   ContextOptionsPtr options,
                                   WireServer::CreateContextCallback callback) {
  // TODO(crbug.com/1273291): Supporting Webnn Service on the platform.
  std::move(callback).Run(mojo::NullRemote());
  DLOG(ERROR) << "Platform not supported for Webnn Service.";
}

}  // namespace webnn

}  // namespace content
