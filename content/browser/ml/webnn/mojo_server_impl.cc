// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/browser/ml/webnn/mojo_server_impl.h"

#include "base/memory/ptr_util.h"
#include "mojo/public/cpp/bindings/self_owned_receiver.h"

namespace content::webnn {

namespace {

using ml::webnn::mojom::ContextOptionsPtr;
using ml::webnn::mojom::MojoServer;

}  // namespace

// static
void MojoServerImpl::Create(mojo::PendingReceiver<MojoServer> receiver) {
  mojo::MakeSelfOwnedReceiver<MojoServer>(
      base::WrapUnique(new MojoServerImpl()), std::move(receiver));
}

MojoServerImpl::~MojoServerImpl() = default;

MojoServerImpl::MojoServerImpl() = default;

void MojoServerImpl::CreateContext(ContextOptionsPtr options,
                                   uint32_t context_id,
                                   MojoServer::CreateContextCallback callback) {
  // TODO(crbug.com/1273291): Supporting Webnn Service on the platform.
  std::move(callback).Run(mojo::NullRemote());
  DLOG(ERROR) << "Platform not supported for Webnn Service.";
}

}  // namespace content::webnn
