// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/browser/ml/webnn/mojo_server_dml_impl.h"

#include "base/memory/ptr_util.h"
#include "content/browser/ml/webnn/context_dml_impl.h"
#include "mojo/public/cpp/bindings/self_owned_receiver.h"

namespace content::webnn {

namespace {

using ml::webnn::mojom::Context;
using ml::webnn::mojom::ContextOptionsPtr;
using ml::webnn::mojom::MojoServer;

}  // namespace

// static
void MojoServerDMLImpl::Create(mojo::PendingReceiver<MojoServer> receiver) {
  mojo::MakeSelfOwnedReceiver<MojoServer>(
      base::WrapUnique(new MojoServerDMLImpl()), std::move(receiver));
}

MojoServerDMLImpl::~MojoServerDMLImpl() = default;

MojoServerDMLImpl::MojoServerDMLImpl() = default;

void MojoServerDMLImpl::CreateContext(
    ContextOptionsPtr options,
    uint32_t context_id,
    MojoServer::CreateContextCallback callback) {
  // The remote sent to the renderer.
  mojo::PendingRemote<Context> blink_remote;
  // The receiver bind to ContextDMLImpl.
  ContextDMLImpl::Create(blink_remote.InitWithNewPipeAndPassReceiver());
  std::move(callback).Run(std::move(blink_remote));
}

}  // namespace content::webnn
