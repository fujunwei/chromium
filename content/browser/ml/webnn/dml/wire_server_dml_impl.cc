// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/browser/ml/webnn/dml/wire_server_dml_impl.h"

#include "base/memory/ptr_util.h"
#include "content/browser/ml/webnn/dml/context_dml_impl.h"
#include "mojo/public/cpp/bindings/self_owned_receiver.h"

namespace content {

namespace {

using ml::webnn::mojom::Context;
using ml::webnn::mojom::ContextOptionsPtr;
using ml::webnn::mojom::WireServer;

}  // namespace

namespace webnn {

// static
void WireServerDMLImpl::Create(mojo::PendingReceiver<WireServer> receiver) {
  mojo::MakeSelfOwnedReceiver<WireServer>(
      base::WrapUnique(new WireServerDMLImpl()), std::move(receiver));
}

WireServerDMLImpl::~WireServerDMLImpl() = default;

WireServerDMLImpl::WireServerDMLImpl() = default;

void WireServerDMLImpl::CreateContext(
    uint32_t id,
    ContextOptionsPtr options,
    WireServer::CreateContextCallback callback) {
  // The remote sent to the renderer.
  mojo::PendingRemote<Context> blink_remote;
  // The receiver bind to ContextDMLImpl.
  ContextDMLImpl::Create(blink_remote.InitWithNewPipeAndPassReceiver());
  std::move(callback).Run(std::move(blink_remote));
}

}  // namespace webnn

}  // namespace content
