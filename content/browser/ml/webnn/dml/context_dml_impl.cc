// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/browser/ml/webnn/dml/context_dml_impl.h"

#include "base/memory/ptr_util.h"
#include "content/browser/ml/webnn/dml/graph_dml_impl.h"
#include "mojo/public/cpp/bindings/self_owned_receiver.h"

namespace content {

namespace {

using ml::webnn::mojom::Graph;

}  // namespace

namespace webnn {

// static
void ContextDMLImpl::Create(mojo::PendingReceiver<Context> receiver) {
  mojo::MakeSelfOwnedReceiver<Context>(base::WrapUnique(new ContextDMLImpl()),
                                       std::move(receiver));
}

ContextDMLImpl::~ContextDMLImpl() = default;

ContextDMLImpl::ContextDMLImpl() = default;

void ContextDMLImpl::CreateGraph(uint32_t self_id,
                                 uint32_t context_id,
                                 CreateGraphCallback callback) {
  // The remote sent to the renderer.
  mojo::PendingRemote<Graph> blink_remote;
  // The receiver bind to ContextDMLImpl.
  GraphDMLImpl::Create(blink_remote.InitWithNewPipeAndPassReceiver());
  std::move(callback).Run(std::move(blink_remote));
}

}  // namespace webnn

}  // namespace content
