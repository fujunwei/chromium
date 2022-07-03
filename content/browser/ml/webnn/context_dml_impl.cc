// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/browser/ml/webnn/context_dml_impl.h"

#include "base/memory/ptr_util.h"
#include "content/browser/ml/webnn/graph_dml_impl.h"
#include "mojo/public/cpp/bindings/self_owned_receiver.h"

namespace content::webnn {

// static
void ContextDMLImpl::Create(mojo::PendingReceiver<Context> receiver) {
  mojo::MakeSelfOwnedReceiver<Context>(base::WrapUnique(new ContextDMLImpl()),
                                       std::move(receiver));
}

ContextDMLImpl::~ContextDMLImpl() = default;

ContextDMLImpl::ContextDMLImpl() = default;

void ContextDMLImpl::CreateGraph(uint32_t graph_id,
                                 CreateGraphCallback callback) {
  // The remote sent to the renderer.
  mojo::PendingRemote<ml::webnn::mojom::Graph> blink_remote;
  // The receiver bind to GraphDMLImpl.
  GraphDMLImpl::Create(blink_remote.InitWithNewPipeAndPassReceiver());
  std::move(callback).Run(std::move(blink_remote));
}

}  // namespace content::webnn
