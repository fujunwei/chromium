// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/browser/ml/webnn/dml/context_dml_impl.h"

#include "base/memory/ptr_util.h"
#include "mojo/public/cpp/bindings/self_owned_receiver.h"

namespace content {

namespace {

using ml::webnn::mojom::Context;

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
  // TODO(crbug.com/1273291): Supporting Webnn Service on the platform.
  std::move(callback).Run(mojo::NullRemote());
  DLOG(ERROR) << "Platform not supported for Webnn Service.";
}

}  // namespace webnn

}  // namespace content
