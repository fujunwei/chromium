// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/browser/ml/webnn/neural_network_impl.h"

#include "base/memory/ptr_util.h"
#include "mojo/public/cpp/bindings/self_owned_receiver.h"

namespace content {

namespace {

using ml::webnn::mojom::ContextOptionsPtr;
using ml::webnn::mojom::NeuralNetwork;

}  // namespace

namespace webnn {

// static
void NeuralNetwoekImpl::Create(mojo::PendingReceiver<NeuralNetwork> receiver) {
  mojo::MakeSelfOwnedReceiver<NeuralNetwork>(
      base::WrapUnique(new NeuralNetwoekImpl()), std::move(receiver));
}

NeuralNetwoekImpl::~NeuralNetwoekImpl() = default;

NeuralNetwoekImpl::NeuralNetwoekImpl() = default;

void NeuralNetwoekImpl::CreateContext(
    uint32_t self_id,
    ContextOptionsPtr options,
    NeuralNetwork::CreateContextCallback callback) {
  // TODO(crbug.com/1273291): Supporting Webnn Service on the platform.
  std::move(callback).Run(mojo::NullRemote());
  DLOG(ERROR) << "Platform not supported for Webnn Service.";
}

void NeuralNetwoekImpl::CreateGraph(uint32_t self_id,
                                    uint32_t context_id,
                                    CreateGraphCallback callback) {
  // TODO(crbug.com/1273291): Supporting Webnn Service on the platform.
  std::move(callback).Run(mojo::NullRemote());
  DLOG(ERROR) << "Platform not supported for Webnn Service.";
}

}  // namespace webnn

}  // namespace content
