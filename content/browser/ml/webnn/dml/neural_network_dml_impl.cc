// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/browser/ml/webnn/dml/neural_network_dml_impl.h"

#include "base/memory/ptr_util.h"
#include "content/browser/ml/webnn/dml/context_dml_impl.h"
#include "mojo/public/cpp/bindings/self_owned_receiver.h"

namespace content {

namespace {

using ml::webnn::mojom::Context;
using ml::webnn::mojom::ContextOptionsPtr;
using ml::webnn::mojom::NeuralNetwork;

}  // namespace

namespace webnn {

// static
void NeuralNetwoekDMLImpl::Create(
    mojo::PendingReceiver<NeuralNetwork> receiver) {
  mojo::MakeSelfOwnedReceiver<NeuralNetwork>(
      base::WrapUnique(new NeuralNetwoekDMLImpl()), std::move(receiver));
}

NeuralNetwoekDMLImpl::~NeuralNetwoekDMLImpl() = default;

NeuralNetwoekDMLImpl::NeuralNetwoekDMLImpl() = default;

void NeuralNetwoekDMLImpl::CreateContext(
    uint32_t id,
    ContextOptionsPtr options,
    NeuralNetwork::CreateContextCallback callback) {
  // The remote sent to the renderer.
  mojo::PendingRemote<Context> blink_remote;
  // The receiver bind to ContextDMLImpl.
  ContextDMLImpl::Create(blink_remote.InitWithNewPipeAndPassReceiver());
  std::move(callback).Run(std::move(blink_remote));
}

}  // namespace webnn

}  // namespace content
