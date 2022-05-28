// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/browser/ml/webnn/webnn_service.h"

#include "base/no_destructor.h"
#include "content/browser/gpu/gpu_process_host.h"
#include "mojo/public/cpp/bindings/remote.h"

#if BUILDFLAG(IS_WIN)
#include "content/browser/ml/webnn/dml/neural_network_dml_impl.h"
#else
#include "content/browser/ml/webnn/neural_network_impl.h"
#endif

namespace content {

ml::webnn::mojom::WebnnService* GetWebnnService() {
  static base::NoDestructor<mojo::Remote<ml::webnn::mojom::WebnnService>>
      remote;
  if (!*remote) {
    auto* gpu = GpuProcessHost::Get();
    if (gpu) {
      gpu->RunService(remote->BindNewPipeAndPassReceiver());
    }
    remote->reset_on_disconnect();
  }

  return remote->get();
}

void BindNeuralNetwork(
    mojo::PendingReceiver<ml::webnn::mojom::NeuralNetwork> receiver) {
  GetWebnnService()->BindNeuralNetwork(std::move(receiver));
}

namespace webnn {

WebnnService::WebnnService(
    mojo::PendingReceiver<ml::webnn::mojom::WebnnService> receiver)
    : receiver_(this, std::move(receiver)) {}

WebnnService::~WebnnService() = default;

void WebnnService::BindNeuralNetwork(
    mojo::PendingReceiver<ml::webnn::mojom::NeuralNetwork> receiver) {
#if BUILDFLAG(IS_WIN)
  NeuralNetwoekDMLImpl::Create(std::move(receiver));
#else
  NeuralNetwoekImpl::Create(std::move(receiver));
#endif
}

}  // namespace webnn

}  // namespace content
