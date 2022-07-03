// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/browser/ml/webnn/webnn_service.h"

#include "base/no_destructor.h"
#include "content/browser/gpu/gpu_process_host.h"
#include "mojo/public/cpp/bindings/remote.h"

#if BUILDFLAG(IS_WIN)
#include "content/browser/ml/webnn/mojo_server_dml_impl.h"
#else
#include "content/browser/ml/webnn/mojo_server_impl.h"
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

void BindMojoServer(
    mojo::PendingReceiver<ml::webnn::mojom::MojoServer> receiver) {
  GetWebnnService()->BindMojoServer(std::move(receiver));
}

namespace webnn {

WebnnService::WebnnService(
    mojo::PendingReceiver<ml::webnn::mojom::WebnnService> receiver)
    : receiver_(this, std::move(receiver)) {}

WebnnService::~WebnnService() = default;

void WebnnService::BindMojoServer(
    mojo::PendingReceiver<ml::webnn::mojom::MojoServer> receiver) {
#if BUILDFLAG(IS_WIN)
  MojoServerDMLImpl::Create(std::move(receiver));
#else
  MojoServerImpl::Create(std::move(receiver));
#endif
}

}  // namespace webnn

}  // namespace content
