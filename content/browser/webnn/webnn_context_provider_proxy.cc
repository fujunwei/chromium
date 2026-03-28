// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/browser/webnn/webnn_context_provider_proxy.h"

#include <utility>

#include "components/viz/host/gpu_client.h"
#include "content/browser/renderer_host/render_process_host_impl.h"
#include "content/public/browser/render_process_host.h"
#include "mojo/public/cpp/bindings/generic_pending_receiver.h"
#include "services/webnn/error.h"
#include "services/webnn/public/mojom/webnn_error.mojom.h"

namespace content {

WebNNContextProviderProxy::WebNNContextProviderProxy(int render_process_id,
                                                     bool is_incognito)
    : render_process_id_(render_process_id), is_incognito_(is_incognito) {}

WebNNContextProviderProxy::~WebNNContextProviderProxy() = default;

void WebNNContextProviderProxy::CreateWebNNContext(
    webnn::mojom::CreateContextOptionsPtr options,
    CreateWebNNContextCallback callback) {
  EnsureGpuProviderConnection();
  if (!gpu_provider_.is_bound()) {
    // GPU process not available, try TFLite directly.
    FallbackToTFLite(std::move(options), std::move(callback));
    return;
  }

  auto options_clone = options->Clone();
  gpu_provider_->CreateWebNNContext(
      std::move(options),
      base::BindOnce(&WebNNContextProviderProxy::OnGpuContextCreated,
                     weak_factory_.GetWeakPtr(), std::move(options_clone),
                     std::move(callback)));
}

void WebNNContextProviderProxy::OnGpuContextCreated(
    webnn::mojom::CreateContextOptionsPtr options,
    CreateWebNNContextCallback callback,
    webnn::mojom::CreateContextResultPtr result) {
  if (result->is_success()) {
    std::move(callback).Run(std::move(result));
    return;
  }

  // GPU backend (ORT/DML) failed, try TFLite in render process.
  FallbackToTFLite(std::move(options), std::move(callback));
}

void WebNNContextProviderProxy::FallbackToTFLite(
    webnn::mojom::CreateContextOptionsPtr options,
    CreateWebNNContextCallback callback) {
  EnsureTFLiteProviderConnection();
  if (!tflite_provider_.is_bound()) {
    std::move(callback).Run(webnn::ToError<webnn::mojom::CreateContextResult>(
        webnn::mojom::Error::Code::kNotSupportedError,
        "WebNN service is not available."));
    return;
  }

  tflite_provider_->CreateWebNNContext(std::move(options),
                                       std::move(callback));
}

void WebNNContextProviderProxy::EnsureGpuProviderConnection() {
  if (gpu_provider_.is_bound()) {
    return;
  }

  auto* process_host = RenderProcessHost::FromID(render_process_id_);
  if (!process_host) {
    return;
  }

  auto* impl = static_cast<RenderProcessHostImpl*>(process_host);
  impl->GetGpuClient()->BindWebNNContextProvider(
      gpu_provider_.BindNewPipeAndPassReceiver(), is_incognito_);
  gpu_provider_.set_disconnect_handler(
      base::BindOnce(&WebNNContextProviderProxy::OnGpuProviderDisconnected,
                     weak_factory_.GetWeakPtr()));
}

void WebNNContextProviderProxy::EnsureTFLiteProviderConnection() {
  if (tflite_provider_.is_bound()) {
    return;
  }

  auto* process_host = RenderProcessHost::FromID(render_process_id_);
  if (!process_host) {
    return;
  }

  process_host->BindReceiver(mojo::GenericPendingReceiver(
      tflite_provider_.BindNewPipeAndPassReceiver()));
}

void WebNNContextProviderProxy::OnGpuProviderDisconnected() {
  gpu_provider_.reset();
}

}  // namespace content
