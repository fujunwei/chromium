// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CONTENT_BROWSER_WEBNN_WEBNN_CONTEXT_PROVIDER_PROXY_H_
#define CONTENT_BROWSER_WEBNN_WEBNN_CONTEXT_PROVIDER_PROXY_H_

#include "base/memory/weak_ptr.h"
#include "content/common/content_export.h"
#include "mojo/public/cpp/bindings/remote.h"
#include "services/webnn/public/mojom/webnn_context_provider.mojom.h"

namespace content {

// A browser-process proxy for WebNNContextProvider that first tries the
// GPU process backend (e.g., ORT/DML) and falls back to the TFLite backend
// in the render process if the GPU backend fails to create a context.
//
// This is needed because the ORT backend runs in the GPU process while the
// TFLite backend has been moved to the render process. The proxy transparently
// handles the cross-process fallback so that the renderer's ML::createContext
// does not need to be aware of the backend routing.
class CONTENT_EXPORT WebNNContextProviderProxy
    : public webnn::mojom::WebNNContextProvider {
 public:
  WebNNContextProviderProxy(int render_process_id, bool is_incognito);
  ~WebNNContextProviderProxy() override;

  WebNNContextProviderProxy(const WebNNContextProviderProxy&) = delete;
  WebNNContextProviderProxy& operator=(const WebNNContextProviderProxy&) =
      delete;

  // webnn::mojom::WebNNContextProvider:
  void CreateWebNNContext(webnn::mojom::CreateContextOptionsPtr options,
                          CreateWebNNContextCallback callback) override;

 private:
  void OnGpuContextCreated(webnn::mojom::CreateContextOptionsPtr options,
                           CreateWebNNContextCallback callback,
                           webnn::mojom::CreateContextResultPtr result);

  void FallbackToTFLite(webnn::mojom::CreateContextOptionsPtr options,
                        CreateWebNNContextCallback callback);

  void EnsureGpuProviderConnection();
  void EnsureTFLiteProviderConnection();

  void OnGpuProviderDisconnected();

  const int render_process_id_;
  const bool is_incognito_;

  mojo::Remote<webnn::mojom::WebNNContextProvider> gpu_provider_;
  mojo::Remote<webnn::mojom::WebNNContextProvider> tflite_provider_;

  base::WeakPtrFactory<WebNNContextProviderProxy> weak_factory_{this};
};

}  // namespace content

#endif  // CONTENT_BROWSER_WEBNN_WEBNN_CONTEXT_PROVIDER_PROXY_H_
