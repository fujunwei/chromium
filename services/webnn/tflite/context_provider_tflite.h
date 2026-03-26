// Copyright 2025 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef SERVICES_WEBNN_TFLITE_CONTEXT_PROVIDER_TFLITE_H_
#define SERVICES_WEBNN_TFLITE_CONTEXT_PROVIDER_TFLITE_H_

#include <memory>
#include <vector>

#include "base/component_export.h"
#include "services/webnn/public/mojom/webnn_context_provider.mojom.h"
#include "services/webnn/tflite/context_impl_tflite.h"

namespace webnn::tflite {

// A lightweight WebNNContextProvider implementation for TFLite that runs
// without GPU dependencies (e.g., in the renderer process).
class COMPONENT_EXPORT(WEBNN_SERVICE) ContextProviderTflite
    : public mojom::WebNNContextProvider {
 public:
  ContextProviderTflite();
  ~ContextProviderTflite() override;

  ContextProviderTflite(const ContextProviderTflite&) = delete;
  ContextProviderTflite& operator=(const ContextProviderTflite&) = delete;

  // mojom::WebNNContextProvider:
  void CreateWebNNContext(mojom::CreateContextOptionsPtr options,
                          CreateWebNNContextCallback callback) override;

 private:
  // Contexts created by this provider. Cleaned up when the provider is
  // destroyed (when the mojo pipe closes).
  std::vector<WebNNContextImpl::WebNNContextImplPtr> context_impls_;
};

}  // namespace webnn::tflite

#endif  // SERVICES_WEBNN_TFLITE_CONTEXT_PROVIDER_TFLITE_H_
