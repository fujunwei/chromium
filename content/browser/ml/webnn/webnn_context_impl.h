// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CONTENT_BROWSER_ML_WEBNN_WEBNN_CONTEXT_IMPL_H_
#define CONTENT_BROWSER_ML_WEBNN_WEBNN_CONTEXT_IMPL_H_

#include "components/ml/mojom/webnn_service.mojom.h"
#include "mojo/public/cpp/bindings/pending_receiver.h"

namespace content::webnn {

class WebnnContextImpl : public ml::webnn::mojom::WebnnContext {
 public:
  ~WebnnContextImpl() override;
  static void Create(
      mojo::PendingReceiver<ml::webnn::mojom::WebnnContext> receiver);

  WebnnContextImpl(const WebnnContextImpl&) = delete;
  WebnnContextImpl& operator=(const WebnnContextImpl&) = delete;

 protected:
  WebnnContextImpl();

 private:
  // ml::webnn::mojom::WebnnContext
  void CreateGraph(ml::webnn::mojom::CreateGraphOptionsPtr options,
                   CreateGraphCallback callback) override;
};

}  // namespace content::webnn

#endif  // CONTENT_BROWSER_ML_WEBNN_WEBNN_CONTEXT_IMPL_H_
