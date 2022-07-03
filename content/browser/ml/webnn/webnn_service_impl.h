// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CONTENT_BROWSER_ML_WEBNN_WEBNN_SERVICE_IMPL_H_
#define CONTENT_BROWSER_ML_WEBNN_WEBNN_SERVICE_IMPL_H_

#include "components/ml/mojom/webnn_service.mojom.h"
#include "mojo/public/cpp/bindings/pending_receiver.h"
#include "mojo/public/cpp/bindings/receiver.h"

namespace content::webnn {

class WebnnServiceImpl : public ml::webnn::mojom::WebnnService {
 public:
  static void Create(
      mojo::PendingReceiver<ml::webnn::mojom::WebnnService> receiver);

  explicit WebnnServiceImpl(
      mojo::PendingReceiver<ml::webnn::mojom::WebnnService> receiver);
  ~WebnnServiceImpl() override;

  WebnnServiceImpl(const WebnnServiceImpl&) = delete;
  WebnnServiceImpl& operator=(const WebnnServiceImpl&) = delete;

  void BindMojoServer(
      mojo::PendingReceiver<ml::webnn::mojom::MojoServer> receiver) override;

 private:
  mojo::Receiver<ml::webnn::mojom::WebnnService> receiver_;
};

}  // namespace content::webnn

#endif  // CONTENT_BROWSER_ML_WEBNN_WEBNN_SERVICE_IMPL_H_
