// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CONTENT_BROWSER_ML_WEBNN_NEURAL_NETWORK_IMPL_H_
#define CONTENT_BROWSER_ML_WEBNN_NEURAL_NETWORK_IMPL_H_

#include "components/ml/webnn/mojom/webnn_service.mojom.h"
#include "mojo/public/cpp/bindings/pending_receiver.h"
#include "mojo/public/cpp/system/handle.h"

namespace content {

namespace webnn {

class WireServerImpl : public ml::webnn::mojom::WireServer {
 public:
  ~WireServerImpl() override;
  static void Create(
      mojo::PendingReceiver<ml::webnn::mojom::WireServer> receiver);

  WireServerImpl(const WireServerImpl&) = delete;
  WireServerImpl& operator=(const WireServerImpl&) = delete;

 protected:
  WireServerImpl();

 private:
  // ml::webnn::mojom::WireServer
  void CreateContext(uint32_t self_id,
                     ml::webnn::mojom::ContextOptionsPtr options,
                     CreateContextCallback callback) override;
};

}  // namespace webnn
}  // namespace content

#endif  // CONTENT_BROWSER_ML_WEBNN_NEURAL_NETWORK_IMPL_H_
