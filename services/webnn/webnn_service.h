// Copyright 2023 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef SERVICES_WEBNN_WEBNN_SERVICE_H_
#define SERVICES_WEBNN_WEBNN_SERVICE_H_

#include "mojo/public/cpp/bindings/pending_receiver.h"
#include "mojo/public/cpp/bindings/receiver.h"
#include "services/webnn/public/mojom/webnn_service.mojom.h"

namespace webnn {

class WebnnService : public mojom::WebnnService {
 public:
  explicit WebnnService(mojo::PendingReceiver<mojom::WebnnService> receiver);
  ~WebnnService() override;

  WebnnService(const WebnnService&) = delete;
  WebnnService& operator=(const WebnnService&) = delete;

  void BindWebnnContextProvider(
      mojo::PendingReceiver<mojom::WebnnContextProvider> receiver) override;

 private:
  mojo::Receiver<mojom::WebnnService> receiver_;
};

}  // namespace webnn

#endif  // SERVICES_WEBNN_WEBNN_SERVICE_H_
