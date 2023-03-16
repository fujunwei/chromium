// Copyright 2023 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef SERVICES_WEBNN_WEBNN_CONTEXT_PROVIDER_IMPL_H_
#define SERVICES_WEBNN_WEBNN_CONTEXT_PROVIDER_IMPL_H_

#include "mojo/public/cpp/bindings/pending_receiver.h"
#include "services/webnn/public/mojom/webnn_service.mojom.h"

namespace webnn {

class WebnnContextProviderImpl : public mojom::WebnnContextProvider {
 public:
  ~WebnnContextProviderImpl() override;
  static void Create(
      mojo::PendingReceiver<mojom::WebnnContextProvider> receiver);

  WebnnContextProviderImpl(const WebnnContextProviderImpl&) = delete;
  WebnnContextProviderImpl& operator=(const WebnnContextProviderImpl&) = delete;

 protected:
  WebnnContextProviderImpl();

 private:
  // mojom::WebnnContextProvider
  void CreateWebnnContext(mojom::CreateContextOptionsPtr options,
                          CreateWebnnContextCallback callback) override;
};

}  // namespace webnn

#endif  // SERVICES_WEBNN_WEBNN_CONTEXT_PROVIDER_IMPL_H_
