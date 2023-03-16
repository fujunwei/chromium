// Copyright 2023 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef SERVICES_WEBNN_WEBNN_CONTEXT_PROVIDER_IMPL_WIN_H_
#define SERVICES_WEBNN_WEBNN_CONTEXT_PROVIDER_IMPL_WIN_H_

#include "mojo/public/cpp/bindings/pending_receiver.h"
#include "services/webnn/public/mojom/webnn_service.mojom.h"

namespace webnn {

class WebnnContextProviderImplWin : public mojom::WebnnContextProvider {
 public:
  ~WebnnContextProviderImplWin() override;
  static void Create(
      mojo::PendingReceiver<mojom::WebnnContextProvider> receiver);

  WebnnContextProviderImplWin(const WebnnContextProviderImplWin&) = delete;
  WebnnContextProviderImplWin& operator=(const WebnnContextProviderImplWin&) =
      delete;

 protected:
  WebnnContextProviderImplWin();

 private:
  // mojom::WebnnContextProvider
  void CreateWebnnContext(mojom::CreateContextOptionsPtr options,
                          CreateWebnnContextCallback callback) override;
};

}  // namespace webnn

#endif  // SERVICES_WEBNN_WEBNN_CONTEXT_PROVIDER_IMPL_WIN_H_
