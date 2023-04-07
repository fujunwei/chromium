// Copyright 2023 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef SERVICES_WEBNN_WEBNN_CONTEXT_PROVIDER_IMPL_WIN_H_
#define SERVICES_WEBNN_WEBNN_CONTEXT_PROVIDER_IMPL_WIN_H_

#include "mojo/public/cpp/bindings/pending_receiver.h"
#include "services/webnn/public/mojom/webnn_service.mojom.h"

namespace webnn {

class WebNNContextProviderImplWin : public mojom::WebNNContextProvider {
 public:
  ~WebNNContextProviderImplWin() override;
  static void Create(
      mojo::PendingReceiver<mojom::WebNNContextProvider> receiver);

  WebNNContextProviderImplWin(const WebNNContextProviderImplWin&) = delete;
  WebNNContextProviderImplWin& operator=(const WebNNContextProviderImplWin&) =
      delete;

 protected:
  WebNNContextProviderImplWin();

 private:
  // mojom::WebNNContextProvider
  void CreateWebNNContext(mojom::CreateContextOptionsPtr options,
                          CreateWebNNContextCallback callback) override;
};

}  // namespace webnn

#endif  // SERVICES_WEBNN_WEBNN_CONTEXT_PROVIDER_IMPL_WIN_H_
