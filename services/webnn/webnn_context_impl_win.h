// Copyright 2023 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef SERVICES_WEBNN_WEBNN_CONTEXT_IMPL_WIN_H_
#define SERVICES_WEBNN_WEBNN_CONTEXT_IMPL_WIN_H_

#include "mojo/public/cpp/bindings/pending_receiver.h"
#include "services/webnn/public/mojom/webnn_service.mojom.h"

namespace webnn {

class WebNNContextImplWin : public mojom::WebNNContext {
 public:
  ~WebNNContextImplWin() override;
  static void Create(mojo::PendingReceiver<mojom::WebNNContext> receiver);

  WebNNContextImplWin(const WebNNContextImplWin&) = delete;
  WebNNContextImplWin& operator=(const WebNNContextImplWin&) = delete;

 protected:
  WebNNContextImplWin();

 private:
  // mojom::WebNNContext
  void CreateGraph(CreateGraphCallback callback) override;
};

}  // namespace webnn

#endif  // SERVICES_WEBNN_WEBNN_CONTEXT_IMPL_WIN_H_
