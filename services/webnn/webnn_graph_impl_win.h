// Copyright 2023 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef SERVICES_WEBNN_WEBNN_GRAPH_IMPL_WIN_H_
#define SERVICES_WEBNN_WEBNN_GRAPH_IMPL_WIN_H_

#include "mojo/public/cpp/bindings/pending_receiver.h"
#include "services/webnn/public/mojom/webnn_graph.mojom.h"

namespace webnn {

class WebnnGraphImplWin : public mojom::WebnnGraph {
 public:
  ~WebnnGraphImplWin() override;
  static void Create(mojo::PendingReceiver<mojom::WebnnGraph> receiver);

  WebnnGraphImplWin(const WebnnGraphImplWin&) = delete;
  WebnnGraphImplWin& operator=(const WebnnGraphImplWin&) = delete;

 protected:
  WebnnGraphImplWin();

 private:
  // mojom::WebnnGraph
};

}  // namespace webnn

#endif  // SERVICES_WEBNN_WEBNN_GRAPH_IMPL_WIN_H_
