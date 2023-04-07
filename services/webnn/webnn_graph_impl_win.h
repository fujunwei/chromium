// Copyright 2023 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef SERVICES_WEBNN_WEBNN_GRAPH_IMPL_WIN_H_
#define SERVICES_WEBNN_WEBNN_GRAPH_IMPL_WIN_H_

#include "mojo/public/cpp/bindings/pending_receiver.h"
#include "services/webnn/public/mojom/webnn_graph.mojom.h"

namespace webnn {

class WebNNGraphImplWin : public mojom::WebNNGraph {
 public:
  ~WebNNGraphImplWin() override;
  static void Create(mojo::PendingReceiver<mojom::WebNNGraph> receiver);

  WebNNGraphImplWin(const WebNNGraphImplWin&) = delete;
  WebNNGraphImplWin& operator=(const WebNNGraphImplWin&) = delete;

 protected:
  WebNNGraphImplWin();

 private:
  // mojom::WebNNGraph
};

}  // namespace webnn

#endif  // SERVICES_WEBNN_WEBNN_GRAPH_IMPL_WIN_H_
