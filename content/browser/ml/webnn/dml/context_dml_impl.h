// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CONTENT_BROWSER_ML_WEBNN_DML_CONTEXT_DML_IMPL_H_
#define CONTENT_BROWSER_ML_WEBNN_DML_CONTEXT_DML_IMPL_H_

#include "components/ml/webnn/mojom/context.mojom.h"
#include "mojo/public/cpp/bindings/pending_receiver.h"
#include "mojo/public/cpp/system/handle.h"

namespace content {

namespace webnn {

class ContextDMLImpl : public ml::webnn::mojom::Context {
 public:
  ~ContextDMLImpl() override;
  static void Create(mojo::PendingReceiver<ml::webnn::mojom::Context> receiver);

  ContextDMLImpl(const ContextDMLImpl&) = delete;
  ContextDMLImpl& operator=(const ContextDMLImpl&) = delete;

 protected:
  ContextDMLImpl();

 private:
  // ml::webnn::mojom::context
  void CreateGraph(uint32_t self_id,
                   uint32_t context_id,
                   CreateGraphCallback callback) override;
};

}  // namespace webnn
}  // namespace content

#endif  // CONTENT_BROWSER_ML_WEBNN_DML_CONTEXT_DML_IMPL_H_
