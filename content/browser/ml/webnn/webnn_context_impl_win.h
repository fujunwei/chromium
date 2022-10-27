// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CONTENT_BROWSER_ML_WEBNN_WEBNN_CONTEXT_IMPL_WIN_H_
#define CONTENT_BROWSER_ML_WEBNN_WEBNN_CONTEXT_IMPL_WIN_H_

#include <map>

#include "base/memory/ref_counted.h"
#include "components/ml/mojom/webnn_service.mojom.h"
#include "content/browser/ml/webnn/dml/adapter_dml.h"
#include "mojo/public/cpp/bindings/pending_receiver.h"

namespace content::webnn {

namespace {

using ml::model_loader::mojom::PowerPreference;

}

class WebnnContextImplWin : public ml::webnn::mojom::WebnnContext {
 public:
  ~WebnnContextImplWin() override;
  static void Create(
      mojo::PendingReceiver<ml::webnn::mojom::WebnnContext> receiver);

  WebnnContextImplWin(const WebnnContextImplWin&) = delete;
  WebnnContextImplWin& operator=(const WebnnContextImplWin&) = delete;

 protected:
  WebnnContextImplWin();

 private:
  // ml::webnn::mojom::WebnnContext
  void CreateGraph(ml::webnn::mojom::CreateGraphOptionsPtr options,
                   CreateGraphCallback callback) override;

  // Enumerate all adapters at once and share with different context.
  static std::map<AdapterType, scoped_refptr<AdapterDML>> adapter_map_;
  // The DMLExecutionContext is shared and reference-counted by all instances
  // of WebnnGraph.
//   scoped_refptr<ExecutionContext> execution_context_;
};

}  // namespace content::webnn

#endif  // CONTENT_BROWSER_ML_WEBNN_WEBNN_CONTEXT_IMPL_WIN_H_
