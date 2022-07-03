// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/browser/ml/webnn/webnn_service_factory.h"

#if BUILDFLAG(IS_WIN)
#include "content/browser/ml/webnn/dml/webnn_service_dml_impl.h"
#else
#include "content/browser/ml/webnn/webnn_service_impl.h"
#endif

namespace content::webnn {

void CreateWebNNService(
    mojo::PendingReceiver<ml::webnn::mojom::WebnnService> pending_receiver) {
#if BUILDFLAG(IS_WIN)
  WebnnServiceDMLImpl::Create(std::move(pending_receiver));
#else
  WebnnServiceImpl::Create(std::move(pending_receiver));
#endif
}

}  // namespace content::webnn
