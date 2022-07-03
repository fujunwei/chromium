// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CONTENT_BROWSER_ML_WEBNN_WEBNN_SERVICE_FACTORY_H_
#define CONTENT_BROWSER_ML_WEBNN_WEBNN_SERVICE_FACTORY_H_

#include "components/ml/mojom/webnn_service.mojom.h"
#include "mojo/public/cpp/bindings/pending_receiver.h"

namespace content::webnn {

// Create WebNN service on different platforms.
void CreateWebNNService(mojo::PendingReceiver<ml::webnn::mojom::WebnnService>);

}  // namespace content::webnn

#endif  // CONTENT_BROWSER_ML_WEBNN_WEBNN_SERVICE_FACTORY_H_
