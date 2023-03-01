// Copyright 2023 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "services/webnn/webnn_service.h"

#include "base/no_destructor.h"
#include "services/webnn/webnn_context_impl.h"

namespace webnn {

WebnnService::WebnnService(mojo::PendingReceiver<mojom::WebnnService> receiver)
    : receiver_(this, std::move(receiver)) {}

WebnnService::~WebnnService() = default;

void WebnnService::BindWebnnContext(
    mojo::PendingReceiver<mojom::WebnnContext> receiver) {
  WebnnContextImpl::Create(std::move(receiver));
}

}  // namespace webnn
