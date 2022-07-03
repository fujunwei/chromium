// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/browser/ml/webnn/webnn_service_impl.h"

#include "base/logging.h"
#include "base/no_destructor.h"
#include "content/browser/ml/webnn/mojo_server_impl.h"

namespace content::webnn {

// static
void WebnnServiceImpl::Create(
    mojo::PendingReceiver<ml::webnn::mojom::WebnnService> receiver) {
  static base::NoDestructor<WebnnServiceImpl> service{std::move(receiver)};
}

WebnnServiceImpl::WebnnServiceImpl(
    mojo::PendingReceiver<ml::webnn::mojom::WebnnService> receiver)
    : receiver_(this, std::move(receiver)) {}

WebnnServiceImpl::~WebnnServiceImpl() = default;

void WebnnServiceImpl::BindMojoServer(
    mojo::PendingReceiver<ml::webnn::mojom::MojoServer> receiver) {
  MojoServerImpl::Create(std::move(receiver));
}

}  // namespace content::webnn
