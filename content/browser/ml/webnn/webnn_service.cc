// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/browser/ml/webnn/webnn_service.h"

#include "content/browser/ml/webnn/buildflags.h"

#if BUILDFLAG(ENABLE_MOJO_WEBNN_IN_UTILITY_PROCESS)
#if BUILDFLAG(IS_WIN)
#include "content/browser/ml/webnn/webnn_context_impl_win.h"
#endif
#else
#include "content/browser/ml/webnn/webnn_context_impl.h"
#endif

namespace content::webnn {

WebnnService::WebnnService(
    mojo::PendingReceiver<ml::webnn::mojom::WebnnService> receiver)
    : receiver_(this, std::move(receiver)) {}

WebnnService::~WebnnService() = default;

void WebnnService::BindWebnnContext(
    mojo::PendingReceiver<ml::webnn::mojom::WebnnContext> receiver) {
#if BUILDFLAG(ENABLE_MOJO_WEBNN_IN_UTILITY_PROCESS)
#if BUILDFLAG(IS_WIN)
  WebnnContextImplWin::Create(std::move(receiver));
#endif
#else
  WebnnContextImpl::Create(std::move(receiver));
#endif
}

}  // namespace content::webnn
