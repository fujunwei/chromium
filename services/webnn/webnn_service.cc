// Copyright 2023 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "services/webnn/webnn_service.h"

#include "base/no_destructor.h"
#include "build/buildflag.h"
#include "services/webnn/webnn_context_provider_impl.h"

#if BUILDFLAG(IS_WIN)
#include "services/webnn/webnn_context_provider_impl_win.h"
#endif

namespace webnn {

WebNNService::WebNNService(mojo::PendingReceiver<mojom::WebNNService> receiver)
    : receiver_(this, std::move(receiver)) {}

WebNNService::~WebNNService() = default;

void WebNNService::BindWebNNContextProvider(
    mojo::PendingReceiver<mojom::WebNNContextProvider> receiver) {
#if BUILDFLAG(IS_WIN)
  WebNNContextProviderImplWin::Create(std::move(receiver));
#else
  WebNNContextProviderImpl::Create(std::move(receiver));
#endif
}

}  // namespace webnn
