// Copyright 2023 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "services/webnn/webnn_service.h"

#include "base/no_destructor.h"
#include "services/webnn/webnn_context_provider_impl.h"
#include "build/buildflag.h"

#if BUILDFLAG(IS_WIN)
#include "services/webnn/webnn_context_provider_impl_win.h"
#endif

namespace webnn {

WebnnService::WebnnService(mojo::PendingReceiver<mojom::WebnnService> receiver)
    : receiver_(this, std::move(receiver)) {}

WebnnService::~WebnnService() = default;

void WebnnService::BindWebnnContextProvider(
    mojo::PendingReceiver<mojom::WebnnContextProvider> receiver) {
#if BUILDFLAG(IS_WIN)
  WebnnContextProviderImplWin::Create(std::move(receiver));
#else
  WebnnContextProviderImpl::Create(std::move(receiver));
#endif
}

}  // namespace webnn
