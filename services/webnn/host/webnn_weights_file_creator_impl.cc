// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "services/webnn/host/webnn_weights_file_creator_impl.h"

#include "base/files/file.h"
#include "mojo/public/cpp/bindings/self_owned_receiver.h"
#include "services/webnn/host/weights_file_provider.h"

namespace webnn {

WebNNWeightsFileCreatorImpl::WebNNWeightsFileCreatorImpl(bool is_incognito)
    : is_incognito_(is_incognito) {}

WebNNWeightsFileCreatorImpl::~WebNNWeightsFileCreatorImpl() = default;

// static
void WebNNWeightsFileCreatorImpl::Create(
    bool is_incognito,
    mojo::PendingReceiver<mojom::WebNNWeightsFileCreator> receiver) {
  mojo::MakeSelfOwnedReceiver(
      std::make_unique<WebNNWeightsFileCreatorImpl>(is_incognito),
      std::move(receiver));
}

void WebNNWeightsFileCreatorImpl::CreateWeightsFile(
    CreateWeightsFileCallback callback) {
  if (is_incognito_) {
    // In incognito mode, don't create files on disk. Return a null file
    // handle so the TFLite backend stores weights inline.
    std::move(callback).Run(base::File());
    return;
  }

  // Delegate to the free function in weights_file_provider.h which creates
  // a temporary file on a background thread.
  ::webnn::CreateWeightsFile(std::move(callback));
}

}  // namespace webnn
