// Copyright 2026 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef SERVICES_WEBNN_HOST_WEBNN_WEIGHTS_FILE_CREATOR_IMPL_H_
#define SERVICES_WEBNN_HOST_WEBNN_WEIGHTS_FILE_CREATOR_IMPL_H_

#include "mojo/public/cpp/bindings/pending_receiver.h"
#include "services/webnn/public/mojom/webnn_weights_file_creator.mojom.h"

namespace webnn {

// Browser-side implementation of WebNNWeightsFileCreator.
// Creates temporary files for WebNN model weights on behalf of sandboxed
// renderer processes running the TFLite backend.
//
// In incognito mode, returns a null file handle so the renderer's TFLite
// backend stores weights inline in the Flatbuffer model.
class WebNNWeightsFileCreatorImpl : public mojom::WebNNWeightsFileCreator {
 public:
  explicit WebNNWeightsFileCreatorImpl(bool is_incognito);
  ~WebNNWeightsFileCreatorImpl() override;

  WebNNWeightsFileCreatorImpl(const WebNNWeightsFileCreatorImpl&) = delete;
  WebNNWeightsFileCreatorImpl& operator=(const WebNNWeightsFileCreatorImpl&) =
      delete;

  // Binds a receiver, creating a self-owned instance.
  static void Create(
      bool is_incognito,
      mojo::PendingReceiver<mojom::WebNNWeightsFileCreator> receiver);

  // mojom::WebNNWeightsFileCreator:
  void CreateWeightsFile(CreateWeightsFileCallback callback) override;

 private:
  const bool is_incognito_;
};

}  // namespace webnn

#endif  // SERVICES_WEBNN_HOST_WEBNN_WEIGHTS_FILE_CREATOR_IMPL_H_
