// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CONTENT_BROWSER_ML_WEBNN_NEURAL_NETWORK_IMPL_DML_H_
#define CONTENT_BROWSER_ML_WEBNN_NEURAL_NETWORK_IMPL_DML_H_

#include "components/ml/webnn/mojom/webnn_service.mojom.h"
#include "mojo/public/cpp/bindings/pending_receiver.h"

namespace content {

namespace webnn {

class NeuralNetwoekDMLImpl : public ml::webnn::mojom::NeuralNetwork {
 public:
  ~NeuralNetwoekDMLImpl() override;
  static void Create(
      mojo::PendingReceiver<ml::webnn::mojom::NeuralNetwork> receiver);

  NeuralNetwoekDMLImpl(const NeuralNetwoekDMLImpl&) = delete;
  NeuralNetwoekDMLImpl& operator=(const NeuralNetwoekDMLImpl&) = delete;

 protected:
  NeuralNetwoekDMLImpl();

 private:
  // ml::webnn::mojom::NeuralNetwork
  void CreateContext(uint32_t id,
                     ml::webnn::mojom::ContextOptionsPtr options,
                     CreateContextCallback callback) override;
};

}  // namespace webnn
}  // namespace content

#endif  // CONTENT_BROWSER_ML_WEBNN_NEURAL_NETWORK_IMPL_DML_H_
