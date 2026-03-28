// Copyright 2025 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "services/webnn/tflite/context_provider_tflite.h"

#include <utility>

#include "base/task/sequenced_task_runner.h"
#include "services/webnn/error.h"
#include "services/webnn/public/mojom/webnn_context_provider.mojom.h"
#include "services/webnn/public/mojom/webnn_error.mojom.h"
#include "services/webnn/tflite/context_impl_tflite.h"
#include "services/webnn/tflite/graph_builder_tflite.h"

namespace webnn::tflite {

ContextProviderTflite::ContextProviderTflite(
    WebNNContextImpl::CreateWeightsFileFn create_weights_file_fn)
    : create_weights_file_fn_(std::move(create_weights_file_fn)) {}
ContextProviderTflite::~ContextProviderTflite() = default;

void ContextProviderTflite::CreateWebNNContext(
    mojom::CreateContextOptionsPtr options,
    CreateWebNNContextCallback callback) {
  if (options->device != mojom::Device::kCpu) {
    std::move(callback).Run(ToError<mojom::CreateContextResult>(
        mojom::Error::Code::kNotSupportedError,
        "Only CPU device is supported for TFLite backend in this process."));
    return;
  }

  mojo::PendingRemote<mojom::WebNNContext> remote;
  auto receiver = remote.InitWithNewPipeAndPassReceiver();

  auto task_runner = base::SequencedTaskRunner::GetCurrentDefault();

  auto context_impl = ContextImplTflite::CreateForRenderer(
      std::move(receiver), std::move(options), task_runner,
      create_weights_file_fn_);

  ContextProperties context_properties = context_impl->properties();
  const blink::WebNNContextToken& context_handle = context_impl->handle();

  context_impls_.push_back(std::move(context_impl));

  auto success = mojom::CreateContextSuccess::New(
      std::move(remote), std::move(context_properties),
      std::move(context_handle),
      /*write_tensor_producer=*/mojo::ScopedDataPipeProducerHandle(),
      /*read_tensor_consumer=*/mojo::ScopedDataPipeConsumerHandle());
  std::move(callback).Run(
      mojom::CreateContextResult::NewSuccess(std::move(success)));
}

}  // namespace webnn::tflite
