// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/webnn_client.h"

#include "third_party/blink/renderer/bindings/core/v8/script_promise_resolver.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_context_options.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"
#include "third_party/blink/renderer/core/inspector/console_message.h"
#include "third_party/blink/renderer/modules/ml/webnn/webnn_context.h"
#include "third_party/blink/renderer/modules/ml/webnn/webnn_object.h"
#include "third_party/blink/renderer/platform/bindings/exception_state.h"

namespace blink {

namespace {

using ml::webnn::mojom::blink::ContextOptionsPtr;
using ml::webnn::mojom::blink::NeuralNetwork;

}  // namespace

// static
scoped_refptr<WebnnClient> WebnnClient::Create(
    ExecutionContext* execution_context) {
  return base::MakeRefCounted<WebnnClient>(execution_context);
}

WebnnClient::WebnnClient(ExecutionContext* execution_context)
    : webnn_service_(execution_context) {
  if (!webnn_service_.is_bound()) {
    execution_context->GetBrowserInterfaceBroker().GetInterface(
        webnn_service_.BindNewPipeAndPassReceiver(
            execution_context->GetTaskRunner(TaskType::kInternalDefault)));

    // webnn_service_.set_disconnect_handler(WTF::Bind(
    //     &WebnnClient::OnWebnnServiceConnectionError,
    //     WrapWeakPersistent(this)));
  }
}

void WebnnClient::Trace(Visitor* visitor) const {
  visitor->Trace(webnn_service_);
}

uint32_t WebnnClient::GetNewId() {
  if (free_ids_.empty()) {
    return current_id_++;
  }
  uint32_t id = free_ids_.back();
  free_ids_.pop_back();
  return id;
}

void WebnnClient::FreeId(uint32_t id) {
  free_ids_.push_back(id);
}

void WebnnClient::CreateWebnnContext(
    uint32_t context_id,
    ContextOptionsPtr options,
    NeuralNetwork::CreateContextCallback callback) {
  webnn_service_->CreateContext(context_id, std::move(options),
                                std::move(callback));
}

void WebnnClient::OnWebnnServiceConnectionError() {
  webnn_service_.reset();
}

}  // namespace blink
