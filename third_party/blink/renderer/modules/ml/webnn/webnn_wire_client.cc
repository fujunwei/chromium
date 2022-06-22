// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/webnn_wire_client.h"

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
using ml::webnn::mojom::blink::WireServer;

}  // namespace

// static
scoped_refptr<WebnnWireClient> WebnnWireClient::Create(
    ExecutionContext* execution_context) {
  return base::MakeRefCounted<WebnnWireClient>(execution_context);
}

WebnnWireClient::WebnnWireClient(ExecutionContext* execution_context)
    : wire_server_(execution_context) {
  if (!wire_server_.is_bound()) {
    execution_context->GetBrowserInterfaceBroker().GetInterface(
        wire_server_.BindNewPipeAndPassReceiver(
            execution_context->GetTaskRunner(TaskType::kInternalDefault)));

    // wire_server_.set_disconnect_handler(WTF::Bind(
    //     &WebnnWireClient::OnWebnnServiceConnectionError,
    //     WrapWeakPersistent(this)));
  }
}

void WebnnWireClient::Trace(Visitor* visitor) const {
  visitor->Trace(wire_server_);
}

uint32_t WebnnWireClient::GetNewId() {
  if (free_ids_.empty()) {
    return current_id_++;
  }
  uint32_t id = free_ids_.back();
  free_ids_.pop_back();
  return id;
}

void WebnnWireClient::FreeId(uint32_t id) {
  free_ids_.push_back(id);
}

void WebnnWireClient::CreateWebnnContext(
    uint32_t context_id,
    ContextOptionsPtr options,
    WireServer::CreateContextCallback callback) {
  wire_server_->CreateContext(context_id, std::move(options),
                              std::move(callback));
}

void WebnnWireClient::OnWebnnServiceConnectionError() {
  wire_server_.reset();
}

}  // namespace blink
