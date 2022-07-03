// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/mojo_client.h"

#include "third_party/blink/renderer/bindings/core/v8/script_promise_resolver.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_context_options.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"

namespace blink {

// static
scoped_refptr<MojoClient> MojoClient::Create(
    ExecutionContext* execution_context) {
  return base::MakeRefCounted<MojoClient>(execution_context);
}

MojoClient::MojoClient(ExecutionContext* execution_context)
    : mojo_server_(execution_context) {
  execution_context->GetBrowserInterfaceBroker().GetInterface(
      mojo_server_.BindNewPipeAndPassReceiver(
          execution_context->GetTaskRunner(TaskType::kInternalDefault)));
}

void MojoClient::CreateMojoContext(ScriptPromiseResolver* resolver,
                                   ObjectId context_id,
                                   ContextOptionsPtr options,
                                   MojoServer::CreateContextCallback callback) {
  if (!mojo_server_.is_bound()) {
    resolver->Reject(MakeGarbageCollected<DOMException>(
        DOMExceptionCode::kNotSupportedError, "WebNN server unavailable."));
    return;
  }
  mojo_server_->CreateContext(std::move(options), context_id,
                              std::move(callback));
}

void MojoClient::Trace(Visitor* visitor) const {
  visitor->Trace(mojo_server_);
}

uint32_t MojoClient::GetNewId() {
  if (free_ids_.IsEmpty()) {
    current_id_++;
    return current_id_.ValueOrDie();
  }
  uint32_t id = free_ids_.back();
  free_ids_.pop_back();
  return id;
}

void MojoClient::FreeId(uint32_t id) {
  free_ids_.push_back(id);
}

}  // namespace blink
