// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/ml_object.h"

#include "components/ml/mojom/webnn_graph.mojom-blink.h"
#include "third_party/blink/renderer/modules/ml/ml.h"
#include "third_party/blink/renderer/modules/ml/ml_context.h"

namespace blink {

MLObject::MLObject(MLContext* context) : context_(context) {
  // The mojo remote is used to generate a message pipe handle to identify the
  // operand.
  mojo::Remote<ml::webnn::mojom::blink::MLObject> mojo_object;
  // The PendingReceiver is no longer in a valid state and can no longer
  // be used to bind a Receiver after calling PassPipe().
  receiver_handle_ = mojo_object.BindNewPipeAndPassReceiver().PassPipe();
}

MLObject::~MLObject() = default;

MLContext* MLObject::GetContext() const {
  return context_.Get();
}

mojo::Handle MLObject::GetMojoHandle() const {
  return receiver_handle_.get();
}

void MLObject::Trace(Visitor* visitor) const {
  visitor->Trace(context_);
  ScriptWrappable::Trace(visitor);
}

}  // namespace blink
