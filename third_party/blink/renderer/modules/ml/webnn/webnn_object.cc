// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/webnn_object.h"

#include "third_party/blink/renderer/modules/ml/ml_context.h"
#include "third_party/blink/renderer/modules/ml/webnn/webnn_context.h"

namespace blink {

WebnnObjectBase::WebnnObjectBase(scoped_refptr<WebnnClient> client)
    : webnn_client_(std::move(client)), id_(webnn_client_->GetNewId()) {}

WebnnObjectBase::~WebnnObjectBase() {
  webnn_client_->FreeId(id_);
}

const scoped_refptr<WebnnClient>& WebnnObjectBase::GetWebnnClient() const {
  return webnn_client_;
}

uint32_t WebnnObjectBase::GetObjectId() const {
  return id_;
}

WebnnObject::WebnnObject(MLContext* context)
    : WebnnObjectBase(static_cast<WebnnContext*>(context)->GetWebnnClient()),
      context_(context) {}

WebnnObject::~WebnnObject() = default;

void WebnnObject::Trace(Visitor* visitor) const {
  visitor->Trace(context_);
  ScriptWrappable::Trace(visitor);
}

}  // namespace blink
