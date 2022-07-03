// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/ml_object.h"

#include "third_party/blink/renderer/modules/ml/ml.h"
#include "third_party/blink/renderer/modules/ml/ml_context.h"

namespace blink {

ObjectHandle::ObjectHandle(scoped_refptr<MojoClient> client)
    : mojo_client_(client), id_(mojo_client_->GetNewId()) {}

ObjectHandle::~ObjectHandle() {
  mojo_client_->FreeId(id_);
}

ObjectId ObjectHandle::GetObjectId() const {
  return id_;
}

MLObject::MLObject(MLContext* context)
    : ObjectHandle(context->GetML()->GetMojoClient()), context_(context) {}

MLObject::~MLObject() = default;

MLContext* MLObject::GetContext() const {
  return context_.Get();
}

void MLObject::Trace(Visitor* visitor) const {
  visitor->Trace(context_);
  ScriptWrappable::Trace(visitor);
}

}  // namespace blink
