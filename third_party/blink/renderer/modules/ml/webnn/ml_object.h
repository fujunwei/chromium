// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_OBJECT_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_OBJECT_H_

#include "base/memory/scoped_refptr.h"
#include "third_party/blink/renderer/modules/ml/webnn/mojo_client.h"
#include "third_party/blink/renderer/platform/bindings/script_wrappable.h"
#include "third_party/blink/renderer/platform/heap/member.h"
#include "third_party/blink/renderer/platform/heap/visitor.h"

namespace blink {

// The class allows objects to generate the Object Id which will talk with
// server side.
// MojoClient is used to manager all objects id,
// 1, Get a id for new object
// 2, Free the id when releasing the object
class ObjectHandle {
 public:
  explicit ObjectHandle(scoped_refptr<MojoClient> client);

  ~ObjectHandle();

  ObjectId GetObjectId() const;

 private:
  scoped_refptr<MojoClient> mojo_client_;
  ObjectId id_;
};

class MLContext;

class MLObject : public ScriptWrappable, public ObjectHandle {
 public:
  explicit MLObject(MLContext* context);

  ~MLObject() override;

  MLContext* GetContext() const;

  void Trace(Visitor* visitor) const override;

 private:
  Member<MLContext> context_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_OBJECT_H_
