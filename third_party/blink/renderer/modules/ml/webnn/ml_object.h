// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_OBJECT_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_OBJECT_H_

#include "mojo/public/cpp/system/handle.h"
#include "mojo/public/cpp/system/message_pipe.h"
#include "third_party/blink/renderer/platform/bindings/script_wrappable.h"
#include "third_party/blink/renderer/platform/heap/member.h"
#include "third_party/blink/renderer/platform/heap/visitor.h"

namespace blink {

class MLContext;

class MLObject : public ScriptWrappable {
 public:
  explicit MLObject(MLContext* context);

  ~MLObject() override;

  MLContext* GetContext() const;

  mojo::Handle GetMojoHandle() const;

  void Trace(Visitor* visitor) const override;

 private:
  Member<MLContext> context_;
  // The message pipe handle for receiver.
  mojo::ScopedMessagePipeHandle receiver_handle_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_OBJECT_H_
