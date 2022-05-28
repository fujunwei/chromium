// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_WEBNN_WEBNN_OBJECT_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_WEBNN_WEBNN_OBJECT_H_

#include "base/memory/scoped_refptr.h"
#include "third_party/blink/renderer/modules/ml/webnn/webnn_client.h"
#include "third_party/blink/renderer/platform/bindings/script_wrappable.h"
#include "third_party/blink/renderer/platform/heap/member.h"
#include "third_party/blink/renderer/platform/heap/visitor.h"
#include "third_party/blink/renderer/platform/wtf/ref_counted.h"

#include <vector>

namespace blink {

class WebnnObjectBase {
 public:
  explicit WebnnObjectBase(scoped_refptr<WebnnClient> client);

  ~WebnnObjectBase();

  const scoped_refptr<WebnnClient>& GetWebnnClient() const;

  uint32_t GetObjectId() const;

 private:
  scoped_refptr<WebnnClient> webnn_client_;
  uint32_t id_;
};

class MLContext;

class WebnnObject : public ScriptWrappable, public WebnnObjectBase {
 public:
  explicit WebnnObject(MLContext* context);

  ~WebnnObject() override;

  void Trace(Visitor* visitor) const override;

 protected:
  Member<MLContext> context_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_WEBNN_WEBNN_OBJECT_H_
