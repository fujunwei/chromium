// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_WEBNN_CONTEXT_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_WEBNN_CONTEXT_H_

#include "base/memory/scoped_refptr.h"
#include "components/ml/webnn/mojom/context.mojom-blink.h"
#include "mojo/public/cpp/bindings/pending_remote.h"
#include "third_party/blink/renderer/modules/ml/ml_context.h"
#include "third_party/blink/renderer/modules/ml/webnn/webnn_object.h"
#include "third_party/blink/renderer/platform/bindings/script_wrappable.h"
#include "third_party/blink/renderer/platform/heap/member.h"
#include "third_party/blink/renderer/platform/heap/visitor.h"
#include "third_party/blink/renderer/platform/mojo/heap_mojo_remote.h"

namespace blink {

class ExecutionContext;
class ML;
class ScriptState;
class ScriptPromiseResolver;
class WebnnWireClient;

class WebnnContext : public MLContext, public WebnnObjectBase {
 public:
  WebnnContext(ScriptState* script_state,
               ScriptPromiseResolver* resolver,
               const V8MLPowerPreference power_preference,
               ML* ml,
               scoped_refptr<WebnnWireClient> webnn_client);

  WebnnContext(const WebnnContext&) = delete;
  WebnnContext& operator=(const WebnnContext&) = delete;

  ~WebnnContext() override;

  void CreateGraph(uint32_t graph_id,
                   uint32_t context_id,
                   ml::webnn::mojom::blink::Context::CreateGraphCallback);

  void Trace(Visitor* visitor) const override;

 private:
  void OnContextCreated(
      ScriptState* script_state,
      ScriptPromiseResolver* resolver,
      mojo::PendingRemote<ml::webnn::mojom::blink::Context> pending_remote);

  HeapMojoRemote<ml::webnn::mojom::blink::Context> remote_context_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_WEBNN_CONTEXT_H_
