// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_MOJO_CONTEXT_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_MOJO_CONTEXT_H_

#include "components/ml/mojom/webnn_context.mojom-blink.h"
#include "third_party/blink/renderer/modules/ml/ml_context.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_object.h"
#include "third_party/blink/renderer/platform/heap/member.h"
#include "third_party/blink/renderer/platform/heap/visitor.h"
#include "third_party/blink/renderer/platform/mojo/heap_mojo_remote.h"

namespace blink {

class ML;
class ScriptState;
class ScriptPromiseResolver;
class MojoClient;

using ml::webnn::mojom::blink::Context;

// The class represents the remote of WebNN context which is bound to the
// context of server side. The `MLContext` is a global state of neural network
// compute workload and execution processes.
class MojoContext : public MLContext, public ObjectHandle {
 public:
  MojoContext(ScriptState*,
              ScriptPromiseResolver*,
              scoped_refptr<MojoClient>,
              ML*);

  MojoContext(const MojoContext&) = delete;
  MojoContext& operator=(const MojoContext&) = delete;

  ~MojoContext() override;

  void CreateGraph(ScriptPromiseResolver*,
                   ObjectId,
                   Context::CreateGraphCallback);

  void Trace(Visitor* visitor) const override;

 private:
  void OnContextCreated(ScriptState*,
                        ScriptPromiseResolver* resolver,
                        mojo::PendingRemote<Context>);

  HeapMojoRemote<Context> remote_context_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_MOJO_CONTEXT_H_
