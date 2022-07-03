// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_MOJO_CLIENT_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_MOJO_CLIENT_H_

#include "base/memory/scoped_refptr.h"
#include "components/ml/mojom/webnn_service.mojom-blink.h"
#include "third_party/blink/renderer/core/execution_context/execution_context.h"
#include "third_party/blink/renderer/platform/heap/member.h"
#include "third_party/blink/renderer/platform/heap/visitor.h"
#include "third_party/blink/renderer/platform/mojo/heap_mojo_remote.h"
#include "third_party/blink/renderer/platform/wtf/ref_counted.h"
#include "third_party/blink/renderer/platform/wtf/vector.h"

namespace blink {

class ScriptPromiseResolver;

using ml::webnn::mojom::blink::ContextOptionsPtr;
using ml::webnn::mojom::blink::MojoServer;

// The class is used to manage all objects id and create context in server side
// with `MojoServer`.
// `MojoServer` is a one-to-one mapping with `MojoClient`.
class MojoClient final : public GarbageCollected<MojoClient> {
 public:
  explicit MojoClient(ExecutionContext*);

  MojoClient(const MojoClient&) = delete;
  MojoClient& operator=(const MojoClient&) = delete;

  void CreateMojoContext(ScriptPromiseResolver*,
                         ContextOptionsPtr,
                         MojoServer::CreateContextCallback);

  void Trace(Visitor* visitor) const;

 private:
  HeapMojoRemote<MojoServer> mojo_server_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_MOJO_CLIENT_H_
