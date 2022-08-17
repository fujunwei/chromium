// Copyright 2022 The Chromium Authors. All rights reserved.
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

using ObjectId = uint32_t;
using ml::webnn::mojom::blink::ContextOptionsPtr;
using ml::webnn::mojom::blink::MojoServer;

// The class is used to manage all objects id and create context in server side
// with `MojoServer`.
// `MojoServer` is a one-to-one mapping with `MojoClient`.
class MojoClient final : public RefCounted<MojoClient> {
 public:
  static scoped_refptr<MojoClient> Create(ExecutionContext*);

  explicit MojoClient(ExecutionContext*);

  MojoClient(const MojoClient&) = delete;
  MojoClient& operator=(const MojoClient&) = delete;

  void CreateMojoContext(ScriptPromiseResolver*,
                         ObjectId,
                         ContextOptionsPtr,
                         MojoServer::CreateContextCallback);

  void Trace(Visitor* visitor) const;

  uint32_t GetNewId();

  void FreeId(uint32_t id);

 private:
  base::CheckedNumeric<uint32_t> current_id_ = 1;
  Vector<uint32_t> free_ids_;

  HeapMojoRemote<MojoServer> mojo_server_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_MOJO_CLIENT_H_
