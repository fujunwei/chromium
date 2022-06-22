// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_WEBNN_CLIENT_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_WEBNN_CLIENT_H_

#include <memory>

#include "components/ml/webnn/mojom/webnn_service.mojom-blink.h"
#include "third_party/blink/renderer/core/execution_context/execution_context.h"
#include "third_party/blink/renderer/modules/ml/ml_context.h"
#include "third_party/blink/renderer/platform/heap/member.h"
#include "third_party/blink/renderer/platform/heap/visitor.h"
#include "third_party/blink/renderer/platform/mojo/heap_mojo_remote.h"
#include "third_party/blink/renderer/platform/wtf/ref_counted.h"

namespace blink {

class MLContext;
class MLContextOptions;
class ScriptState;

class WebnnWireClient final : public RefCounted<WebnnWireClient> {
 public:
  static scoped_refptr<WebnnWireClient> Create(ExecutionContext*);

  explicit WebnnWireClient(ExecutionContext*);

  WebnnWireClient(const WebnnWireClient&) = delete;
  WebnnWireClient& operator=(const WebnnWireClient&) = delete;

  void CreateWebnnContext(
      uint32_t,
      ml::webnn::mojom::blink::ContextOptionsPtr,
      ml::webnn::mojom::blink::WireServer::CreateContextCallback);

  void Trace(Visitor* visitor) const;

  uint32_t GetNewId();

  void FreeId(uint32_t id);

 private:
  void OnWebnnServiceConnectionError();

  HeapMojoRemote<ml::webnn::mojom::blink::WireServer> wire_server_;

  uint32_t current_id_ = 1;
  std::vector<uint32_t> free_ids_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_WEBNN_CLIENT_H_
