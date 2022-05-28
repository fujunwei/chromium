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

class WebnnClient final : public RefCounted<WebnnClient> {
 public:
  static scoped_refptr<WebnnClient> Create(ExecutionContext*);

  explicit WebnnClient(ExecutionContext*);

  WebnnClient(const WebnnClient&) = delete;
  WebnnClient& operator=(const WebnnClient&) = delete;

  void CreateWebnnContext(
      uint32_t,
      ml::webnn::mojom::blink::ContextOptionsPtr,
      ml::webnn::mojom::blink::NeuralNetwork::CreateContextCallback);

  void Trace(Visitor* visitor) const;

  uint32_t GetNewId();

  void FreeId(uint32_t id);

 private:
  void OnWebnnServiceConnectionError();

  HeapMojoRemote<ml::webnn::mojom::blink::NeuralNetwork> webnn_service_;

  uint32_t current_id_ = 1;
  std::vector<uint32_t> free_ids_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_WEBNN_CLIENT_H_
