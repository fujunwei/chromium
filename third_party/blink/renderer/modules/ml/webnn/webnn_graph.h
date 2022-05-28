// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_WEBNN_GRAPH_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_WEBNN_GRAPH_H_

#include "components/ml/webnn/mojom/graph.mojom-blink.h"
#include "mojo/public/cpp/bindings/pending_remote.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph.h"
#include "third_party/blink/renderer/modules/ml/webnn/webnn_object.h"
#include "third_party/blink/renderer/platform/heap/member.h"
#include "third_party/blink/renderer/platform/heap/visitor.h"
#include "third_party/blink/renderer/platform/mojo/heap_mojo_remote.h"

namespace blink {

class ExecutionContext;

class WebnnGraph : public MLGraph {
 public:
  explicit WebnnGraph(MLContext* context, ExecutionContext* execution_context);
  ~WebnnGraph() override;

  void Trace(Visitor* visitor) const override;

  bool BuildImpl(const MLNamedOperands& named_outputs,
                 const HeapVector<Member<const MLOperand>>& inputs,
                 const HeapVector<Member<const MLOperand>>& constants,
                 const HeapVector<Member<const MLOperator>>& sorted_operators,
                 ExceptionState& exception_state) override;

  void ComputeImpl(const MLNamedArrayInputs& inputs,
                   const MLNamedArrayOutputs& outputs,
                   ExceptionState& exception_state) override;

 private:
  void OnGraphCreated(
      mojo::PendingRemote<ml::webnn::mojom::blink::Graph> pending_remote);

  Member<ExecutionContext> execution_context_;
  HeapMojoRemote<ml::webnn::mojom::blink::Graph> remote_graph_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_WEBNN_GRAPH_H_
