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
class ScriptPromiseResolver;

using ml::webnn::mojom::blink::FusionOperatorPtr;
using ml::webnn::mojom::blink::NamedOutputsPtr;
using ml::webnn::mojom::blink::OperandDescriptorPtr;

class WebnnGraph : public MLGraph {
 public:
  WebnnGraph(ScriptState* script_state,
             ScriptPromiseResolver* resolver,
             MLContext* context,
             MLNamedOperands named_outputs,
             HeapVector<Member<const MLOperand>> inputs,
             HeapVector<Member<const MLOperand>> constants,
             HeapVector<Member<const MLOperator>> sorted_operators);
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
  ScriptPromise ComputeAsyncImpl(ScriptState* script_state,
                                 const MLNamedArrayInputs& inputs,
                                 const MLNamedArrayOutputs& outputs,
                                 ExceptionState& exception_state) override;

 private:
  void AddPool2d(const MLOperator* pool2d, OperandDescriptorPtr desc);
  void AddUnary(const MLOperator* unary, OperandDescriptorPtr desc);
  FusionOperatorPtr AddFusionOperator(const MLOperator* activation);

  bool BuildGraph(const MLNamedOperands& named_outputs,
                  const HeapVector<Member<const MLOperand>>& inputs,
                  const HeapVector<Member<const MLOperand>>& constants,
                  const HeapVector<Member<const MLOperator>>& sorted_operators);
  void OnGraphCreated(ScriptState*,
                      ScriptPromiseResolver*,
                      mojo::PendingRemote<ml::webnn::mojom::blink::Graph>);
  void OnBuildFinished(ScriptPromiseResolver*,
                       ml::webnn::mojom::blink::BuildResult);
  void OnGraphComputed(ScriptPromiseResolver* resolver,
                       ml::webnn::mojom::blink::ComputeResult result,
                       NamedOutputsPtr named_outputs);

  HeapMojoRemote<ml::webnn::mojom::blink::Graph> remote_graph_;
  MLNamedOperands named_outputs_;
  HeapVector<Member<const MLOperand>> inputs_;
  HeapVector<Member<const MLOperand>> constants_;
  HeapVector<Member<const MLOperator>> sorted_operators_;
  MLNamedArrayOutputs named_array_outputs_;

  struct MemoryInfo {
    size_t byte_offset;
    size_t byte_length;
    mojo::ScopedSharedBufferMapping mapping;
  };
  HashMap<String, MemoryInfo> inputs_info_;
  mojo::ScopedSharedBufferHandle input_buffer_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_WEBNN_GRAPH_H_
