// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_H_

#include "third_party/blink/renderer/bindings/core/v8/v8_union_arraybufferallowshared_arraybufferviewallowshared.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_union_arraybufferviewallowshared_mltensor.h"
#include "third_party/blink/renderer/core/typed_arrays/array_buffer_view_helpers.h"
#include "third_party/blink/renderer/core/typed_arrays/dom_array_buffer_view.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_builder.h"
#include "third_party/blink/renderer/platform/bindings/script_wrappable.h"
#include "third_party/blink/renderer/platform/heap/collection_support/heap_vector.h"
#include "third_party/blink/renderer/platform/heap/member.h"
#include "third_party/blink/renderer/platform/heap/visitor.h"

namespace blink {

typedef HeapVector<
    std::pair<String, Member<V8UnionArrayBufferViewAllowSharedOrMLTensor>>>
    MLNamedArrayInputs;
typedef HeapVector<std::pair<
    String,
    Member<V8UnionArrayBufferAllowSharedOrArrayBufferViewAllowShared>>>
    MLNamedArrayOutputs;

class MLGraph : public ScriptWrappable {
  DEFINE_WRAPPERTYPEINFO();

 public:
  explicit MLGraph(MLContext* context);

  MLGraph(const MLGraph&) = delete;
  MLGraph& operator=(const MLGraph&) = delete;

  ~MLGraph() override;

  void Trace(Visitor* visitor) const override;

  // ml_graph.idl
  void compute(const MLNamedArrayInputs& inputs,
               const MLNamedArrayOutputs& outputs,
               ExceptionState& exception_state);

  virtual bool BuildImpl(
      const MLNamedOperands& named_outputs,
      const HeapVector<Member<const MLOperand>>& inputs,
      const HeapVector<Member<const MLOperand>>& constants,
      const HeapVector<Member<const MLOperator>>& sorted_operators,
      ExceptionState& exception_state) = 0;

  virtual void ComputeImpl(const MLNamedArrayInputs& inputs,
                           const MLNamedArrayOutputs& outputs,
                           ExceptionState& exception_state) = 0;

 protected:
  Member<MLContext> ml_context_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_H_
