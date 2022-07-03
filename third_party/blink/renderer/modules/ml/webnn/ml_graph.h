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
#include "third_party/blink/renderer/modules/ml/webnn/ml_object.h"
#include "third_party/blink/renderer/platform/bindings/script_wrappable.h"
#include "third_party/blink/renderer/platform/heap/collection_support/heap_vector.h"
#include "third_party/blink/renderer/platform/heap/member.h"
#include "third_party/blink/renderer/platform/heap/visitor.h"

namespace blink {

typedef std::pair<String, Member<V8UnionArrayBufferViewAllowSharedOrMLTensor>>
    MLNamedInput;
typedef HeapVector<MLNamedInput> MLNamedArrayInputs;
typedef std::pair<
    String,
    Member<V8UnionArrayBufferAllowSharedOrArrayBufferViewAllowShared>>
    MLNamedOutput;
typedef HeapVector<MLNamedOutput> MLNamedArrayOutputs;

class MLContext;
class MLOperator;
class MLOperand;

class MLGraph : public MLObject {
  DEFINE_WRAPPERTYPEINFO();

 public:
  explicit MLGraph(MLContext* context);

  MLGraph(const MLGraph&) = delete;
  MLGraph& operator=(const MLGraph&) = delete;

  ~MLGraph() override;

  virtual ScriptPromise BuildImpl(ScriptState* script_state,
                                  MLNamedOperands named_outputs,
                                  ExceptionState& exception_state) = 0;

  virtual void BuildSyncImpl(MLNamedOperands named_outputs,
                             ExceptionState& exception_state) = 0;

  virtual ScriptPromise ComputeImpl(ScriptState* script_state,
                                    MLNamedArrayInputs inputs,
                                    MLNamedArrayOutputs outputs,
                                    ExceptionState& exception_state) = 0;

  virtual void ComputeSyncImpl(MLNamedArrayInputs inputs,
                               MLNamedArrayOutputs outputs,
                               ExceptionState& exception_state) = 0;

 protected:
  struct BuildRequest final : public GarbageCollected<BuildRequest> {
    explicit BuildRequest(MLNamedOperands named_outputs);
    void Trace(Visitor*) const;

    MLNamedOperands outputs_;
    HeapVector<Member<const MLOperand>> inputs_;
    HeapVector<Member<const MLOperand>> constants_;
    HeapVector<Member<const MLOperator>> sorted_operators_;
  };

  struct ComputeRequest final : public GarbageCollected<ComputeRequest> {
    ComputeRequest(MLNamedArrayInputs inputs, MLNamedArrayOutputs outputs);
    void Trace(Visitor*) const;

    MLNamedArrayInputs inputs_;
    MLNamedArrayOutputs outputs_;
  };

  DOMArrayBufferView* ValidateInputBuffer(const MLNamedInput& named_input,
                                          String& error_message);
  void* ValidateOutputBuffer(const MLNamedOutput& named_output,
                             String& error_message);

  // The map of input name and input data length.
  HashMap<String, size_t> inputs_byte_length_;
  // The map of output name and output data length.
  HashMap<String, size_t> outputs_byte_length_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_H_
