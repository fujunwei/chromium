// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_BUILDER_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_BUILDER_H_

#include "third_party/blink/renderer/bindings/core/v8/script_promise.h"
#include "third_party/blink/renderer/core/typed_arrays/array_buffer_view_helpers.h"
#include "third_party/blink/renderer/core/typed_arrays/dom_array_buffer_view.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operator.h"
#include "third_party/blink/renderer/platform/bindings/script_wrappable.h"
#include "third_party/blink/renderer/platform/heap/collection_support/heap_vector.h"
#include "third_party/blink/renderer/platform/heap/member.h"
#include "third_party/blink/renderer/platform/heap/visitor.h"

namespace blink {

class ExceptionState;
class MLContext;
class MLClampOptions;
class MLConv2dOptions;
class MLGemmOptions;
class MLGraph;
class MLPool2dOptions;
class MLOperand;
class MLOperandDescriptor;
class ScriptPromise;

typedef HeapVector<std::pair<String, Member<MLOperand>>> MLNamedOperands;

class MLGraphBuilder final : public ScriptWrappable {
  DEFINE_WRAPPERTYPEINFO();

 public:
  static MLGraphBuilder* Create(MLContext* context);

  explicit MLGraphBuilder(MLContext* context);

  MLGraphBuilder(const MLGraphBuilder&) = delete;
  MLGraphBuilder& operator=(const MLGraphBuilder&) = delete;

  ~MLGraphBuilder() override;

  void Trace(Visitor* visitor) const override;

  MLContext* GetContext() const;

  // ml_graph_builder.idl
  MLOperand* input(String name,
                   const MLOperandDescriptor* desc,
                   ExceptionState& exception_state);
  MLOperand* constant(const MLOperandDescriptor* desc,
                      MaybeShared<DOMArrayBufferView> buffer_view,
                      ExceptionState& exception_state);

  // The order of operations declaration is the same as spec.
  MLOperand* clamp(const MLOperand*, const MLClampOptions*, ExceptionState&);
  MLOperator* clamp(const MLClampOptions*, ExceptionState&);

  MLOperand* conv2d(const MLOperand*,
                    const MLOperand*,
                    const MLConv2dOptions*,
                    ExceptionState&);

  // Element-wise binary operations
  MLOperand* add(const MLOperand*, const MLOperand*, ExceptionState&);

  MLOperand* gemm(const MLOperand*,
                  const MLOperand*,
                  const MLGemmOptions*,
                  ExceptionState&);

  // Pooling operations
  MLOperand* averagePool2d(const MLOperand*,
                           const MLPool2dOptions*,
                           ExceptionState&);

  MLOperand* relu(const MLOperand*, ExceptionState&);
  MLOperator* relu(ExceptionState&);

  MLOperand* reshape(const MLOperand*, const Vector<int32_t>&, ExceptionState&);

  MLOperand* softmax(const MLOperand*, ExceptionState&);

  ScriptPromise build(ScriptState* script_state,
                      MLNamedOperands outputs,
                      ExceptionState& exception_state);

  MLGraph* buildSync(MLNamedOperands outputs, ExceptionState& exception_state);

  static void SortOperators(
      const MLNamedOperands& named_outputs,
      HeapVector<Member<const MLOperand>>& inputs,
      HeapVector<Member<const MLOperand>>& constants,
      HeapVector<Member<const MLOperator>>& sorted_operators);

 private:
  MLOperand* BuildElementWiseBinary(MLOperator::OpKind,
                                    const MLOperand*,
                                    const MLOperand*,
                                    ExceptionState&);
  MLOperand* BuildElementWiseUnary(MLOperator::OpKind,
                                   const MLOperand*,
                                   ExceptionState&);
  MLOperand* BuildPool2d(MLOperator::OpKind,
                         const MLOperand*,
                         const MLPool2dOptions*,
                         ExceptionState&);

  Member<MLContext> ml_context_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_BUILDER_H_
