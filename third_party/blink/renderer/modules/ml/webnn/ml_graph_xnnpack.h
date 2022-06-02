// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_XNNPACK_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_XNNPACK_H_

#include "third_party/blink/renderer/modules/ml/webnn/ml_graph.h"
#include "third_party/blink/renderer/platform/wtf/hash_map.h"
#include "third_party/blink/renderer/platform/wtf/text/string_hash.h"
#include "third_party/blink/renderer/platform/wtf/text/wtf_string.h"
#include "third_party/xnnpack/src/include/xnnpack.h"

#include <unordered_map>

namespace blink {

class MLGraphXnnpack : public MLGraph {
 public:
  explicit MLGraphXnnpack(MLContext* context);
  ~MLGraphXnnpack() override;

  bool BuildImpl(const MLNamedOperands& named_outputs,
                 const HeapVector<Member<const MLOperand>>& inputs,
                 const HeapVector<Member<const MLOperand>>& constants,
                 const HeapVector<Member<const MLOperator>>& sorted_operators,
                 ExceptionState& exception_state) override;

  void ComputeImpl(const MLNamedArrayInputs& inputs,
                   const MLNamedArrayOutputs& outputs,
                   ExceptionState& exception_state) override;

 private:
  bool DefineTensor(xnn_subgraph_t,
                    HashMap<Member<const MLOperand>, uint32_t>&,
                    const MLOperand*,
                    ExceptionState&,
                    bool external = false);
  bool DefineClamp(xnn_subgraph_t,
                   HashMap<Member<const MLOperand>, uint32_t>&,
                   const MLOperator*,
                   const MLClampOptions*,
                   ExceptionState&);
  bool DefineConv2d(xnn_subgraph_t,
                    HashMap<Member<const MLOperand>, uint32_t>&,
                    const MLOperator*,
                    const MLConv2dOptions*,
                    ExceptionState&);
  bool DefineBinary(xnn_subgraph_t,
                    HashMap<Member<const MLOperand>, uint32_t>&,
                    const MLOperator*,
                    ExceptionState&);
  bool DefineGemm(xnn_subgraph_t,
                  HashMap<Member<const MLOperand>, uint32_t>&,
                  const MLOperator*,
                  const MLGemmOptions*,
                  ExceptionState&);
  bool DefinePool2d(xnn_subgraph_t,
                    HashMap<Member<const MLOperand>, uint32_t>&,
                    const MLOperator*,
                    const MLPool2dOptions*,
                    ExceptionState&);
  bool DefineReshape(xnn_subgraph_t,
                     HashMap<Member<const MLOperand>, uint32_t>&,
                     const MLOperator*,
                     ExceptionState&);
  bool DefineUnary(xnn_subgraph_t,
                   HashMap<Member<const MLOperand>, uint32_t>&,
                   const MLOperator*,
                   ExceptionState&);

  Vector<std::unique_ptr<char>> constant_data_;

  struct TensorValueInfo {
    uint32_t id;
    size_t byte_length;
  };
  HashMap<String, TensorValueInfo> inputs_info_;
  HashMap<String, TensorValueInfo> outputs_info_;
  xnn_runtime_t runtime_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_XNNPACK_H_
