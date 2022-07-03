// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_MOJO_MODEL_INFO_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_MOJO_MODEL_INFO_H_

#include "components/ml/mojom/webnn_graph.mojom-blink.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operand.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operator.h"
#include "third_party/blink/renderer/platform/heap/collection_support/heap_hash_map.h"

namespace blink {

#if BUILDFLAG(IS_WIN)
static const uint32_t kBufferAlignment = 16;
#else
static const uint32_t kBufferAlignment = 1;
#endif
base::CheckedNumeric<size_t> Align(size_t value, uint32_t aligment);

using ml::webnn::mojom::blink::ModelInfoPtr;

class MojoModelInfo final : public GarbageCollected<MojoModelInfo> {
 public:
  MojoModelInfo();
  ~MojoModelInfo();

  MojoModelInfo(const MojoModelInfo&) = delete;
  MojoModelInfo& operator=(const MojoModelInfo&) = delete;

  void Trace(Visitor* visitor) const;

  void AddInput(const MLOperand* input);
  void AddConstant(const MLOperand* constant);
  void AddOutput(String name, const MLOperand* output);

  // The order of operations declaration is the same as spec.
  void AddClamp(const MLOperator* clamp);

  void AddConv2d(const MLOperator* conv2d);

  // Element-wise binary operations
  void AddElementWiseBinary(const MLOperator* binary);

  void AddGemm(const MLOperator* gemm);

  // Pooling operations
  void AddPool2d(const MLOperator* pool2d);

  void AddRelu(const MLOperator* relu);

  void AddReshape(const MLOperator* reshape);

  void AddSoftmax(const MLOperator* softmax);

  void FillConstantsWithArrayBuffer();

  ModelInfoPtr GetModelInfo();

 private:
  // Add a operand to model which is output of the operation.
  size_t AddOperandToModel(const MLOperand* output);
  // Hold all operands of model to index the operand.
  HeapHashMap<Member<const MLOperand>, size_t> operand_index_map_;
  // All constant data will share a big shared memory, so hold the index of
  // constant temporarily.
  HeapHashMap<Member<const MLOperand>, size_t> constant_index_map_;
  ModelInfoPtr model_info_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_MOJO_MODEL_INFO_H_
