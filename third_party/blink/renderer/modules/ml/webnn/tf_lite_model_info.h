// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_TF_LITE_MODEL_INFO_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_TF_LITE_MODEL_INFO_H_

#include "third_party/blink/renderer/modules/ml/webnn/ml_operand.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operator.h"
#include "third_party/blink/renderer/platform/heap/collection_support/heap_hash_map.h"
#include "third_party/flatbuffers/src/include/flatbuffers/flatbuffers.h"
#include "third_party/tflite/src/tensorflow/lite/schema/schema_generated.h"

namespace blink {

class TFLiteModelInfo final : public GarbageCollected<TFLiteModelInfo> {
 public:
  TFLiteModelInfo();
  ~TFLiteModelInfo();

  TFLiteModelInfo(const TFLiteModelInfo&) = delete;
  TFLiteModelInfo& operator=(const TFLiteModelInfo&) = delete;

  void Trace(Visitor* visitor) const;

  // The `Buffer` in TF-Lite schema is the table of raw data buffers, it is used
  // for constant operations in WebNN Spec. Referenced by tensors by index.
  void BuildBuffer(const MLOperand* constant);
  // A list of all tensors used in this model including input, constant and
  // output operand defined in WebNN.
  int32_t BuildTensor(const MLOperand* operand, uint32_t buffer_index = 0);
  void BuildOperator(const MLOperator* op);
  void BuildModel(const std::vector<int32_t>& subgraph_inputs,
                  const std::vector<int32_t>& subgraph_outputs);

  int32_t GetTensorIndex(const MLOperand* operand);
  flatbuffers::FlatBufferBuilder& GetFlatBufferBuilder();

 private:
  // Add a operand to model which is output of the operation.
  size_t AddOperandToModel(const MLOperand* output);

  // // The order of operations declaration is the same as spec.
  // void AddClamp(const MLOperator* clamp);

  // void AddConv2d(const MLOperator* conv2d);

  // Element-wise binary operations
  void AddElementWiseBinary(const MLOperator* binary);

  // void AddGemm(const MLOperator* gemm);

  // // Pooling operations
  // void AddPool2d(const MLOperator* pool2d);

  // void AddRelu(const MLOperator* relu);

  // void AddReshape(const MLOperator* reshape);

  // void AddSoftmax(const MLOperator* softmax);

  // The index of buffer is not equal the index of tensor. For example, The
  // Input tensors index is {1}, but the Constant buffer index is 1 and tensor
  // index is 0 in the following topology:
  //    Constant  Input
  //      \        /
  //     Elementwise Add
  //           |
  //         output
  struct OperandIndex {
    // An index starting from 1 that refers to the buffers table which is used
    // to create the constant's tensor. if there is no data buffer associated
    // (i.e. intermediate results), then this is 0.
    size_t buffer_index;
    // An index starting from 0 that refers to the tensors table which is used
    // to create `Operator` and `SubGraph`.
    size_t tensor_index;
  };

  // Hold all operands of model to index the operand.
  HeapHashMap<Member<const MLOperand>, int32_t> operand_index_map_;
  flatbuffers::FlatBufferBuilder builder_;
  std::vector<flatbuffers::Offset<tflite::Buffer>> buffers_;
  std::vector<flatbuffers::Offset<tflite::Tensor>> tensors_;
  // A list of all operator codes used in this model. This is
  // kept in order because operators carry an index into this
  // vector.
  std::vector<flatbuffers::Offset<tflite::OperatorCode>> operator_codes_;
  std::vector<flatbuffers::Offset<tflite::Operator>> operators_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_TF_LITE_MODEL_INFO_H_
