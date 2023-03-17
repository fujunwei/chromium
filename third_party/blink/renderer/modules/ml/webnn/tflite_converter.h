// Copyright 2023 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_TFLITE_CONVERTER_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_TFLITE_CONVERTER_H_

#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_builder.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operand.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operator.h"
#include "third_party/blink/renderer/platform/heap/collection_support/heap_hash_map.h"
#include "third_party/flatbuffers/src/include/flatbuffers/flatbuffers.h"
#include "third_party/tflite/src/tensorflow/lite/schema/schema_generated.h"

namespace blink {

using OperandTensorIndexMap = HeapHashMap<Member<const MLOperand>, int32_t>;
// The output name map so that the name can be got quickly with operand when
// building output tensor.
using OperandNameMap = HeapHashMap<Member<const MLOperand>, String>;
using OperatorOffset = flatbuffers::Offset<tflite::Operator>;

// The class used to convert WebNN graph to tflite model and persist into
// FlatBuffer. the schema_generated.h file defines the format for each data
// structure to serialize.
class TfLiteConverter final : public GarbageCollected<TfLiteConverter> {
 public:
  TfLiteConverter();
  ~TfLiteConverter();

  TfLiteConverter(const TfLiteConverter&) = delete;
  TfLiteConverter& operator=(const TfLiteConverter&) = delete;

  void Trace(Visitor* visitor) const;

  // Serializes the constant data (e.g. weights) to the FlatBuffer |builder_|
  // and returns success. Returns false if it could not be serialized because of
  // unsupported options or it is otherwise invalid.
  //
  // The `Buffer` in TF-Lite schema is the table of raw data buffers, it is used
  // for WebNN constant operations. Referenced by tensors with the index of
  // buffer.
  bool SerializeConstant(const MLOperand* constant,
                         OperandTensorIndexMap& operand_index_map);
  // Serializes the |input| operand to the FlatBuffer |builder_|.
  bool SerializeInput(const MLOperand* input,
                      OperandTensorIndexMap& operand_index_map);
  // Serializes the |output| operand to the FlatBuffer |builder_|.
  bool SerializeOutput(const MLOperand* output,
                       const OperandNameMap& output_operand_name_map,
                       OperandTensorIndexMap& operand_index_map);
  bool SerializeOperations(const MLOperator* op,
                           const OperandTensorIndexMap& operand_index_map,
                           String& error_message);

  bool BuildModel();
  flatbuffers::FlatBufferBuilder& GetFlatBufferBuilder();

 private:
  // The buffer index starting from 1 that refers to the buffers table which is
  // used to create the constant's tensor. if there is no data buffer associated
  // (i.e. intermediate results), then this is 0. The index of buffer is not
  // equal the index of tensor.
  //
  // For example, The |Input| tensors index is {1}, but the |Constant| buffer
  // index is 1 and tensor index is 0 in the following topology:
  //    Constant  Input
  //      \        /
  //     ElementwiseBinary
  //           |
  //         output
  bool SerializeTensor(const MLOperand* operand,
                       uint32_t buffer_index,
                       String name = "");
  uint32_t SerializeEmptyBuffer(uint32_t output_channels);
  OperatorOffset SerializeConv2d(const MLOperator* conv2d,
                                 const OperandTensorIndexMap& operand_index_map,
                                 String& error_message);
  OperatorOffset SerializePool2d(const MLOperator* pool2d,
                                 const OperandTensorIndexMap& operand_index_map,
                                 String& error_message);
  OperatorOffset SerializeElementWiseBinary(
      const MLOperator* binary,
      const OperandTensorIndexMap& operand_index_map);
  OperatorOffset SerializeRelu(const MLOperator* ml_relu,
                               const OperandTensorIndexMap& operand_index_map);
  OperatorOffset SerializeSoftmax(
      const MLOperator* ml_softmax,
      const OperandTensorIndexMap& operand_index_map);
  OperatorOffset SerializeReshape(
      const MLOperator* ml_reshape,
      const OperandTensorIndexMap& operand_index_map);

  flatbuffers::FlatBufferBuilder builder_;

  struct TfLiteSubGraphDesc {
    // An tensor index starting from 0 that refers to the tensors table which is
    // used to create `Operator` and `SubGraph`.
    Vector<flatbuffers::Offset<tflite::Tensor>> tensors;
    // Hold the index of subgraph inputs.
    Vector<int32_t> inputs_index;
    // Hold the index of subgraph outputs.
    Vector<int32_t> outputs_index;
    Vector<flatbuffers::Offset<tflite::Operator>> operators;
  };
  // Represent tflite::SubGraph
  TfLiteSubGraphDesc subgraph_desc_;

  struct TfLiteModel {
    // A list of all operator codes used in this model. This is
    // kept in order because operators carry an index into this
    // vector.
    Vector<flatbuffers::Offset<tflite::OperatorCode>> operator_codes;
    // The buffers of the model. The first entry of this array must be an empty
    // buffer.
    Vector<flatbuffers::Offset<tflite::Buffer>> buffers;
  };
  // Represent tflite::Model
  TfLiteModel model_info_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_TFLITE_CONVERTER_H_
