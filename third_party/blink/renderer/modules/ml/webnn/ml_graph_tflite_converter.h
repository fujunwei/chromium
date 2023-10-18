// Copyright 2023 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_TFLITE_CONVERTER_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_TFLITE_CONVERTER_H_

#include "base/types/expected.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_builder.h"
#include "third_party/blink/renderer/platform/heap/collection_support/heap_hash_map.h"
#include "third_party/blink/renderer/platform/wtf/vector.h"
#include "third_party/flatbuffers/src/include/flatbuffers/flatbuffers.h"
#include "third_party/tflite/src/tensorflow/lite/schema/schema_generated.h"

namespace blink {

class MLOperand;
class MLOperator;

// The class used to convert WebNN graph to tflite model and persist into
// FlatBuffer. the schema_generated.h file defines the format for each data
// structure to serialize.
class MLGraphTfLiteConverter final {
 public:
  MLGraphTfLiteConverter();
  ~MLGraphTfLiteConverter();

  MLGraphTfLiteConverter(const MLGraphTfLiteConverter&) = delete;
  MLGraphTfLiteConverter& operator=(const MLGraphTfLiteConverter&) = delete;

  // Serializes the constant data (e.g. weights) to the flat buffer and returns
  // the index in the `tflite::Buffer` array if it's successful.
  //
  // The `Buffer` in TF-Lite schema is the table of raw data buffers, it is used
  // for WebNN constant operations. Referenced by tensors with the index of
  // buffer.
  base::expected<uint32_t, String> SerializeBuffer(const MLOperand* constant);

  // Serialize tensor for input, constant and output operand, it's output
  // operand of graph if the `graph_output_name` is specified, returns the index
  // in the `tflite::Tensor` array if it's successful.
  base::expected<int32_t, String> SerializeTensor(
      const MLOperand* operand,
      absl::optional<String> graph_output_name = absl::nullopt);

  // Returns error messages if it could not be serialized because of unsupported
  // options or it is otherwise invalid.
  base::expected<void, String> SerializeOperation(
      const HeapHashMap<Member<const MLOperand>, int32_t>& operand_index_map,
      const MLOperator* ml_operator);

  base::expected<flatbuffers::DetachedBuffer, String> FinishAndGetFlatBuffer();

 private:
  flatbuffers::FlatBufferBuilder builder_;

  // Hold the index of graph inputs.
  Vector<int32_t> graph_inputs_;
  // Hold the index of graph outputs.
  Vector<int32_t> graph_outputs_;
  // The buffers of the model. The first entry of this array must be an empty
  // buffer.
  Vector<flatbuffers::Offset<tflite::Buffer>> buffers_;
  // An tensor index starting from 0 that refers to the tensors table which is
  // used to create `Operator` and `SubGraph`.
  Vector<flatbuffers::Offset<tflite::Tensor>> tensors_;
  // A list of all operator codes used in this model. This is kept in order
  // because operators carry an index into this vector.
  Vector<flatbuffers::Offset<tflite::OperatorCode>> operator_codes_;
  Vector<flatbuffers::Offset<tflite::Operator>> operators_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_TFLITE_CONVERTER_H_
