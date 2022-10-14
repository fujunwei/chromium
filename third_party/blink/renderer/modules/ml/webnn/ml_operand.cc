// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/ml_operand.h"

#include "components/ml/mojom/webnn_graph.mojom-blink.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_builder.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operator.h"

namespace blink {

namespace {

size_t GetBytesPerElement(V8MLOperandType::Enum operand_type) {
  switch (operand_type) {
    case V8MLOperandType::Enum::kFloat32:
      return sizeof(float);
    case V8MLOperandType::Enum::kFloat16:
      // Using Uint16Array for float16 is a workaround of WebNN spec issue:
      // https://github.com/webmachinelearning/webnn/issues/127
      return sizeof(uint16_t);
    case V8MLOperandType::Enum::kInt32:
      return sizeof(int32_t);
    case V8MLOperandType::Enum::kUint32:
      return sizeof(uint32_t);
    case V8MLOperandType::Enum::kInt8:
      return sizeof(int8_t);
    case V8MLOperandType::Enum::kUint8:
      return sizeof(uint8_t);
  }
}

}  // namespace

absl::optional<size_t> ValidateAndCalculateElementsNumber(
    const Vector<uint32_t>& dimensions,
    String& error_message) {
  if (dimensions.empty()) {
    error_message = "The dimensions is empty.";
    return absl::nullopt;
  }
  base::CheckedNumeric<size_t> checked_elements_number = 1;
  for (auto& d : dimensions) {
    if (d == 0) {
      error_message = "All dimensions should be positive";
      return absl::nullopt;
    }
    checked_elements_number *= d;
  }
  if (!checked_elements_number.IsValid()) {
    error_message = "The elements number of the dimensions is too large.";
    return absl::nullopt;
  }
  return checked_elements_number.ValueOrDie();
}

absl::optional<size_t> ValidateAndCalculateByteLength(
    V8MLOperandType::Enum type,
    const Vector<uint32_t>& dimensions,
    String& error_message) {
  absl::optional<size_t> elements_num =
      ValidateAndCalculateElementsNumber(dimensions, error_message);
  if (!elements_num) {
    return absl::nullopt;
  }
  base::CheckedNumeric<size_t> checked_byte_length =
      elements_num.value() * GetBytesPerElement(type);
  if (!checked_byte_length.IsValid()) {
    error_message = "The byte length of the dimensions is too large.";
    return absl::nullopt;
  }
  return checked_byte_length.ValueOrDie();
}

// static
MLOperand* MLOperand::CreateInput(MLGraphBuilder* builder,
                                  const V8MLOperandType::Enum type,
                                  Vector<uint32_t> dimensions,
                                  String name) {
  auto* input = MakeGarbageCollected<MLOperand>(builder, OperandKind::kInput,
                                                type, std::move(dimensions));
  input->name_ = std::move(name);
  return input;
}

// static
MLOperand* MLOperand::CreateConstant(
    MLGraphBuilder* builder,
    const V8MLOperandType::Enum type,
    Vector<uint32_t> dimensions,
    const DOMArrayBufferView* array_buffer_view) {
  auto* constant = MakeGarbageCollected<MLOperand>(
      builder, OperandKind::kConstant, type, std::move(dimensions));
  constant->array_buffer_view_ = array_buffer_view;
  return constant;
}

// static
MLOperand* MLOperand::CreateOutput(MLGraphBuilder* builder,
                                   const V8MLOperandType::Enum type,
                                   Vector<uint32_t> dimensions,
                                   const MLOperator* ml_operator) {
  auto* output = MakeGarbageCollected<MLOperand>(builder, OperandKind::kOutput,
                                                 type, std::move(dimensions));
  output->operator_ = ml_operator;
  return output;
}

MLOperand::MLOperand(MLGraphBuilder* builder,
                     OperandKind kind,
                     const V8MLOperandType::Enum type,
                     Vector<uint32_t> dimensions)
    : builder_(builder),
      kind_(kind),
      type_(type),
      dimensions_(std::move(dimensions)) {}

MLOperand::~MLOperand() = default;

MLGraphBuilder* MLOperand::Builder() const {
  return builder_.Get();
}

MLOperand::OperandKind MLOperand::Kind() const {
  return kind_;
}

V8MLOperandType::Enum MLOperand::Type() const {
  return type_;
}

const Vector<uint32_t>& MLOperand::Dimensions() const {
  return dimensions_;
}

const String& MLOperand::Name() const {
  DCHECK_EQ(kind_, OperandKind::kInput);
  return name_;
}

const DOMArrayBufferView* MLOperand::ArrayBufferView() const {
  DCHECK_EQ(kind_, OperandKind::kConstant);
  return array_buffer_view_.Get();
}

const MLOperator* MLOperand::Operator() const {
  // DCHECK_EQ(kind_, OperandKind::kOutput);
  return operator_.Get();
}

void MLOperand::Trace(Visitor* visitor) const {
  visitor->Trace(builder_);
  visitor->Trace(array_buffer_view_);
  visitor->Trace(operator_);
  ScriptWrappable::Trace(visitor);
}

}  // namespace blink
