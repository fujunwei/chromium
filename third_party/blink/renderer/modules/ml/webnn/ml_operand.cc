// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/ml_operand.h"

#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_builder.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operator.h"

namespace blink {

MLOperand::MLOperand(MLGraphBuilder* builder, KindEnum kind)
    : builder_(builder), kind_(kind), dimensions_({1}) {}

MLOperand::~MLOperand() = default;

MLGraphBuilder* MLOperand::Builder() const {
  return builder_.Get();
}

MLOperand::KindEnum MLOperand::Kind() const {
  return kind_;
}

void MLOperand::SetName(const String& name) {
  name_ = name;
}

const String& MLOperand::Name() const {
  return name_;
}

void MLOperand::SetOperator(const MLOperator* ml_operator) {
  operator_ = ml_operator;
}

const MLOperator* MLOperand::Operator() const {
  return operator_.Get();
}

void MLOperand::SetType(const V8MLOperandType::Enum type) {
  type_ = type;
}

V8MLOperandType::Enum MLOperand::Type() const {
  return type_;
}

void MLOperand::SetDimensions(const Vector<int32_t>& dimensions) {
  dimensions_ = dimensions;
}

const Vector<int32_t>& MLOperand::Dimensions() const {
  return dimensions_;
}

void MLOperand::SetArrayBufferView(
    const DOMArrayBufferView* array_buffer_view) {
  array_buffer_view_ = array_buffer_view;
}

const DOMArrayBufferView* MLOperand::ArrayBufferView() const {
  return array_buffer_view_.Get();
}

void MLOperand::Trace(Visitor* visitor) const {
  visitor->Trace(builder_);
  visitor->Trace(operator_);
  visitor->Trace(array_buffer_view_);
  ScriptWrappable::Trace(visitor);
}

}  // namespace blink
