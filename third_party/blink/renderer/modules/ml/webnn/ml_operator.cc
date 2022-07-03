// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/ml_operator.h"

#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_builder.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operand.h"

namespace blink {

MLOperator::MLOperator(MLGraphBuilder* builder, MLOperator::OpKind kind)
    : MLObject(builder->GetContext()), builder_(builder), kind_(kind) {}

MLOperator::~MLOperator() = default;

void MLOperator::Trace(Visitor* visitor) const {
  visitor->Trace(builder_);
  visitor->Trace(options_);
  visitor->Trace(inputs_);
  visitor->Trace(outputs_);
  MLObject::Trace(visitor);
}

const HeapVector<Member<const MLOperand>>& MLOperator::Inputs() const {
  return inputs_;
}

void MLOperator::SetInputs(HeapVector<Member<const MLOperand>> inputs) {
  inputs_ = std::move(inputs);
}

const HeapVector<Member<const MLOperand>>& MLOperator::Outputs() const {
  return outputs_;
}

void MLOperator::SetOutputs(HeapVector<Member<const MLOperand>> outputs) {
  outputs_ = std::move(outputs);
}

MLOperator::OpKind MLOperator::Kind() const {
  return kind_;
}

void MLOperator::SetOptions(const bindings::DictionaryBase* options) {
  options_ = options;
}

const bindings::DictionaryBase* MLOperator::Options() const {
  return options_;
}

}  // namespace blink
