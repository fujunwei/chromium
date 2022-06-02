// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/ml_operator.h"

#include "third_party/blink/renderer/modules/ml/ml_context.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_builder.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operand.h"

namespace blink {

MLOperator::MLOperator(MLGraphBuilder* builder, MLOperator::OpKind kind)
    : builder_(builder), kind_(kind) {}

MLOperator::~MLOperator() = default;

void MLOperator::Trace(Visitor* visitor) const {
  visitor->Trace(builder_);
  visitor->Trace(options_);
  visitor->Trace(inputs_);
  visitor->Trace(outputs_);
  ScriptWrappable::Trace(visitor);
}

HeapVector<Member<const MLOperand>>& MLOperator::Inputs() {
  return inputs_;
}

const HeapVector<Member<const MLOperand>>& MLOperator::Inputs() const {
  return inputs_;
}

HeapVector<Member<const MLOperand>>& MLOperator::Outputs() {
  return outputs_;
}

const HeapVector<Member<const MLOperand>>& MLOperator::Outputs() const {
  return outputs_;
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
