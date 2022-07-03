// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_OPERATOR_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_OPERATOR_H_

#include "third_party/blink/renderer/modules/ml/webnn/ml_object.h"
#include "third_party/blink/renderer/platform/bindings/dictionary_base.h"
#include "third_party/blink/renderer/platform/bindings/script_wrappable.h"
#include "third_party/blink/renderer/platform/heap/collection_support/heap_vector.h"
#include "third_party/blink/renderer/platform/heap/member.h"
#include "third_party/blink/renderer/platform/heap/visitor.h"

namespace blink {

class MLGraphBuilder;
class MLOperand;

class MLOperator : public MLObject {
  DEFINE_WRAPPERTYPEINFO();

 public:
  enum OpKind {
    kClamp,
    kConv2d,
    // Element-wise binary operations
    kAdd,
    kGemm,
    // Pooling operations
    kAveragePool2d,
    kRelu,
    kReshape,
    kSoftmax
  };
  explicit MLOperator(MLGraphBuilder* builder, OpKind kind);

  MLOperator(const MLOperator&) = delete;
  MLOperator& operator=(const MLOperator&) = delete;

  ~MLOperator() override;

  void Trace(Visitor* visitor) const override;

  OpKind Kind() const;
  const HeapVector<Member<const MLOperand>>& Inputs() const;
  void SetInputs(HeapVector<Member<const MLOperand>>);
  const HeapVector<Member<const MLOperand>>& Outputs() const;
  void SetOutputs(HeapVector<Member<const MLOperand>>);

  void SetOptions(const bindings::DictionaryBase* options);
  const bindings::DictionaryBase* Options() const;

 protected:
  Member<MLGraphBuilder> builder_;
  OpKind kind_;
  Member<const bindings::DictionaryBase> options_;
  HeapVector<Member<const MLOperand>> inputs_;
  HeapVector<Member<const MLOperand>> outputs_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_OPERATOR_H_
