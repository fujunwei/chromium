// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_OPERAND_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_OPERAND_H_

#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_operand_descriptor.h"
#include "third_party/blink/renderer/core/typed_arrays/array_buffer_view_helpers.h"
#include "third_party/blink/renderer/core/typed_arrays/dom_array_buffer_view.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_object.h"
#include "third_party/blink/renderer/platform/bindings/script_wrappable.h"
#include "third_party/blink/renderer/platform/heap/member.h"
#include "third_party/blink/renderer/platform/heap/visitor.h"

namespace blink {

class MLGraphBuilder;
class MLOperator;

class MLOperand final : public MLObject {
  DEFINE_WRAPPERTYPEINFO();

 public:
  enum KindEnum { kInput, kConstant, kOutput };
  explicit MLOperand(MLGraphBuilder* builder, KindEnum kind = kOutput);

  MLOperand(const MLOperand&) = delete;
  MLOperand& operator=(const MLOperand&) = delete;

  ~MLOperand() override;

  void Trace(Visitor* visitor) const override;

  MLGraphBuilder* Builder() const;
  KindEnum Kind() const;
  void SetName(String name);
  const String& Name() const;
  void SetOperator(const MLOperator* ml_operator);
  const MLOperator* Operator() const;
  void SetType(const V8MLOperandType::Enum type);
  V8MLOperandType::Enum Type() const;
  void SetDimensions(Vector<int32_t> dimensions);
  const Vector<int32_t>& Dimensions() const;
  bool CalculateByteLength(size_t& byte_length) const;
  void SetArrayBufferView(const DOMArrayBufferView* array_buffer_view);
  const DOMArrayBufferView* ArrayBufferView() const;

 private:
  Member<MLGraphBuilder> builder_;
  KindEnum kind_;
  String name_;
  Member<const MLOperator> operator_;
  V8MLOperandType::Enum type_;
  Vector<int32_t> dimensions_;
  Member<const DOMArrayBufferView> array_buffer_view_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_OPERAND_H_
