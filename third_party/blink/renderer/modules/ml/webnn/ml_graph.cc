// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/ml_graph.h"

#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_tensor.h"
#include "third_party/blink/renderer/modules/ml/ml_context.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operand.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operator.h"

namespace blink {

MLGraph::MLGraph(MLContext* context) : context_(context) {}

MLGraph::~MLGraph() = default;

void MLGraph::Trace(Visitor* visitor) const {
  visitor->Trace(context_);
  ScriptWrappable::Trace(visitor);
}

MLGraph::ComputeRequest::ComputeRequest(MLNamedArrayInputs inputs,
                                        MLNamedArrayOutputs outputs)
    : inputs_(std::move(inputs)), outputs_(std::move(outputs)) {}

void MLGraph::ComputeRequest::Trace(Visitor* visitor) const {
  visitor->Trace(inputs_);
  visitor->Trace(outputs_);
}

DOMArrayBufferView* MLGraph::ValidateInputBuffer(const MLNamedInput& input,
                                                 String& error_message) {
  // TODO:: Use output_resources_info_ instead of input_length_map_
  auto iter = input_length_map_.find(input.first);
  // if (iter == input_length_map_.end()) {
  //   error_message = "There is unknown input: " + input.first;
  //   return nullptr;
  // }
  DOMArrayBufferView* array_buffer_view = nullptr;
  if (input.second->IsArrayBufferViewAllowShared()) {
    array_buffer_view = input.second->GetAsArrayBufferViewAllowShared().Get();
  } else if (input.second->IsMLTensor()) {
    auto* ml_tensor = input.second->GetAsMLTensor();
    array_buffer_view = ml_tensor->data().Get();
  }
  DCHECK(array_buffer_view != nullptr);
  if (array_buffer_view->byteLength() < iter->value) {
    error_message = "Wrong size of input: " + input.first;
    return nullptr;
  }
  return array_buffer_view;
}

void* MLGraph::ValidateOutputBuffer(const MLNamedOutput& output,
                                    String& error_message) {
  // TODO:: Use output_resources_info_ instead of output_length_map_
  // auto iter = output_length_map_.find(output.first);
  // if (iter == output_length_map_.end()) {
  //   error_message = "There is unknown output: " + output.first;
  //   return nullptr;
  // }
  void* output_buffer_address = nullptr;
  if (output.second->IsArrayBufferViewAllowShared()) {
    DOMArrayBufferView* array_buffer_view =
        output.second->GetAsArrayBufferViewAllowShared().Get();
    // if (array_buffer_view->byteLength() < iter->value) {
    //   error_message = "Wrong size of output: " + output.first;
    //   return nullptr;
    // }
    output_buffer_address = array_buffer_view->BaseAddressMaybeShared();
  } else if (output.second->IsArrayBufferAllowShared()) {
    DOMArrayBufferBase* array_buffer =
        output.second->GetAsArrayBufferAllowShared();
    // if (array_buffer->ByteLength() < iter->value) {
    //   error_message = "Wrong size of output: " + output.first;
    //   return nullptr;
    // }
    output_buffer_address = array_buffer->DataMaybeShared();
  }
  return output_buffer_address;
}

}  // namespace blink
