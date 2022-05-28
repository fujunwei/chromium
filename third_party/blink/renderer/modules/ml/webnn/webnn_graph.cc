// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/webnn_graph.h"

#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_clamp_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_conv_2d_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_gemm_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_operand_descriptor.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_pool_2d_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_tensor.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"
#include "third_party/blink/renderer/core/execution_context/execution_context.h"
#include "third_party/blink/renderer/modules/ml/ml_context.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operand.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operator.h"
#include "third_party/blink/renderer/modules/ml/webnn/webnn_context.h"
#include "third_party/blink/renderer/platform/bindings/exception_state.h"
#include "third_party/blink/renderer/platform/wtf/functional.h"

#include <memory>

namespace blink {

WebnnGraph::WebnnGraph(MLContext* context, ExecutionContext* execution_context)
    : MLGraph(context),
      execution_context_(execution_context),
      remote_graph_(execution_context) {
  WebnnContext* webnn_context = static_cast<WebnnContext*>(context);
  webnn_context->CreateGraph(
      GetObjectId(), webnn_context->GetObjectId(),
      WTF::Bind(&WebnnGraph::OnGraphCreated, WrapPersistent(this)));
}

WebnnGraph::~WebnnGraph() {}

bool WebnnGraph::BuildImpl(
    const MLNamedOperands& named_outputs,
    const HeapVector<Member<const MLOperand>>& inputs,
    const HeapVector<Member<const MLOperand>>& constants,
    const HeapVector<Member<const MLOperator>>& sorted_operators,
    ExceptionState& exception_state) {
  // HashMap<Member<const MLOperand>, uint32_t> tensors_map;
  // uint32_t external_id = 0;
  // for (const auto& input : inputs) {
  //   uint32_t input_id = external_id++;
  //   tensors_map.insert(input, input_id);
  //   if (!DefineTensor(subgraph.get(), tensors_map, input, exception_state,
  //                     true)) {
  //     return false;
  //   }
  //   TensorValueInfo info = {0};
  //   info.id = input_id;
  //   info.byte_length = GetByteLength(input);
  //   inputs_info_.insert(input->Name(), std::move(info));
  // }
  // for (const auto& named_output : named_outputs) {
  //   auto* output = named_output.second.Get();
  //   uint32_t output_id = external_id++;
  //   tensors_map.insert(output, output_id);
  //   if (!DefineTensor(subgraph.get(), tensors_map, output, exception_state,
  //                     true)) {
  //     return false;
  //   }
  //   TensorValueInfo info = {0};
  //   info.id = output_id;
  //   info.byte_length = GetByteLength(output);
  //   outputs_info_.insert(named_output.first, std::move(info));
  // }
  // for (const auto& constant : constants) {
  //   if (!DefineTensor(subgraph.get(), tensors_map, constant,
  //   exception_state)) {
  //     return false;
  //   }
  // }
  // for (const auto& op : sorted_operators) {
  //   switch (op->Kind()) {
  //     case MLOperator::OpKind::kClamp: {
  //       const MLClampOptions* options =
  //           static_cast<const MLClampOptions*>(op->Options());
  //       if (!DefineClamp(subgraph.get(), tensors_map, op, options,
  //                        exception_state)) {
  //         return false;
  //       }
  //       break;
  //     }
  //     case MLOperator::OpKind::kConv2d: {
  //       const MLConv2dOptions* options =
  //           static_cast<const MLConv2dOptions*>(op->Options());
  //       if (!DefineConv2d(subgraph.get(), tensors_map, op, options,
  //                         exception_state)) {
  //         return false;
  //       }
  //       break;
  //     }
  //     case MLOperator::OpKind::kAdd: {
  //       if (!DefineBinary(subgraph.get(), tensors_map, op, exception_state))
  //       {
  //         return false;
  //       }
  //       break;
  //     }
  //     case MLOperator::OpKind::kGemm: {
  //       const MLGemmOptions* options =
  //           static_cast<const MLGemmOptions*>(op->Options());
  //       if (!DefineGemm(subgraph.get(), tensors_map, op, options,
  //                       exception_state)) {
  //         return false;
  //       }
  //       break;
  //     }
  //     case MLOperator::OpKind::kAveragePool2d: {
  //       const MLPool2dOptions* options =
  //           static_cast<const MLPool2dOptions*>(op->Options());
  //       if (!DefinePool2d(subgraph.get(), tensors_map, op, options,
  //                         exception_state)) {
  //         return false;
  //       }
  //       break;
  //     }
  //     case MLOperator::OpKind::kRelu: {
  //       if (!DefineUnary(subgraph.get(), tensors_map, op, exception_state)) {
  //         return false;
  //       }
  //       break;
  //     }
  //     case MLOperator::OpKind::kReshape: {
  //       if (!DefineReshape(subgraph.get(), tensors_map, op, exception_state))
  //       {
  //         return false;
  //       }
  //       break;
  //     }
  //     case MLOperator::OpKind::kSoftmax: {
  //       if (!DefineUnary(subgraph.get(), tensors_map, op, exception_state)) {
  //         return false;
  //       }
  //       break;
  //     }
  //     default:
  //       exception_state.ThrowDOMException(DOMExceptionCode::kNotSupportedError,
  //                                         "the operator (" +
  //                                             OpKindToString(op->Kind()) +
  //                                             ") is not supported");
  //       return false;
  //   }
  // }
  // uint32_t flags = XNN_FLAG_YIELD_WORKERS;
  // if (xnn_create_runtime_v2(
  //         subgraph.get(),
  //         static_cast<MLContextXnnpack*>(ml_context_.Get())->Pthreadpool(),
  //         flags, &runtime_) != xnn_status_success) {
  //   exception_state.ThrowDOMException(DOMExceptionCode::kOperationError,
  //                                     "failed to create XNNPACK runtime");
  //   return false;
  // }
  return true;
}

void WebnnGraph::ComputeImpl(const MLNamedArrayInputs& inputs,
                             const MLNamedArrayOutputs& outputs,
                             ExceptionState& exception_state) {
  // Vector<xnn_external_value> external_values;
  // if (inputs.size() != inputs_info_.size()) {
  //   exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
  //                                     "The number of inputs is invalid");
  //   return;
  // }
  // if (outputs.size() != outputs_info_.size()) {
  //   exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
  //                                     "The number of outputs is invalid");
  //   return;
  // }
  // for (const auto& input : inputs) {
  //   auto iter = inputs_info_.find(input.first);
  //   if (iter == inputs_info_.end()) {
  //     exception_state.ThrowDOMException(
  //         DOMExceptionCode::kDataError,
  //         "There is unknown input: " + input.first);
  //     return;
  //   }
  //   xnn_external_value value = {0};
  //   value.id = iter->value.id;
  //   DOMArrayBufferView* array_buffer_view = nullptr;
  //   if (input.second->IsArrayBufferViewAllowShared()) {
  //     array_buffer_view =
  //     input.second->GetAsArrayBufferViewAllowShared().Get();
  //   } else if (input.second->IsMLTensor()) {
  //     auto* ml_tensor = input.second->GetAsMLTensor();
  //     array_buffer_view = ml_tensor->data().Get();
  //   }
  //   DCHECK(array_buffer_view != nullptr);
  //   if (array_buffer_view->byteLength() < iter->value.byte_length) {
  //     exception_state.ThrowDOMException(
  //         DOMExceptionCode::kDataError,
  //         "The input (" + input.first + ") buffer length is invalid.");
  //     return;
  //   }
  //   value.data = array_buffer_view->BaseAddressMaybeShared();
  //   external_values.push_back(value);
  // }
  // for (const auto& output : outputs) {
  //   auto iter = outputs_info_.find(output.first);
  //   if (iter == outputs_info_.end()) {
  //     exception_state.ThrowDOMException(
  //         DOMExceptionCode::kDataError,
  //         "There is unknown output: " + output.first);
  //     return;
  //   }
  //   xnn_external_value value = {0};
  //   value.id = iter->value.id;
  //   if (output.second->IsArrayBufferViewAllowShared()) {
  //     DOMArrayBufferView* array_buffer_view =
  //         output.second->GetAsArrayBufferViewAllowShared().Get();
  //     if (array_buffer_view->byteLength() < iter->value.byte_length) {
  //       exception_state.ThrowDOMException(
  //           DOMExceptionCode::kDataError,
  //           "The output (" + output.first + ") buffer length is invalid.");
  //       return;
  //     }
  //     value.data = array_buffer_view->BaseAddressMaybeShared();
  //   } else if (output.second->IsArrayBufferAllowShared()) {
  //     DOMArrayBufferBase* array_buffer =
  //         output.second->GetAsArrayBufferAllowShared();
  //     if (array_buffer->ByteLength() < iter->value.byte_length) {
  //       exception_state.ThrowDOMException(
  //           DOMExceptionCode::kDataError,
  //           "The output (" + output.first + ") buffer length is invalid.");
  //       return;
  //     }
  //     value.data = array_buffer->DataMaybeShared();
  //   }
  //   DCHECK(value.data);
  //   external_values.push_back(value);
  // }
  // if (xnn_setup_runtime(runtime_, external_values.size(),
  //                       external_values.data()) != xnn_status_success) {
  //   exception_state.ThrowDOMException(DOMExceptionCode::kOperationError,
  //                                     "failed to setup runtime");
  //   return;
  // }
  // if (xnn_invoke_runtime(runtime_) != xnn_status_success) {
  //   exception_state.ThrowDOMException(DOMExceptionCode::kOperationError,
  //                                     "failed to invoke runtime");
  //   return;
  // }
}

void WebnnGraph::Trace(Visitor* visitor) const {
  visitor->Trace(execution_context_);
  visitor->Trace(remote_graph_);
  MLGraph::Trace(visitor);
}

void WebnnGraph::OnGraphCreated(
    mojo::PendingRemote<ml::webnn::mojom::blink::Graph> pending_remote) {
  remote_graph_.Bind(
      std::move(pending_remote),
      execution_context_->GetTaskRunner(TaskType::kInternalDefault));
}

}  // namespace blink
