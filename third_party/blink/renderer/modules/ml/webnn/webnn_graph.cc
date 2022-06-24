// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/webnn_graph.h"

#include "third_party/blink/renderer/bindings/core/v8/script_promise_resolver.h"
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

namespace {

using ml::webnn::mojom::blink::AutoPad;
using ml::webnn::mojom::blink::BinaryOperandType;
using ml::webnn::mojom::blink::BuildResult;
using ml::webnn::mojom::blink::ComputeResult;
using ml::webnn::mojom::blink::Conv2dFilterOperandLayout;
using ml::webnn::mojom::blink::FusionType;
using ml::webnn::mojom::blink::InputOperandLayout;
using ml::webnn::mojom::blink::OperandType;

OperandType ConvertBlinkOperandTypeToMojo(V8MLOperandType::Enum type) {
  switch (type) {
    case V8MLOperandType::Enum::kFloat32:
      return OperandType::kFloat32;
    case V8MLOperandType::Enum::kFloat16:
      return OperandType::kFloat16;
    case V8MLOperandType::Enum::kInt32:
      return OperandType::kInt32;
    case V8MLOperandType::Enum::kUint32:
      return OperandType::kUint32;
    case V8MLOperandType::Enum::kInt8:
      return OperandType::kInt8;
    case V8MLOperandType::Enum::kUint8:
      return OperandType::kUint8;
  }
}

InputOperandLayout ConvertBlinkInputOperandLayoutToMojo(
    V8MLInputOperandLayout::Enum type) {
  switch (type) {
    case V8MLInputOperandLayout::Enum::kNchw:
      return InputOperandLayout::kNchw;
    case V8MLInputOperandLayout::Enum::kNhwc:
      return InputOperandLayout::kNhwc;
  }
}

Conv2dFilterOperandLayout ConvertBlinkConv2dFilterOperandLayoutToMojo(
    V8MLConv2dFilterOperandLayout::Enum type) {
  switch (type) {
    case V8MLConv2dFilterOperandLayout::Enum::kOihw:
      return Conv2dFilterOperandLayout::kOihw;
    case V8MLConv2dFilterOperandLayout::Enum::kHwio:
      return Conv2dFilterOperandLayout::kHwio;
    case V8MLConv2dFilterOperandLayout::Enum::kOhwi:
      return Conv2dFilterOperandLayout::kOhwi;
    case V8MLConv2dFilterOperandLayout::Enum::kIhwo:
      return Conv2dFilterOperandLayout::kIhwo;
  }
}

AutoPad ConvertBlinkAutoPadToMojo(V8MLAutoPad::Enum type) {
  switch (type) {
    case V8MLAutoPad::Enum::kExplicit:
      return AutoPad::kExplicit;
    case V8MLAutoPad::Enum::kSameUpper:
      return AutoPad::kSameUpper;
    case V8MLAutoPad::Enum::kSameLower:
      return AutoPad::kSameLower;
  }
}

FusionType ConvertBlinkFusionTypeToMojo(MLOperator::OpKind type) {
  switch (type) {
    case MLOperator::OpKind::kClamp:
      return FusionType::kClamp;
    case MLOperator::OpKind::kRelu:
      return FusionType::kRelu;
    default: {
      // TODO: how to deal with the default.
      assert(0);
      return FusionType::kRelu;
    }
  }
}

}  // namespace

WebnnGraph::WebnnGraph(ScriptState* script_state,
                       ScriptPromiseResolver* resolver,
                       MLContext* context,
                       MLNamedOperands named_outputs,
                       HeapVector<Member<const MLOperand>> inputs,
                       HeapVector<Member<const MLOperand>> constants,
                       HeapVector<Member<const MLOperator>> sorted_operators)
    : MLGraph(context),
      remote_graph_(ExecutionContext::From(script_state)),
      named_outputs_(named_outputs),
      inputs_(inputs),
      constants_(constants),
      sorted_operators_(sorted_operators) {
  WebnnContext* webnn_context = static_cast<WebnnContext*>(context);
  webnn_context->CreateGraph(
      GetObjectId(), webnn_context->GetObjectId(),
      WTF::Bind(&WebnnGraph::OnGraphCreated, WrapPersistent(this),
                WrapPersistent(script_state), WrapPersistent(resolver)));
}

WebnnGraph::~WebnnGraph() {}

bool WebnnGraph::BuildImpl(
    const MLNamedOperands& named_outputs,
    const HeapVector<Member<const MLOperand>>& inputs,
    const HeapVector<Member<const MLOperand>>& constants,
    const HeapVector<Member<const MLOperator>>& sorted_operators,
    ExceptionState& exception_state) {
  if (!BuildGraph(named_outputs, inputs, constants, sorted_operators)) {
    return false;
  }
  // remote_graph_->Build();
  return true;
}

bool WebnnGraph::BuildGraph(
    const MLNamedOperands& named_outputs,
    const HeapVector<Member<const MLOperand>>& inputs,
    const HeapVector<Member<const MLOperand>>& constants,
    const HeapVector<Member<const MLOperator>>& sorted_operators) {
  for (const auto& input : inputs) {
    auto desc = ml::webnn::mojom::blink::OperandDescriptor::New();
    desc->data_type = ConvertBlinkOperandTypeToMojo(input->Type());
    desc->dimensions = input->Dimensions();
    desc->object_id = input->GetObjectId();
    remote_graph_->AddInput(input->Name(), std::move(desc));
  }
  for (const auto& constant : constants) {
    auto desc = ml::webnn::mojom::blink::OperandDescriptor::New();
    desc->data_type = ConvertBlinkOperandTypeToMojo(constant->Type());
    desc->dimensions = constant->Dimensions();
    desc->object_id = constant->GetObjectId();

    auto* array_buffer_view = constant->ArrayBufferView();
    wtf_size_t size =
        base::checked_cast<wtf_size_t>(array_buffer_view->byteLength());
    Vector<uint8_t> tensor(size);
    memcpy(tensor.data(), array_buffer_view->BaseAddressMaybeShared(), size);
    remote_graph_->AddConstant(std::move(desc), std::move(tensor));
  }
  for (const auto& op : sorted_operators) {
    auto* output = op->Outputs()[0].Get();
    auto* input = op->Inputs()[0].Get();
    auto desc = ml::webnn::mojom::blink::OperandDescriptor::New();
    desc->dimensions = output->Dimensions();
    desc->object_id = output->GetObjectId();
    switch (op->Kind()) {
      case MLOperator::OpKind::kClamp: {
        const MLClampOptions* ml_options =
            static_cast<const MLClampOptions*>(op->Options());
        const float min = ml_options->hasMinValue()
                              ? ml_options->minValue()
                              : -std::numeric_limits<float>::infinity();
        const float max = ml_options->hasMaxValue()
                              ? ml_options->maxValue()
                              : +std::numeric_limits<float>::infinity();
        auto options = ml::webnn::mojom::blink::ClampOptions::New();
        options->minValue = min;
        options->maxValue = max;
        remote_graph_->AddClamp(input->GetObjectId(), std::move(options),
                                std::move(desc));
        break;
      }
      case MLOperator::OpKind::kConv2d: {
        const MLConv2dOptions* ml_options =
            static_cast<const MLConv2dOptions*>(op->Options());
        auto options = ml::webnn::mojom::blink::Conv2dOptions::New();
        options->padding = ml_options->hasPadding() ? ml_options->padding()
                                                    : Vector<int32_t>(4, 0);
        options->strides = ml_options->hasStrides() ? ml_options->strides()
                                                    : Vector<int32_t>(2, 1);
        options->dilations = ml_options->hasDilations()
                                 ? ml_options->dilations()
                                 : Vector<int32_t>(2, 1);
        options->autoPad =
            ConvertBlinkAutoPadToMojo(ml_options->autoPad().AsEnum());
        options->groups = ml_options->groups();
        options->inputLayout = ConvertBlinkInputOperandLayoutToMojo(
            ml_options->inputLayout().AsEnum());
        options->filterLayout = ConvertBlinkConv2dFilterOperandLayoutToMojo(
            ml_options->filterLayout().AsEnum());
        options->bias_id =
            ml_options->hasBias() ? ml_options->bias()->GetObjectId() : 0;
        if (ml_options->hasActivation()) {
          options->activation = ml::webnn::mojom::blink::FusionOperator::New();
          options->activation->object_id =
              ml_options->activation()->GetObjectId();
          options->activation->fusion_type =
              ConvertBlinkFusionTypeToMojo(ml_options->activation()->Kind());
        }
        auto* filter = op->Inputs()[1].Get();
        remote_graph_->AddConv2d(input->GetObjectId(), filter->GetObjectId(),
                                 std::move(options), std::move(desc));
        break;
      }
      case MLOperator::OpKind::kAdd: {
        auto* input1 = op->Inputs()[1].Get();
        remote_graph_->AddElementWiseBinary(
            input->GetObjectId(), input1->GetObjectId(),
            BinaryOperandType::kAdd, std::move(desc));
        break;
      }
      case MLOperator::OpKind::kGemm: {
        // const MLGemmOptions* options =
        //     static_cast<const MLGemmOptions*>(op->Options());
        // if (!DefineGemm(subgraph.get(), tensors_map, op, options,
        //                 exception_state)) {
        //   return false;
        // }
        break;
      }
      case MLOperator::OpKind::kAveragePool2d: {
        // const MLPool2dOptions* options =
        //     static_cast<const MLPool2dOptions*>(op->Options());
        // if (!DefinePool2d(subgraph.get(), tensors_map, op, options,
        //                   exception_state)) {
        //   return false;
        // }
        break;
      }
      case MLOperator::OpKind::kRelu: {
        // if (!DefineUnary(subgraph.get(), tensors_map, op, exception_state)) {
        //   return false;
        // }
        break;
      }
      case MLOperator::OpKind::kReshape: {
        // if (!DefineReshape(subgraph.get(), tensors_map, op, exception_state))
        // {
        //   return false;
        // }
        break;
      }
      case MLOperator::OpKind::kSoftmax: {
        // if (!DefineUnary(subgraph.get(), tensors_map, op, exception_state)) {
        //   return false;
        // }
        break;
      }
      default:
        return false;
    }
  }
  for (const auto& [name, output] : named_outputs) {
    remote_graph_->AddOutput(name, output->GetObjectId());
  }
  return true;
}

void WebnnGraph::OnBuildFinished(ScriptPromiseResolver* resolver,
                                 BuildResult result) {
  switch (result) {
    case BuildResult::kUnknownError: {
      resolver->Reject(MakeGarbageCollected<DOMException>(
          DOMExceptionCode::kUnknownError, "Internal error."));
      return;
    }
    case BuildResult::kMissingInput: {
      resolver->Reject(MakeGarbageCollected<DOMException>(
          DOMExceptionCode::kDataError, "The build operation missing input."));
      return;
    }
    case BuildResult::kOk: {
      resolver->Resolve(this);
      return;
    }
  }
}

ScriptPromise WebnnGraph::ComputeAsyncImpl(ScriptState* script_state,
                                           const MLNamedArrayInputs& inputs,
                                           const MLNamedArrayOutputs& outputs,
                                           ExceptionState& exception_state) {
  HashMap<String, Vector<uint8_t>> input_mojo;
  for (const auto& input : inputs) {
    DOMArrayBufferView* array_buffer_view = nullptr;
    if (input.second->IsArrayBufferViewAllowShared()) {
      array_buffer_view = input.second->GetAsArrayBufferViewAllowShared().Get();
    } else if (input.second->IsMLTensor()) {
      auto* ml_tensor = input.second->GetAsMLTensor();
      array_buffer_view = ml_tensor->data().Get();
    }
    DCHECK(array_buffer_view != nullptr);
    wtf_size_t size =
        base::checked_cast<wtf_size_t>(array_buffer_view->byteLength());
    Vector<uint8_t> tensor(size);
    memcpy(tensor.data(), array_buffer_view->BaseAddressMaybeShared(), size);

    input_mojo.insert(input.first, std::move(tensor));
  }
  Vector<String> output_names;
  for (const auto& output : outputs) {
    output_names.push_back(output.first);
  }
  ScriptPromiseResolver* resolver =
      MakeGarbageCollected<ScriptPromiseResolver>(script_state);
  remote_graph_->ComputeAsync(
      std::move(input_mojo), std::move(output_names),
      WTF::Bind(&WebnnGraph::OnGraphComputed, WrapPersistent(this),
                WrapPersistent(resolver)));
  named_array_outputs_ = std::move(outputs);
  return resolver->Promise();
}

void WebnnGraph::ComputeImpl(const MLNamedArrayInputs& inputs,
                             const MLNamedArrayOutputs& outputs,
                             ExceptionState& exception_state) {}

void WebnnGraph::Trace(Visitor* visitor) const {
  visitor->Trace(remote_graph_);
  visitor->Trace(named_outputs_);
  visitor->Trace(inputs_);
  visitor->Trace(constants_);
  visitor->Trace(sorted_operators_);
  visitor->Trace(named_array_outputs_);
  MLGraph::Trace(visitor);
}

void WebnnGraph::OnGraphCreated(
    ScriptState* script_state,
    ScriptPromiseResolver* resolver,
    mojo::PendingRemote<ml::webnn::mojom::blink::Graph> pending_remote) {
  auto* execution_context = ExecutionContext::From(script_state);
  remote_graph_.Bind(
      std::move(pending_remote),
      execution_context->GetTaskRunner(TaskType::kInternalDefault));

  BuildGraph(named_outputs_, inputs_, constants_, sorted_operators_);
  remote_graph_->BuildAsync(WTF::Bind(&WebnnGraph::OnBuildFinished,
                                      WrapPersistent(this),
                                      WrapPersistent(resolver)));
  return;
}

void WebnnGraph::OnGraphComputed(
    ScriptPromiseResolver* resolver,
    ComputeResult result,
    const absl::optional<Vector<Vector<uint8_t>>>& output_buffers) {
  if (result != ComputeResult::kOk || !output_buffers.has_value()) {
    resolver->Reject(MakeGarbageCollected<DOMException>(
        DOMExceptionCode::kOperationError,
        "Failed to obtain the computation result."));
    return;
  }
  wtf_size_t i = 0;
  for (const auto& [_, output] : named_array_outputs_) {
    void* output_buffer_address = nullptr;
    if (output->IsArrayBufferViewAllowShared()) {
      DOMArrayBufferView* array_buffer_view =
          output->GetAsArrayBufferViewAllowShared().Get();
      output_buffer_address = array_buffer_view->BaseAddressMaybeShared();
    } else if (output->IsArrayBufferAllowShared()) {
      DOMArrayBufferBase* array_buffer = output->GetAsArrayBufferAllowShared();
      output_buffer_address = array_buffer->DataMaybeShared();
    }
    DCHECK(output_buffer_address);
    const Vector<uint8_t>& output_buffer = output_buffers.value().at(i++);
    memcpy(output_buffer_address, output_buffer.data(), output_buffer.size());
  }
  resolver->Resolve();
  return;
}

}  // namespace blink
