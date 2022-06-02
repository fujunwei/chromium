// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_xnnpack.h"

#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_clamp_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_conv_2d_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_gemm_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_operand_descriptor.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_pool_2d_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_tensor.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"
#include "third_party/blink/renderer/modules/ml/ml_context.h"
#include "third_party/blink/renderer/modules/ml/ml_context_xnnpack.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operand.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operator.h"
#include "third_party/blink/renderer/platform/bindings/exception_state.h"

#include <memory>

namespace blink {

namespace {

String DataTypeToString(V8MLOperandType::Enum datatype) {
  switch (datatype) {
    case V8MLOperandType::Enum::kFloat32:
      return "float32";
    case V8MLOperandType::Enum::kFloat16:
      return "float16";
    case V8MLOperandType::Enum::kInt32:
      return "int32";
    case V8MLOperandType::Enum::kUint32:
      return "uint32";
    case V8MLOperandType::Enum::kInt8:
      return "int8";
    case V8MLOperandType::Enum::kUint8:
      return "uint8";
  }
}

String OpKindToString(MLOperator::OpKind kind) {
  switch (kind) {
    case MLOperator::OpKind::kClamp:
      return "clamp";
    case MLOperator::OpKind::kConv2d:
      return "conv2d";
    case MLOperator::OpKind::kAdd:
      return "add";
    case MLOperator::OpKind::kGemm:
      return "gemm";
    case MLOperator::OpKind::kAveragePool2d:
      return "averagePool2d";
    case MLOperator::OpKind::kRelu:
      return "relu";
    case MLOperator::OpKind::kReshape:
      return "reshape";
    case MLOperator::OpKind::kSoftmax:
      return "softmax";
  }
}

String XnnStatusToString(xnn_status status) {
  switch (status) {
    case xnn_status_success:
      return "xnn_status_success";
    case xnn_status_uninitialized:
      return "xnn_status_uninitialized";
    case xnn_status_invalid_parameter:
      return "xnn_status_invalid_parameter";
    case xnn_status_invalid_state:
      return "xnn_status_invalid_state";
    case xnn_status_unsupported_parameter:
      return "xnn_status_unsupported_parameter";
    case xnn_status_unsupported_hardware:
      return "xnn_status_unsupported_hardware";
    case xnn_status_out_of_memory:
      return "xnn_status_out_of_memory";
  }
}

size_t GetBytesPerElement(V8MLOperandType::Enum datatype) {
  switch (datatype) {
    case V8MLOperandType::Enum::kFloat32:
      return 4;
    case V8MLOperandType::Enum::kFloat16:
      return 2;
    case V8MLOperandType::Enum::kInt32:
      return 4;
    case V8MLOperandType::Enum::kUint32:
      return 4;
    case V8MLOperandType::Enum::kInt8:
      return 1;
    case V8MLOperandType::Enum::kUint8:
      return 1;
  }
}

size_t GetByteLength(const MLOperand* operand) {
  size_t elements = 1;
  for (auto& d : operand->Dimensions()) {
    elements = elements * d;
  }
  return elements * GetBytesPerElement(operand->Type());
}

}  // namespace

MLGraphXnnpack::MLGraphXnnpack(MLContext* context)
    : MLGraph(context), runtime_(nullptr) {}

MLGraphXnnpack::~MLGraphXnnpack() {
  if (runtime_ != nullptr) {
    xnn_delete_runtime(runtime_);
  }
}

bool MLGraphXnnpack::BuildImpl(
    const MLNamedOperands& named_outputs,
    const HeapVector<Member<const MLOperand>>& inputs,
    const HeapVector<Member<const MLOperand>>& constants,
    const HeapVector<Member<const MLOperator>>& sorted_operators,
    ExceptionState& exception_state) {
  uint32_t externals_size =
      static_cast<uint32_t>(named_outputs.size() + inputs.size());
  xnn_subgraph_t subgraph_ptr = nullptr;
  xnn_status status;
  if ((status = xnn_create_subgraph(externals_size, 0, &subgraph_ptr)) !=
      xnn_status_success) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kOperationError,
        "failed to create XNNPACK subgraph: " + XnnStatusToString(status));
    return false;
  }

  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> subgraph(
      subgraph_ptr, &xnn_delete_subgraph);

  HashMap<Member<const MLOperand>, uint32_t> tensors_map;
  uint32_t external_id = 0;
  for (const auto& input : inputs) {
    uint32_t input_id = external_id++;
    tensors_map.insert(input, input_id);
    if (!DefineTensor(subgraph.get(), tensors_map, input, exception_state,
                      true)) {
      return false;
    }
    TensorValueInfo info = {0};
    info.id = input_id;
    info.byte_length = GetByteLength(input);
    inputs_info_.insert(input->Name(), std::move(info));
  }
  for (const auto& named_output : named_outputs) {
    auto* output = named_output.second.Get();
    uint32_t output_id = external_id++;
    tensors_map.insert(output, output_id);
    if (!DefineTensor(subgraph.get(), tensors_map, output, exception_state,
                      true)) {
      return false;
    }
    TensorValueInfo info = {0};
    info.id = output_id;
    info.byte_length = GetByteLength(output);
    outputs_info_.insert(named_output.first, std::move(info));
  }
  for (const auto& constant : constants) {
    if (!DefineTensor(subgraph.get(), tensors_map, constant, exception_state)) {
      return false;
    }
  }
  for (const auto& op : sorted_operators) {
    switch (op->Kind()) {
      case MLOperator::OpKind::kClamp: {
        const MLClampOptions* options =
            static_cast<const MLClampOptions*>(op->Options());
        if (!DefineClamp(subgraph.get(), tensors_map, op, options,
                         exception_state)) {
          return false;
        }
        break;
      }
      case MLOperator::OpKind::kConv2d: {
        const MLConv2dOptions* options =
            static_cast<const MLConv2dOptions*>(op->Options());
        if (!DefineConv2d(subgraph.get(), tensors_map, op, options,
                          exception_state)) {
          return false;
        }
        break;
      }
      case MLOperator::OpKind::kAdd: {
        if (!DefineBinary(subgraph.get(), tensors_map, op, exception_state)) {
          return false;
        }
        break;
      }
      case MLOperator::OpKind::kGemm: {
        const MLGemmOptions* options =
            static_cast<const MLGemmOptions*>(op->Options());
        if (!DefineGemm(subgraph.get(), tensors_map, op, options,
                        exception_state)) {
          return false;
        }
        break;
      }
      case MLOperator::OpKind::kAveragePool2d: {
        const MLPool2dOptions* options =
            static_cast<const MLPool2dOptions*>(op->Options());
        if (!DefinePool2d(subgraph.get(), tensors_map, op, options,
                          exception_state)) {
          return false;
        }
        break;
      }
      case MLOperator::OpKind::kRelu: {
        if (!DefineUnary(subgraph.get(), tensors_map, op, exception_state)) {
          return false;
        }
        break;
      }
      case MLOperator::OpKind::kReshape: {
        if (!DefineReshape(subgraph.get(), tensors_map, op, exception_state)) {
          return false;
        }
        break;
      }
      case MLOperator::OpKind::kSoftmax: {
        if (!DefineUnary(subgraph.get(), tensors_map, op, exception_state)) {
          return false;
        }
        break;
      }
      default:
        exception_state.ThrowDOMException(DOMExceptionCode::kNotSupportedError,
                                          "the operator (" +
                                              OpKindToString(op->Kind()) +
                                              ") is not supported");
        return false;
    }
  }
  uint32_t flags = XNN_FLAG_YIELD_WORKERS;
  if (xnn_create_runtime_v2(
          subgraph.get(),
          static_cast<MLContextXnnpack*>(ml_context_.Get())->Pthreadpool(),
          flags, &runtime_) != xnn_status_success) {
    exception_state.ThrowDOMException(DOMExceptionCode::kOperationError,
                                      "failed to create XNNPACK runtime");
    return false;
  }
  return true;
}

bool MLGraphXnnpack::DefineTensor(
    xnn_subgraph_t subgraph,
    HashMap<Member<const MLOperand>, uint32_t>& tensors_map,
    const MLOperand* operand,
    ExceptionState& exception_state,
    bool is_external) {
  xnn_datatype data_type = xnn_datatype_invalid;
  if (operand->Type() == V8MLOperandType::Enum::kFloat32) {
    data_type = xnn_datatype_fp32;
  } else {
    exception_state.ThrowDOMException(DOMExceptionCode::kNotSupportedError,
                                      "the data type (" +
                                          DataTypeToString(operand->Type()) +
                                          ") is not supported");
    return false;
  }
  Vector<size_t> dims;
  for (auto& d : operand->Dimensions()) {
    if (d < 0) {
      exception_state.ThrowDOMException(
          DOMExceptionCode::kNotSupportedError,
          "the negative dimension is not supported");
      return false;
    }
    dims.push_back(static_cast<size_t>(d));
  }
  uint32_t flags = 0;
  uint32_t external_id = XNN_INVALID_VALUE_ID;
  std::unique_ptr<char> data;
  if (is_external) {
    DCHECK(tensors_map.find(operand) != tensors_map.end());
    external_id = tensors_map.at(operand);
    if (operand->Kind() == MLOperand::KindEnum::kInput) {
      flags |= XNN_VALUE_FLAG_EXTERNAL_INPUT;
    } else {
      DCHECK(operand->Kind() == MLOperand::KindEnum::kOutput);
      flags |= XNN_VALUE_FLAG_EXTERNAL_OUTPUT;
    }
  } else {
    if (operand->Kind() == MLOperand::KindEnum::kConstant) {
      auto* array_buffer_view = operand->ArrayBufferView();
      data.reset(new char[array_buffer_view->byteLength()]);
      if (data.get() == nullptr) {
        exception_state.ThrowDOMException(DOMExceptionCode::kInvalidStateError,
                                          "out of memory.");
        return false;
      }
      memcpy(data.get(), array_buffer_view->BaseAddressMaybeShared(),
             array_buffer_view->byteLength());
    }
  }
  uint32_t tensor_id;
  xnn_status status;
  if ((status = xnn_define_tensor_value(
           subgraph, data_type, dims.size(), dims.data(), data.get(),
           external_id, flags, &tensor_id)) != xnn_status_success) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kOperationError,
        "failed to define tensor value: " + XnnStatusToString(status));
    return false;
  }
  if (data) {
    constant_data_.push_back(std::move(data));
  }
  if (!is_external) {
    tensors_map.insert(operand, tensor_id);
  }
  return true;
}

bool MLGraphXnnpack::DefineClamp(
    xnn_subgraph_t subgraph,
    HashMap<Member<const MLOperand>, uint32_t>& tensors_map,
    const MLOperator* clamp,
    const MLClampOptions* options,
    ExceptionState& exception_state) {
  auto* input = clamp->Inputs()[0].Get();
  DCHECK(tensors_map.find(input) != tensors_map.end());
  uint32_t input_id = tensors_map.at(input);
  auto* output = clamp->Outputs()[0].Get();
  if (tensors_map.find(output) == tensors_map.end()) {
    if (!DefineTensor(subgraph, tensors_map, output, exception_state)) {
      return false;
    }
  }
  uint32_t output_id = tensors_map.at(output);
  const float output_min = options->hasMinValue()
                               ? options->minValue()
                               : -std::numeric_limits<float>::infinity();
  const float output_max = options->hasMaxValue()
                               ? options->maxValue()
                               : +std::numeric_limits<float>::infinity();
  xnn_status status;
  if ((status = xnn_define_clamp(subgraph, output_min, output_max, input_id,
                                 output_id, 0)) != xnn_status_success) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kOperationError,
        "failed to define clamp: " + XnnStatusToString(status));
    return false;
  }
  return true;
}

bool MLGraphXnnpack::DefineConv2d(
    xnn_subgraph_t subgraph,
    HashMap<Member<const MLOperand>, uint32_t>& tensors_map,
    const MLOperator* conv2d,
    const MLConv2dOptions* options,
    ExceptionState& exception_state) {
  auto* input = conv2d->Inputs()[0].Get();
  DCHECK(tensors_map.find(input) != tensors_map.end());
  uint32_t input_id = tensors_map.at(input);
  auto* filter = conv2d->Inputs()[1].Get();
  DCHECK(tensors_map.find(filter) != tensors_map.end());
  uint32_t filter_id = tensors_map.at(filter);
  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  if (conv2d->Inputs().size() == 3) {
    auto* bias = conv2d->Inputs()[2].Get();
    DCHECK(tensors_map.find(bias) != tensors_map.end());
    bias_id = tensors_map.at(bias);
  }
  auto* output = conv2d->Outputs()[0].Get();
  if (tensors_map.find(output) == tensors_map.end()) {
    if (!DefineTensor(subgraph, tensors_map, output, exception_state)) {
      return false;
    }
  }
  uint32_t output_id = tensors_map.at(output);

  uint32_t groups = options->groups();
  uint32_t stride_height = options->hasStrides() ? options->strides()[0] : 1;
  uint32_t stride_width = options->hasStrides() ? options->strides()[1] : 1;
  uint32_t dilation_height =
      options->hasDilations() ? options->dilations()[0] : 1;
  uint32_t dilation_width =
      options->hasDilations() ? options->dilations()[1] : 1;
  size_t input_height, input_width;
  uint32_t filter_height, filter_width;
  size_t input_channels, output_channels;
  bool depthwise = false;
  if (options->inputLayout().AsEnum() == V8MLInputOperandLayout::Enum::kNhwc) {
    input_height = input->Dimensions()[1];
    input_width = input->Dimensions()[2];
    input_channels = input->Dimensions()[3];
    depthwise = (groups == input_channels);
    if (!depthwise) {
      // For regular conv2d, xnn pack expects weights layed out like (ohwi):
      //   [groups * group_output_channels, kernel_height, kernel_width,
      //   group_input_channels]
      if (options->filterLayout().AsEnum() !=
          V8MLConv2dFilterOperandLayout::Enum::kOhwi) {
        exception_state.ThrowDOMException(
            DOMExceptionCode::kNotSupportedError,
            "only filter layout ohwi for conv2d is supported");
        return false;
      }
    } else {
      // For depthwise conv2d, xnn pack expects weights layed out like (ihwo):
      //   [1, kernel_height, kernel_width, input_channels * depth_multiplier]
      if (options->filterLayout().AsEnum() !=
          V8MLConv2dFilterOperandLayout::Enum::kIhwo) {
        exception_state.ThrowDOMException(
            DOMExceptionCode::kNotSupportedError,
            "only filter layout ihwo for depthwise conv2d is supported");
        return false;
      }
    }
    filter_height = filter->Dimensions()[1];
    filter_width = filter->Dimensions()[2];
    output_channels = output->Dimensions()[3];
  } else {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kNotSupportedError,
        "only input layout nhwc for conv2d is supported");
    return false;
  }
  size_t group_input_channels = input_channels / groups;
  size_t group_output_channels = output_channels / groups;

  size_t output_height, output_width;
  uint32_t pad_top, pad_bottom, pad_left, pad_right;
  if (options->autoPad().AsEnum() == V8MLAutoPad::Enum::kExplicit) {
    // WebNN padding: [beginning_height, ending_height, beginning_width,
    // ending_width]
    pad_top = options->hasPadding() ? options->padding()[0] : 0;
    pad_bottom = options->hasPadding() ? options->padding()[1] : 0;
    pad_left = options->hasPadding() ? options->padding()[2] : 0;
    pad_right = options->hasPadding() ? options->padding()[3] : 0;
  } else {
    output_height = ceil(input_height / stride_height);
    output_width = ceil(input_width / stride_width);
    uint32_t pad_along_height = static_cast<uint32_t>(std::max(
        size_t(0),
        (output_height - 1) * stride_height + filter_height - input_height));
    uint32_t pad_along_width = static_cast<uint32_t>(std::max(
        size_t(0),
        (output_width - 1) * stride_width + filter_width - input_width));
    if (options->autoPad().AsEnum() == V8MLAutoPad::Enum::kSameUpper) {
      pad_top = floor(pad_along_height / 2);
      pad_bottom = pad_along_height - pad_top;
      pad_left = floor(pad_along_width / 2);
      pad_right = pad_along_width - pad_left;
    } else {
      pad_bottom = floor(pad_along_height / 2);
      pad_top = pad_along_height - pad_bottom;
      pad_right = floor(pad_along_width / 2);
      pad_left = pad_along_width - pad_right;
    }
  }

  float output_min = -std::numeric_limits<float>::infinity();
  float output_max = +std::numeric_limits<float>::infinity();
  if (options->hasActivation()) {
    switch (options->activation()->Kind()) {
      case MLOperator::OpKind::kClamp: {
        const MLClampOptions* clamp_options =
            static_cast<const MLClampOptions*>(
                options->activation()->Options());
        if (clamp_options->hasMinValue()) {
          output_min = clamp_options->minValue();
        }
        if (clamp_options->hasMaxValue()) {
          output_max = clamp_options->maxValue();
        }
        break;
      }
      case MLOperator::OpKind::kRelu:
        output_min = 0.0f;
        output_max = std::numeric_limits<float>::infinity();
        break;
      default:
        exception_state.ThrowDOMException(
            DOMExceptionCode::kNotSupportedError,
            "only clamp and relu fused operator are supported");
        return false;
    }
  }
  xnn_status status;
  if (depthwise) {
    status = xnn_define_depthwise_convolution_2d(
        subgraph, pad_top, pad_right, pad_bottom, pad_left, filter_height,
        filter_width, stride_height, stride_width, dilation_height,
        dilation_width, 1, input_channels, output_min, output_max, input_id,
        filter_id, bias_id, output_id, 0);
  } else {
    status = xnn_define_convolution_2d(
        subgraph, pad_top, pad_right, pad_bottom, pad_left, filter_height,
        filter_width, stride_height, stride_width, dilation_height,
        dilation_width, groups, group_input_channels, group_output_channels,
        output_min, output_max, input_id, filter_id, bias_id, output_id, 0);
  }
  if (status != xnn_status_success) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kOperationError,
        "failed to define conv2d: " + XnnStatusToString(status));
    return false;
  }
  return true;
}

bool MLGraphXnnpack::DefineBinary(
    xnn_subgraph_t subgraph,
    HashMap<Member<const MLOperand>, uint32_t>& tensors_map,
    const MLOperator* binary,
    ExceptionState& exception_state) {
  auto* input0 = binary->Inputs()[0].Get();
  DCHECK(tensors_map.find(input0) != tensors_map.end());
  uint32_t input0_id = tensors_map.at(input0);
  auto* input1 = binary->Inputs()[1].Get();
  DCHECK(tensors_map.find(input1) != tensors_map.end());
  uint32_t input1_id = tensors_map.at(input1);
  auto* output = binary->Outputs()[0].Get();
  if (tensors_map.find(output) == tensors_map.end()) {
    if (!DefineTensor(subgraph, tensors_map, output, exception_state)) {
      return false;
    }
  }
  uint32_t output_id = tensors_map.at(output);
  const float output_min = -std::numeric_limits<float>::infinity();
  const float output_max = +std::numeric_limits<float>::infinity();
  xnn_status status = xnn_status_success;
  switch (binary->Kind()) {
    case MLOperator::OpKind::kAdd: {
      status = xnn_define_add2(subgraph, output_min, output_max, input0_id,
                               input1_id, output_id, 0);
      break;
    }
    default:
      NOTREACHED();
  }
  if (status != xnn_status_success) {
    exception_state.ThrowDOMException(DOMExceptionCode::kOperationError,
                                      "failed to define " +
                                          OpKindToString(binary->Kind()) +
                                          ": " + XnnStatusToString(status));
    return false;
  }
  return true;
}

bool MLGraphXnnpack::DefineGemm(
    xnn_subgraph_t subgraph,
    HashMap<Member<const MLOperand>, uint32_t>& tensors_map,
    const MLOperator* gemm,
    const MLGemmOptions* options,
    ExceptionState& exception_state) {
  auto* input = gemm->Inputs()[0].Get();
  DCHECK(tensors_map.find(input) != tensors_map.end());
  uint32_t input_id = tensors_map.at(input);
  auto* filter = gemm->Inputs()[1].Get();
  if (filter->Kind() != MLOperand::KindEnum::kConstant) {
    exception_state.ThrowDOMException(DOMExceptionCode::kNotSupportedError,
                                      "only constant input b is supported");
    return false;
  }
  DCHECK(tensors_map.find(filter) != tensors_map.end());
  uint32_t filter_id = tensors_map.at(filter);
  auto* output = gemm->Outputs()[0].Get();
  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  if (gemm->Inputs().size() == 3) {
    auto* bias = gemm->Inputs()[2].Get();
    if (bias->Dimensions().size() != 1 ||
        bias->Dimensions()[0] != output->Dimensions()[1]) {
      exception_state.ThrowDOMException(DOMExceptionCode::kNotSupportedError,
                                        "the bias shape is invalid");
      return false;
    }
    DCHECK(tensors_map.find(bias) != tensors_map.end());
    bias_id = tensors_map.at(bias);
  }
  if (fabs(options->alpha() - 1.0f) > std::numeric_limits<float>::epsilon()) {
    exception_state.ThrowDOMException(DOMExceptionCode::kNotSupportedError,
                                      "alpha is not supported");
    return false;
  }
  if (fabs(options->beta() - 1.0f) > std::numeric_limits<float>::epsilon()) {
    exception_state.ThrowDOMException(DOMExceptionCode::kNotSupportedError,
                                      "beta is not supported");
    return false;
  }
  if (options->aTranspose()) {
    exception_state.ThrowDOMException(DOMExceptionCode::kNotSupportedError,
                                      "aTranspose is not supported");
    return false;
  }
  uint32_t flags = 0;
  if (!options->bTranspose()) {
    flags = XNN_FLAG_TRANSPOSE_WEIGHTS;
  }
  if (tensors_map.find(output) == tensors_map.end()) {
    if (!DefineTensor(subgraph, tensors_map, output, exception_state)) {
      return false;
    }
  }
  uint32_t output_id = tensors_map.at(output);
  const float output_min = -std::numeric_limits<float>::infinity();
  const float output_max = +std::numeric_limits<float>::infinity();
  xnn_status status;
  if ((status = xnn_define_fully_connected(
           subgraph, output_min, output_max, input_id, filter_id, bias_id,
           output_id, flags)) != xnn_status_success) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kOperationError,
        "failed to define gemm: " + XnnStatusToString(status));
    return false;
  }
  return true;
}

bool MLGraphXnnpack::DefinePool2d(
    xnn_subgraph_t subgraph,
    HashMap<Member<const MLOperand>, uint32_t>& tensors_map,
    const MLOperator* pool2d,
    const MLPool2dOptions* options,
    ExceptionState& exception_state) {
  auto* input = pool2d->Inputs()[0].Get();
  DCHECK(tensors_map.find(input) != tensors_map.end());
  uint32_t input_id = tensors_map.at(input);
  auto* output = pool2d->Outputs()[0].Get();
  if (tensors_map.find(output) == tensors_map.end()) {
    if (!DefineTensor(subgraph, tensors_map, output, exception_state)) {
      return false;
    }
  }
  uint32_t output_id = tensors_map.at(output);

  uint32_t stride_height = options->hasStrides() ? options->strides()[0] : 1;
  uint32_t stride_width = options->hasStrides() ? options->strides()[1] : 1;
  uint32_t dilation_height =
      options->hasDilations() ? options->dilations()[0] : 1;
  uint32_t dilation_width =
      options->hasDilations() ? options->dilations()[1] : 1;
  size_t input_height, input_width;
  uint32_t filter_height, filter_width;
  bool global_pooling = false;
  if (options->layout().AsEnum() == V8MLInputOperandLayout::Enum::kNhwc) {
    input_height = input->Dimensions()[1];
    input_width = input->Dimensions()[2];
    if (options->hasWindowDimensions()) {
      filter_height = options->windowDimensions()[0];
      filter_width = options->windowDimensions()[1];
    } else {
      global_pooling = true;
    }
  } else {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kNotSupportedError,
        "only input layout nhwc for pool2d is supported");
    return false;
  }

  size_t output_height, output_width;
  uint32_t pad_top, pad_bottom, pad_left, pad_right;
  if (options->autoPad().AsEnum() == V8MLAutoPad::Enum::kExplicit) {
    // WebNN padding: [beginning_height, ending_height, beginning_width,
    // ending_width]
    pad_top = options->hasPadding() ? options->padding()[0] : 0;
    pad_bottom = options->hasPadding() ? options->padding()[1] : 0;
    pad_left = options->hasPadding() ? options->padding()[2] : 0;
    pad_right = options->hasPadding() ? options->padding()[3] : 0;
  } else {
    output_height = ceil(input_height / stride_height);
    output_width = ceil(input_width / stride_width);
    uint32_t pad_along_height = static_cast<uint32_t>(std::max(
        size_t(0),
        (output_height - 1) * stride_height + filter_height - input_height));
    uint32_t pad_along_width = static_cast<uint32_t>(std::max(
        size_t(0),
        (output_width - 1) * stride_width + filter_width - input_width));
    if (options->autoPad().AsEnum() == V8MLAutoPad::Enum::kSameUpper) {
      pad_top = floor(pad_along_height / 2);
      pad_bottom = pad_along_height - pad_top;
      pad_left = floor(pad_along_width / 2);
      pad_right = pad_along_width - pad_left;
    } else {
      pad_bottom = floor(pad_along_height / 2);
      pad_top = pad_along_height - pad_bottom;
      pad_right = floor(pad_along_width / 2);
      pad_left = pad_along_width - pad_right;
    }
  }

  float output_min = -std::numeric_limits<float>::infinity();
  float output_max = +std::numeric_limits<float>::infinity();
  const uint32_t flags = 0;
  xnn_status status = xnn_status_success;
  switch (pool2d->Kind()) {
    case MLOperator::OpKind::kAveragePool2d: {
      if (dilation_height != 1 || dilation_width != 1) {
        exception_state.ThrowDOMException(
            DOMExceptionCode::kNotSupportedError,
            "dilation for averagePool2d is not supported");
        return false;
      }
      if (global_pooling) {
        status = xnn_define_global_average_pooling_2d(
            subgraph, output_min, output_max, input_id, output_id, flags);
      } else {
        status = xnn_define_average_pooling_2d(
            subgraph, pad_top, pad_right, pad_bottom, pad_left, filter_height,
            filter_width, stride_height, stride_width, output_min, output_max,
            input_id, output_id, flags);
      }
      break;
    }
    default:
      NOTREACHED();
  }
  if (status != xnn_status_success) {
    exception_state.ThrowDOMException(DOMExceptionCode::kOperationError,
                                      "failed to define " +
                                          OpKindToString(pool2d->Kind()) +
                                          ": " + XnnStatusToString(status));
    return false;
  }
  return true;
}

bool MLGraphXnnpack::DefineReshape(
    xnn_subgraph_t subgraph,
    HashMap<Member<const MLOperand>, uint32_t>& tensors_map,
    const MLOperator* reshape,
    ExceptionState& exception_state) {
  auto* input = reshape->Inputs()[0].Get();
  DCHECK(tensors_map.find(input) != tensors_map.end());
  uint32_t input_id = tensors_map.at(input);
  auto* output = reshape->Outputs()[0].Get();
  Vector<size_t> new_sizes;
  for (auto& d : output->Dimensions()) {
    new_sizes.push_back(static_cast<size_t>(d));
  }
  if (new_sizes.size() > XNN_MAX_TENSOR_DIMS) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kDataError,
        "The rank of new shape is not supported.");
    return false;
  }
  if (tensors_map.find(output) == tensors_map.end()) {
    if (!DefineTensor(subgraph, tensors_map, output, exception_state)) {
      return false;
    }
  }
  uint32_t output_id = tensors_map.at(output);
  xnn_status status = xnn_status_success;
  if ((status = xnn_define_static_reshape(subgraph, new_sizes.size(),
                                          new_sizes.data(), input_id, output_id,
                                          0)) != xnn_status_success) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kOperationError,
        "failed to define reshape: " + XnnStatusToString(status));
    return false;
  }
  return true;
}

bool MLGraphXnnpack::DefineUnary(
    xnn_subgraph_t subgraph,
    HashMap<Member<const MLOperand>, uint32_t>& tensors_map,
    const MLOperator* unary,
    ExceptionState& exception_state) {
  auto* input = unary->Inputs()[0].Get();
  DCHECK(tensors_map.find(input) != tensors_map.end());
  uint32_t input_id = tensors_map.at(input);
  auto* output = unary->Outputs()[0].Get();
  if (tensors_map.find(output) == tensors_map.end()) {
    if (!DefineTensor(subgraph, tensors_map, output, exception_state)) {
      return false;
    }
  }
  uint32_t output_id = tensors_map.at(output);
  xnn_status status = xnn_status_success;
  switch (unary->Kind()) {
    case MLOperator::OpKind::kRelu: {
      status = xnn_define_clamp(subgraph, 0.0f,
                                std::numeric_limits<float>::infinity(),
                                input_id, output_id, 0);
      break;
    }
    case MLOperator::OpKind::kSoftmax: {
      status = xnn_define_softmax(subgraph, input_id, output_id, 0);
      break;
    }
    default:
      NOTREACHED();
  }
  if (status != xnn_status_success) {
    exception_state.ThrowDOMException(DOMExceptionCode::kOperationError,
                                      "failed to define " +
                                          OpKindToString(unary->Kind()) + ": " +
                                          XnnStatusToString(status));
    return false;
  }
  return true;
}

void MLGraphXnnpack::ComputeImpl(const MLNamedArrayInputs& inputs,
                                 const MLNamedArrayOutputs& outputs,
                                 ExceptionState& exception_state) {
  Vector<xnn_external_value> external_values;
  if (inputs.size() != inputs_info_.size()) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "The number of inputs is invalid");
    return;
  }
  if (outputs.size() != outputs_info_.size()) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "The number of outputs is invalid");
    return;
  }
  for (const auto& input : inputs) {
    auto iter = inputs_info_.find(input.first);
    if (iter == inputs_info_.end()) {
      exception_state.ThrowDOMException(
          DOMExceptionCode::kDataError,
          "There is unknown input: " + input.first);
      return;
    }
    xnn_external_value value = {0};
    value.id = iter->value.id;
    DOMArrayBufferView* array_buffer_view = nullptr;
    if (input.second->IsArrayBufferViewAllowShared()) {
      array_buffer_view = input.second->GetAsArrayBufferViewAllowShared().Get();
    } else if (input.second->IsMLTensor()) {
      auto* ml_tensor = input.second->GetAsMLTensor();
      array_buffer_view = ml_tensor->data().Get();
    }
    DCHECK(array_buffer_view != nullptr);
    if (array_buffer_view->byteLength() < iter->value.byte_length) {
      exception_state.ThrowDOMException(
          DOMExceptionCode::kDataError,
          "The input (" + input.first + ") buffer length is invalid.");
      return;
    }
    value.data = array_buffer_view->BaseAddressMaybeShared();
    external_values.push_back(value);
  }
  for (const auto& output : outputs) {
    auto iter = outputs_info_.find(output.first);
    if (iter == outputs_info_.end()) {
      exception_state.ThrowDOMException(
          DOMExceptionCode::kDataError,
          "There is unknown output: " + output.first);
      return;
    }
    xnn_external_value value = {0};
    value.id = iter->value.id;
    if (output.second->IsArrayBufferViewAllowShared()) {
      DOMArrayBufferView* array_buffer_view =
          output.second->GetAsArrayBufferViewAllowShared().Get();
      if (array_buffer_view->byteLength() < iter->value.byte_length) {
        exception_state.ThrowDOMException(
            DOMExceptionCode::kDataError,
            "The output (" + output.first + ") buffer length is invalid.");
        return;
      }
      value.data = array_buffer_view->BaseAddressMaybeShared();
    } else if (output.second->IsArrayBufferAllowShared()) {
      DOMArrayBufferBase* array_buffer =
          output.second->GetAsArrayBufferAllowShared();
      if (array_buffer->ByteLength() < iter->value.byte_length) {
        exception_state.ThrowDOMException(
            DOMExceptionCode::kDataError,
            "The output (" + output.first + ") buffer length is invalid.");
        return;
      }
      value.data = array_buffer->DataMaybeShared();
    }
    DCHECK(value.data);
    external_values.push_back(value);
  }
  if (xnn_setup_runtime(runtime_, external_values.size(),
                        external_values.data()) != xnn_status_success) {
    exception_state.ThrowDOMException(DOMExceptionCode::kOperationError,
                                      "failed to setup runtime");
    return;
  }
  if (xnn_invoke_runtime(runtime_) != xnn_status_success) {
    exception_state.ThrowDOMException(DOMExceptionCode::kOperationError,
                                      "failed to invoke runtime");
    return;
  }
}

}  // namespace blink
