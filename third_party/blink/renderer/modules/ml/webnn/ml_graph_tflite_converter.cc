// Copyright 2023 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_tflite_converter.h"

#include <numeric>

#include "third_party/abseil-cpp/absl/types/optional.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_clamp_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_conv_2d_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_gemm_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_operand_descriptor.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_pool_2d_options.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_activation.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_cros.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operand.h"
#include "third_party/blink/renderer/platform/bindings/exception_code.h"
#include "third_party/blink/renderer/platform/bindings/exception_state.h"

namespace blink {

namespace {

// The version number of the Schema. Ideally all changes will be backward
// compatible. If that ever changes, we must ensure that version is the first
// entry in the new tflite root so that we can see that version is not 1.
#define TFLITE_SCHEMA_VERSION (3)

// Maps MLOperand to its index of `tflite::Tensor` array.
using OperandToIndexMap = HeapHashMap<Member<const MLOperand>, int32_t>;
using OperatorCodeOffset = flatbuffers::Offset<tflite::OperatorCode>;
using OperatorOffset = flatbuffers::Offset<tflite::Operator>;
using BufferOffset = flatbuffers::Offset<tflite::Buffer>;

int32_t GetOperatorInputIndex(const MLOperator* op,
                              const OperandToIndexMap& operand_to_index_map,
                              wtf_size_t index = 0) {
  CHECK_NE(op, nullptr);
  CHECK_LE(index, op->Inputs().size());
  const auto* input = op->Inputs()[index].Get();
  return operand_to_index_map.at(input);
}

int32_t GetOperatorOutputIndex(const MLOperator* op,
                               const OperandToIndexMap& operand_to_index_map,
                               wtf_size_t index = 0) {
  CHECK_NE(op, nullptr);
  CHECK_LE(index, op->Outputs().size());
  const auto* output = op->Outputs()[index].Get();
  return operand_to_index_map.at(output);
}

Vector<int32_t> ConvertDimensions(const Vector<uint32_t>& dimensions) {
  Vector<int32_t> new_dims;
  new_dims.reserve(dimensions.size());
  for (auto dim : dimensions) {
    new_dims.push_back(base::checked_cast<int32_t>(dim));
  }
  return new_dims;
}

tflite::TensorType BlinkOperandTypeToTFLite(V8MLOperandType::Enum type) {
  switch (type) {
    case V8MLOperandType::Enum::kFloat32:
      return tflite::TensorType_FLOAT32;
    case V8MLOperandType::Enum::kFloat16:
      return tflite::TensorType_FLOAT16;
    case V8MLOperandType::Enum::kInt32:
      return tflite::TensorType_INT32;
    case V8MLOperandType::Enum::kUint32:
      return tflite::TensorType_UINT32;
    case V8MLOperandType::Enum::kInt8:
      return tflite::TensorType_INT8;
    case V8MLOperandType::Enum::kUint8:
      return tflite::TensorType_UINT8;
  }
}

// Helper to get padding sizes for tflite convolution 2d or pooling 2d.
template <typename OptionsType>
base::expected<tflite::Padding, String> GetTfLitePaddingMode(
    const OptionsType* options,
    uint32_t input_height,
    uint32_t input_width,
    uint32_t filter_height,
    uint32_t filter_width,
    uint32_t stride_height,
    uint32_t stride_width,
    uint32_t dilation_height,
    uint32_t dilation_width) {
  tflite::Padding padding_mode = tflite::Padding_VALID;
  Vector<uint32_t> explicit_pads;
  switch (options->autoPad().AsEnum()) {
    case V8MLAutoPad::Enum::kExplicit: {
      // Set the XNNPACK padding from WebNN explicit padding that is in
      // [beginning_height, ending_height, beginning_width, ending_width],
      // default to 0.
      const Vector<uint32_t> default_pads({0, 0, 0, 0});
      explicit_pads.push_back(options->getPaddingOr(default_pads)[0]);
      explicit_pads.push_back(options->getPaddingOr(default_pads)[1]);
      explicit_pads.push_back(options->getPaddingOr(default_pads)[2]);
      explicit_pads.push_back(options->getPaddingOr(default_pads)[3]);
      if (explicit_pads == default_pads) {
        padding_mode = tflite::Padding_VALID;
      } else {
        webnn::AutoPad auto_pad =
            BlinkAutoPadToComponent(options->autoPad().AsEnum());
        // Calculate the tflite kSameUpper padding based on WebNN auto padding
        // mode and sizes.
        Vector<uint32_t> same_pads;
        auto padding_sizes_height =
            webnn::CalculateConv2dPadding(auto_pad, input_height, filter_height,
                                          stride_height, dilation_height);
        CHECK(padding_sizes_height);
        same_pads.push_back(padding_sizes_height.value().begin);
        same_pads.push_back(padding_sizes_height.value().end);
        auto padding_sizes_width = webnn::CalculateConv2dPadding(
            auto_pad, input_width, filter_width, stride_width, dilation_width);
        same_pads.push_back(padding_sizes_width.value().begin);
        same_pads.push_back(padding_sizes_width.value().end);

        if (explicit_pads == same_pads) {
          padding_mode = tflite::Padding_SAME;
        } else {
          return base::unexpected(
              "The explicit padding are not supported in tflite");
        }
      }
      break;
    }
    case V8MLAutoPad::Enum::kSameUpper: {
      padding_mode = tflite::Padding_SAME;
      break;
    }
    case V8MLAutoPad::Enum::kSameLower:
      return base::unexpected("Same lower is not supported in tflite");
  }
  return padding_mode;
}

base::expected<uint32_t, String> SerializeOperatorCode(
    tflite::BuiltinOperator code,
    flatbuffers::FlatBufferBuilder& builder,
    Vector<OperatorCodeOffset>& operator_codes) {
  auto operator_code = tflite::CreateOperatorCode(builder, code);
  if (operator_code.IsNull()) {
    return base::unexpected("Failed to create operator code.");
  }
  operator_codes.push_back(operator_code);
  // Create `tflite::Operator` with the tensor index of inputs and outputs
  // operand. The type of operation is determined by an index into the list of
  // valid OperatorCodes.
  return base::checked_cast<uint32_t>(operator_codes.size()) - 1;
}

uint32_t SerializeEmptyBuffer(
    uint32_t output_channels,
    flatbuffers::FlatBufferBuilder& builder,
    Vector<BufferOffset>& buffers,
    Vector<flatbuffers::Offset<tflite::Tensor>>& tensors) {
  Vector<int32_t> dimensions = {base::checked_cast<int32_t>(output_channels)};
  size_t buffer_size = std::accumulate(dimensions.begin(), dimensions.end(),
                                       size_t(1), std::multiplies<int32_t>());
  std::vector<float> empty_data(buffer_size);
  // Create `tflite::Buffer` with raw data buffers for constant operand.
  buffers.push_back(tflite::CreateBuffer(
      builder,
      builder.CreateVector(reinterpret_cast<const uint8_t*>(empty_data.data()),
                           empty_data.size() * sizeof(float))));
  // The index of buffer is referenced by tensors.
  uint32_t buffer_index = static_cast<uint32_t>(buffers.size()) - 1;
  // Create `Tensor` with operand shape, the index of buffer and the name.
  // The tensor shape is NHWC ([batch size, height, width, number of channels]).
  // The buffer index is 0 for intermediate and output operand because there is
  // no data buffer associated. Input and output operand has the name to
  // identify array buffer, there are no name for intermediate operand.
  tensors.emplace_back(
      tflite::CreateTensor(builder, builder.CreateVector<int32_t>(dimensions),
                           tflite::TensorType_FLOAT32, buffer_index));
  return base::checked_cast<uint32_t>(tensors.size() - 1);
}

base::expected<OperatorOffset, String> SerializeConv2d(
    const OperandToIndexMap& operand_to_index_map,
    const MLOperator* conv2d,
    flatbuffers::FlatBufferBuilder& builder,
    Vector<OperatorCodeOffset>& operator_codes,
    Vector<BufferOffset>& buffers,
    Vector<flatbuffers::Offset<tflite::Tensor>>& tensors) {
  const int32_t input_index =
      GetOperatorInputIndex(conv2d, operand_to_index_map, 0);
  const int32_t filter_index =
      GetOperatorInputIndex(conv2d, operand_to_index_map, 1);
  const int32_t output_index =
      GetOperatorOutputIndex(conv2d, operand_to_index_map);

  const MLConv2dOptions* options =
      static_cast<const MLConv2dOptions*>(conv2d->Options());

  // Set input and filter sizes of conv2d.
  uint32_t input_height, input_width;
  uint32_t filter_height, filter_width;
  uint32_t input_channels, output_channels;
  bool depthwise = false;
  if (options->inputLayout().AsEnum() == V8MLInputOperandLayout::Enum::kNhwc) {
    const auto* input = conv2d->Inputs()[0].Get();
    CHECK(input);
    input_height = input->Dimensions()[1];
    input_width = input->Dimensions()[2];
    input_channels = input->Dimensions()[3];
    const auto* output = conv2d->Outputs()[0].Get();
    CHECK(output);
    output_channels = output->Dimensions()[3];
    const uint32_t groups = base::checked_cast<uint32_t>(options->groups());

    // According to WebNN conv2d spec:
    // https://www.w3.org/TR/webnn/#api-mlgraphbuilder-conv2d, A depthwise
    // conv2d operation is a variant of grouped convolution where the
    // options.groups = input_channels = output_channels.
    depthwise =
        (groups == input_channels && groups == output_channels && groups != 1);
    if (!depthwise) {
      // For regular conv2d, TF-Lite expects weights layout in ohwi that is
      // [groups * group_output_channels, kernel_height, kernel_width,
      //  group_input_channels].
      // TODO(crbug.com/1273291): support other layouts by transposing the
      // filter operand.
      if (options->filterLayout().AsEnum() !=
          V8MLConv2dFilterOperandLayout::Enum::kOhwi) {
        return base::unexpected(
            String::Format("The filter layout %s is not supported .",
                           options->filterLayout().AsCStr()));
      }
    } else {
      // For depthwise conv2d, TF-Lite expects weights layout in ihwo that is
      // [1, kernel_height, kernel_width, input_channels * depth_multiplier].
      // TODO(crbug.com/1273291): support other layouts by transposing the
      // filter operand.
      if (options->filterLayout().AsEnum() !=
          V8MLConv2dFilterOperandLayout::Enum::kIhwo) {
        return base::unexpected(
            String::Format("The filter layout %s is not supported.",
                           options->filterLayout().AsCStr()));
      }
    }
    const auto* filter = conv2d->Inputs()[1].Get();
    CHECK(filter);
    filter_height = filter->Dimensions()[1];
    filter_width = filter->Dimensions()[2];
  } else {
    // TODO(crbug.com/1273291): support other layouts by transposing the input
    // operand.
    return base::unexpected(
        String::Format("The input layout %s is not supported.",
                       options->inputLayout().AsCStr()));
  }

  tflite::ActivationFunctionType activation =
      tflite::ActivationFunctionType_NONE;
  if (options->hasActivation()) {
    switch (options->activation()->Operator()->Kind()) {
      case MLOperator::OperatorKind::kClamp: {
        const MLClampOptions* clamp_options =
            static_cast<const MLClampOptions*>(
                options->activation()->Operator()->Options());
        CHECK(clamp_options);
        uint32_t min = clamp_options->getMinValueOr(
            -std::numeric_limits<float>::infinity());
        uint32_t max = clamp_options->getMaxValueOr(
            +std::numeric_limits<float>::infinity());
        if (min == 0 && max == 6) {
          activation = tflite::ActivationFunctionType_RELU6;
        } else {
          return base::unexpected("TODO");
        }
        break;
      }
      case MLOperator::OperatorKind::kRelu:
        activation = tflite::ActivationFunctionType_RELU;
        break;
      default:
        return base::unexpected(
            "Only clamp and relu fused operator are supported by conv2d.");
    }
  }

  // Set strides of conv2d, default to 1.
  const Vector<uint32_t> default_strides({1, 1});
  const int32_t stride_height =
      base::checked_cast<int32_t>(options->getStridesOr(default_strides)[0]);
  const int32_t stride_width =
      base::checked_cast<int32_t>(options->getStridesOr(default_strides)[1]);

  // Set dilations of conv2d, default to 1.
  const Vector<uint32_t> default_dilations({1, 1});
  const int32_t dilation_height = base::checked_cast<int32_t>(
      options->getDilationsOr(default_dilations)[0]);
  const int32_t dilation_width = base::checked_cast<int32_t>(
      options->getDilationsOr(default_dilations)[1]);

  // Set tflite padding mode.
  auto padding_mode = GetTfLitePaddingMode(
      options, input_height, input_width, filter_height, filter_width,
      stride_height, stride_width, dilation_height, dilation_width);
  if (!padding_mode.has_value()) {
    return base::unexpected(padding_mode.error());
  }

  tflite::BuiltinOperator binary_op;
  tflite::BuiltinOptions builtin_options_type = tflite::BuiltinOptions_NONE;
  flatbuffers::Offset<void> builtin_options = 0;
  if (depthwise) {
    const uint32_t depth_multiplier = 1;
    binary_op = tflite::BuiltinOperator_DEPTHWISE_CONV_2D;
    builtin_options =
        tflite::CreateDepthwiseConv2DOptions(
            builder, padding_mode.value(), stride_width, stride_height,
            depth_multiplier, activation, dilation_width, dilation_height)
            .Union();
    builtin_options_type = tflite::BuiltinOptions_DepthwiseConv2DOptions;
  } else {
    binary_op = tflite::BuiltinOperator_CONV_2D;
    builtin_options =
        tflite::CreateConv2DOptions(builder, padding_mode.value(), stride_width,
                                    stride_height, activation, dilation_width,
                                    dilation_height)
            .Union();
    builtin_options_type = tflite::BuiltinOptions_Conv2DOptions;
  }

  // Create `tflite::Operator` with the tensor index of inputs and outputs
  // operand. The type of operation is determined by an index into the list of
  // valid OperatorCodes.
  auto opcode_index = SerializeOperatorCode(binary_op, builder, operator_codes);
  if (!opcode_index.has_value()) {
    return base::unexpected(opcode_index.error());
  }
  // If there is no bias operand, serialize a empty buffer with the size of
  // output channel.
  const int32_t bias_index =
      conv2d->Inputs().size() == 3
          ? GetOperatorInputIndex(conv2d, operand_to_index_map, 2)
          : SerializeEmptyBuffer(output_channels, builder, buffers, tensors);
  const std::vector<int32_t> op_inputs = {input_index, filter_index,
                                          bias_index};
  const std::vector<int32_t> op_outputs = {output_index};

  auto conv2d_operator = tflite::CreateOperator(
      builder, opcode_index.value(), builder.CreateVector<int32_t>(op_inputs),
      builder.CreateVector<int32_t>(op_outputs), builtin_options_type,
      builtin_options);
  if (conv2d_operator.IsNull()) {
    return base::unexpected("Failed to create conv2d operator.");
  }
  return conv2d_operator;
}

base::expected<OperatorOffset, String> SerializePool2d(
    const OperandToIndexMap& operand_to_index_map,
    const MLOperator* pool2d,
    flatbuffers::FlatBufferBuilder& builder,
    Vector<OperatorCodeOffset>& operator_codes) {
  const int32_t input_index =
      GetOperatorInputIndex(pool2d, operand_to_index_map);
  const int32_t output_index =
      GetOperatorOutputIndex(pool2d, operand_to_index_map);

  // Set strides of TF-Lite pooling 2d, default to 1.
  const MLPool2dOptions* options =
      static_cast<const MLPool2dOptions*>(pool2d->Options());
  const Vector<uint32_t> default_strides({1, 1});
  const int32_t stride_height =
      base::checked_cast<int32_t>(options->getStridesOr(default_strides)[0]);
  const int32_t stride_width =
      base::checked_cast<int32_t>(options->getStridesOr(default_strides)[1]);

  // Set dilations of TF-Lite pooling 2d, default to 1.
  const Vector<uint32_t> default_dilations({1, 1});
  const int32_t dilation_height = base::checked_cast<int32_t>(
      options->getDilationsOr(default_dilations)[0]);
  const int32_t dilation_width = base::checked_cast<int32_t>(
      options->getDilationsOr(default_dilations)[1]);
  if (dilation_height != 1 || dilation_width != 1) {
    return base::unexpected(
        "Pool2d in TF-Lite schema doesn't support dilations.");
  }

  // Set window sizes of TF-Lite pooling 2d.
  uint32_t input_height, input_width;
  uint32_t filter_height, filter_width;
  switch (options->layout().AsEnum()) {
    case V8MLInputOperandLayout::Enum::kNhwc: {
      const auto* input = pool2d->Inputs()[0].Get();
      CHECK(input);
      input_height = input->Dimensions()[1];
      input_width = input->Dimensions()[2];
      if (options->hasWindowDimensions()) {
        filter_height = options->windowDimensions()[0];
        filter_width = options->windowDimensions()[1];
      } else {
        // According to WebNN pool2d spec:
        // https://www.w3.org/TR/webnn/#api-mlgraphbuilder-pool2d, if the window
        // dimensions are not present, the window dimensions are assumed to be
        // the height and width dimensions of the input shape that could be
        // mapped to the global pooling operation.
        filter_height = input_height;
        filter_width = input_width;
      }
      break;
    }
    case V8MLInputOperandLayout::Enum::kNchw: {
      // TODO(crbug.com/1273291): support nchw input layout by transposing the
      // input tensor.
      return base::unexpected("The nchw input layout is not supported.");
    }
  }

  tflite::BuiltinOperator pool_op = tflite::BuiltinOperator_AVERAGE_POOL_2D;
  switch (pool2d->Kind()) {
    case MLOperator::OperatorKind::kAveragePool2d: {
      pool_op = tflite::BuiltinOperator_AVERAGE_POOL_2D;
      break;
    }
    case MLOperator::OperatorKind::kMaxPool2d: {
      pool_op = tflite::BuiltinOperator_MAX_POOL_2D;
      break;
    }
    default:
      // Only average and max pool2d are supported by this method.
      NOTREACHED();
  }

  // Set tflite padding mode.
  auto padding_mode = GetTfLitePaddingMode(
      options, input_height, input_width, filter_height, filter_width,
      stride_height, stride_width, dilation_height, dilation_width);
  if (!padding_mode.has_value()) {
    return base::unexpected(padding_mode.error());
  }
  auto pool_2d_options = CreatePool2DOptions(
      builder, tflite::Padding_VALID, stride_width, stride_height, filter_width,
      filter_height, tflite::ActivationFunctionType_NONE);
  if (pool_2d_options.IsNull()) {
    return base::unexpected("Failed to create pool2d options.");
  }

  // Create `tflite::Operator` with the tensor index of inputs and outputs
  // operand. The type of operation is determined by an index into the list of
  // valid OperatorCodes, while the specifics of each operations is configured
  // using builtin_options or custom_options.
  auto opcode_index = SerializeOperatorCode(pool_op, builder, operator_codes);
  if (!opcode_index.has_value()) {
    return base::unexpected(opcode_index.error());
  }
  const std::vector<int32_t> op_inputs = {input_index};
  const std::vector<int32_t> op_outputs = {output_index};

  auto pool2d_operator = tflite::CreateOperator(
      builder, opcode_index.value(), builder.CreateVector<int32_t>(op_inputs),
      builder.CreateVector<int32_t>(op_outputs),
      tflite::BuiltinOptions_Pool2DOptions, pool_2d_options.Union());
  if (pool2d_operator.IsNull()) {
    return base::unexpected("Failed to create pool2d operator.");
  }
  return pool2d_operator;
}

base::expected<OperatorOffset, String> SerializeElementWiseBinary(
    const OperandToIndexMap& operand_to_index_map,
    const MLOperator* binary,
    flatbuffers::FlatBufferBuilder& builder,
    Vector<OperatorCodeOffset>& operator_codes) {
  const int32_t lhs_index =
      GetOperatorInputIndex(binary, operand_to_index_map, 0);
  const int32_t rhs_index =
      GetOperatorInputIndex(binary, operand_to_index_map, 1);
  const int32_t output_index =
      GetOperatorOutputIndex(binary, operand_to_index_map);
  tflite::BuiltinOperator binary_op;
  switch (binary->Kind()) {
    case MLOperator::OperatorKind::kAdd:
      binary_op = tflite::BuiltinOperator_ADD;
      break;
    case MLOperator::OperatorKind::kSub:
      binary_op = tflite::BuiltinOperator_SUB;
      break;
    case MLOperator::OperatorKind::kMul:
      binary_op = tflite::BuiltinOperator_MUL;
      break;
    case MLOperator::OperatorKind::kDiv:
      binary_op = tflite::BuiltinOperator_DIV;
      break;
    case MLOperator::OperatorKind::kMin:
      binary_op = tflite::BuiltinOperator_MINIMUM;
      break;
    case MLOperator::OperatorKind::kMax:
      binary_op = tflite::BuiltinOperator_MAXIMUM;
      break;
    case MLOperator::OperatorKind::kPow:
      binary_op = tflite::BuiltinOperator_POW;
      break;
    default:
      NOTREACHED();
      return OperatorOffset();
  }

  // TF-Lite support activation in elementwise binary operations, but WebNN Spec
  // doesn't define the feature, so the options of elementwise binary doesn't
  // need to be configured.
  tflite::BuiltinOptions builtin_options_type = tflite::BuiltinOptions_NONE;
  flatbuffers::Offset<void> builtin_options = 0;

  // Create `tflite::Operator` with the tensor index of inputs and outputs
  // operand. The type of operation is determined by an index into the list of
  // valid OperatorCodes, while the specifics of each operations is configured
  // using builtin_options or custom_options.
  auto opcode_index = SerializeOperatorCode(binary_op, builder, operator_codes);
  if (!opcode_index.has_value()) {
    return base::unexpected(opcode_index.error());
  }
  const std::vector<int32_t> op_inputs = {lhs_index, rhs_index};
  const std::vector<int32_t> op_outputs = {output_index};

  auto element_wise_binary = tflite::CreateOperator(
      builder, opcode_index.value(), builder.CreateVector<int32_t>(op_inputs),
      builder.CreateVector<int32_t>(op_outputs), builtin_options_type,
      builtin_options);
  if (element_wise_binary.IsNull()) {
    return base::unexpected("Failed to create element wise binary operator.");
  }
  return element_wise_binary;
}

base::expected<OperatorOffset, String> SerializeRelu(
    const OperandToIndexMap& operand_to_index_map,
    const MLOperator* relu,
    flatbuffers::FlatBufferBuilder& builder,
    Vector<OperatorCodeOffset>& operator_codes) {
  const int32_t input_index = GetOperatorInputIndex(relu, operand_to_index_map);
  const int32_t output_index =
      GetOperatorOutputIndex(relu, operand_to_index_map);

  // Create `tflite::Operator` with the tensor index of inputs and outputs
  // operand. The type of operation is determined by an index into the list of
  // valid OperatorCodes.
  auto opcode_index = SerializeOperatorCode(tflite::BuiltinOperator_RELU,
                                            builder, operator_codes);
  if (!opcode_index.has_value()) {
    return base::unexpected(opcode_index.error());
  }
  const std::vector<int32_t> op_inputs = {input_index};
  const std::vector<int32_t> op_outputs = {output_index};

  auto relu_operator = tflite::CreateOperator(
      builder, opcode_index.value(), builder.CreateVector<int32_t>(op_inputs),
      builder.CreateVector<int32_t>(op_outputs));
  if (relu_operator.IsNull()) {
    return base::unexpected("Failed to create relu operator.");
  }
  return relu_operator;
}

base::expected<OperatorOffset, String> SerializeReshape(
    const OperandToIndexMap& operand_to_index_map,
    const MLOperator* reshape,
    flatbuffers::FlatBufferBuilder& builder,
    Vector<OperatorCodeOffset>& operator_codes) {
  const int32_t input_index =
      GetOperatorInputIndex(reshape, operand_to_index_map);
  const int32_t output_index =
      GetOperatorOutputIndex(reshape, operand_to_index_map);

  auto* output = reshape->Outputs()[0].Get();
  CHECK(output);
  auto reshape_options = tflite::CreateReshapeOptions(
      builder,
      builder.CreateVector<int32_t>(ConvertDimensions(output->Dimensions())));
  if (reshape_options.IsNull()) {
    return base::unexpected("Failed to create reshape options.");
  }

  // Create `tflite::OperatorCode` for Softmax operation.
  // Create `tflite::Operator` with the tensor index of inputs and outputs
  // operand. The type of operation is determined by an index into the list of
  // valid OperatorCodes.
  auto opcode_index = SerializeOperatorCode(tflite::BuiltinOperator_RESHAPE,
                                            builder, operator_codes);
  if (!opcode_index.has_value()) {
    return base::unexpected(opcode_index.error());
  }
  const std::vector<int32_t> op_inputs = {input_index};
  const std::vector<int32_t> op_outputs = {output_index};

  auto reshape_operator = tflite::CreateOperator(
      builder, opcode_index.value(), builder.CreateVector<int32_t>(op_inputs),
      builder.CreateVector<int32_t>(op_outputs),
      tflite::BuiltinOptions_ReshapeOptions, reshape_options.Union());
  if (reshape_operator.IsNull()) {
    return base::unexpected("Failed to create reshape operator.");
  }
  return reshape_operator;
}

base::expected<OperatorOffset, String> SerializeSoftmax(
    const OperandToIndexMap& operand_to_index_map,
    const MLOperator* softmax,
    flatbuffers::FlatBufferBuilder& builder,
    Vector<OperatorCodeOffset>& operator_codes) {
  const int32_t input_index =
      GetOperatorInputIndex(softmax, operand_to_index_map);
  const int32_t output_index =
      GetOperatorOutputIndex(softmax, operand_to_index_map);

  auto softmax_options = tflite::CreateSoftmaxOptions(builder, 1.0);
  if (softmax_options.IsNull()) {
    return base::unexpected("Failed to create softmax options.");
  }

  // Create `tflite::OperatorCode` for Softmax operation.

  // Create `tflite::Operator` with the tensor index of inputs and outputs
  // operand. The type of operation is determined by an index into the list of
  // valid OperatorCodes.
  auto opcode_index = SerializeOperatorCode(tflite::BuiltinOperator_SOFTMAX,
                                            builder, operator_codes);
  if (!opcode_index.has_value()) {
    return base::unexpected(opcode_index.error());
  }
  const std::vector<int32_t> op_inputs = {input_index};
  const std::vector<int32_t> op_outputs = {output_index};

  auto softmax_operator = tflite::CreateOperator(
      builder, opcode_index.value(), builder.CreateVector<int32_t>(op_inputs),
      builder.CreateVector<int32_t>(op_outputs),
      tflite::BuiltinOptions_SoftmaxOptions, softmax_options.Union());
  if (softmax_operator.IsNull()) {
    return base::unexpected("Failed to create softmax operator.");
  }
  return softmax_operator;
}

}  // namespace

MLGraphTfLiteConverter::MLGraphTfLiteConverter() {
  // It is required that the first entry in the buffers of model is always an
  // empty buffer. This is so that the default buffer index of zero in Tensor
  // will always refer to a valid empty buffer.
  buffers_.push_back(tflite::CreateBuffer(builder_, builder_.CreateVector({})));
}

MLGraphTfLiteConverter::~MLGraphTfLiteConverter() = default;

base::expected<uint32_t, String> MLGraphTfLiteConverter::SerializeBuffer(
    const MLOperand* constant) {
  // There are two steps to implement the `BuildBuffer` function:
  // 1, Create `tflite::Buffer` with array buffer view.
  // 2, Create `tflite::Tensor` with the index of buffer and the shape
  // of constant operand.
  auto* array_buffer_view = constant->ArrayBufferView();
  CHECK_NE(array_buffer_view, nullptr);
  CHECK(!array_buffer_view->IsDetached());
  // Create `tflite::Buffer` with raw data buffers for constant operand.
  auto buffer = tflite::CreateBuffer(
      builder_,
      builder_.CreateVector(reinterpret_cast<const uint8_t*>(
                                array_buffer_view->BaseAddressMaybeShared()),
                            array_buffer_view->byteLength()));
  if (buffer.IsNull()) {
    return base::unexpected("Failed to create buffer.");
  }
  // The index of buffer is referenced by tensors.
  uint32_t buffer_index = base::checked_cast<uint32_t>(buffers_.size());
  CHECK_GT(buffer_index, uint32_t(0));
  buffers_.emplace_back(buffer);

  return buffer_index;
}

base::expected<int32_t, String> MLGraphTfLiteConverter::SerializeTensor(
    const MLOperand* operand,
    absl::optional<String> graph_output_name) {
  CHECK_NE(operand, nullptr);
  // The buffer index 0 represents input and output operand because there is no
  // data buffer associated.
  uint32_t buffer_index = 0;
  // The index of `tflite::Tensor` array, each `MLOperand` (input, constant,
  // output) will be converted and pushed back into the array, so it's increased
  // by one after each serialization in flat buffer.
  int32_t tensor_index = base::checked_cast<int32_t>(tensors_.size());
  CHECK_GE(tensor_index, int32_t(0));
  // The name identifies the tensor for inference, so only inputs and outputs of
  // graph have this attribute.
  absl::optional<String> name;
  switch (operand->Kind()) {
    case MLOperand::OperandKind::kInput: {
      name = operand->Name();
      // Fill the graph inputs with the index of input tensor.
      graph_inputs_.push_back(tensor_index);
      break;
    }
    case MLOperand::OperandKind::kConstant: {
      // Serialize buffer and return buffer index which starts from 1, it is
      // used to create the constant's tensor.
      auto buffer_index_result = SerializeBuffer(operand);
      if (!buffer_index_result.has_value()) {
        return base::unexpected(buffer_index_result.error());
      }
      buffer_index = buffer_index_result.value();
      break;
    }
    case MLOperand::OperandKind::kOutput: {
      // The `kOutput` represents not only the intermediate operands of
      // operation, but also the outputs of graph.
      // It's graph output if the argument `graph_output_name` has value.
      if (graph_output_name) {
        name = graph_output_name.value();
        // Fill the graph outputs with the index of output tensor.
        graph_outputs_.push_back(tensor_index);
      }
      break;
    }
  }
  // Create `Tensor` with operand shape, the index of buffer and the name.
  const auto tensor = tflite::CreateTensor(
      builder_,
      builder_.CreateVector<int32_t>(ConvertDimensions(operand->Dimensions())),
      BlinkOperandTypeToTFLite(operand->Type()), buffer_index,
      name.has_value() ? builder_.CreateString(name->Utf8()) : 0);
  if (tensor.IsNull()) {
    return base::unexpected("Failed to create tensor.");
  }

  // Add the `tensor` to the list of |Tensor|.
  tensors_.emplace_back(tensor);
  return tensor_index;
}

base::expected<void, String> MLGraphTfLiteConverter::SerializeOperation(
    const OperandToIndexMap& operand_to_index_map,
    const MLOperator* ml_operator) {
  base::expected<OperatorOffset, String> operator_offset;
  switch (ml_operator->Kind()) {
    case MLOperator::OperatorKind::kClamp:
      NOTIMPLEMENTED();
      break;
    case MLOperator::OperatorKind::kConv2d:
      operator_offset =
          SerializeConv2d(operand_to_index_map, ml_operator, builder_,
                          operator_codes_, buffers_, tensors_);
      break;
    case MLOperator::OperatorKind::kAdd:
    case MLOperator::OperatorKind::kSub:
    case MLOperator::OperatorKind::kMul:
    case MLOperator::OperatorKind::kDiv:
    case MLOperator::OperatorKind::kMin:
    case MLOperator::OperatorKind::kMax:
    case MLOperator::OperatorKind::kPow:
      operator_offset = SerializeElementWiseBinary(
          operand_to_index_map, ml_operator, builder_, operator_codes_);
      break;
    case MLOperator::OperatorKind::kGemm:
      NOTIMPLEMENTED();
      break;
    case MLOperator::OperatorKind::kAveragePool2d:
    case MLOperator::OperatorKind::kMaxPool2d:
      operator_offset = SerializePool2d(operand_to_index_map, ml_operator,
                                        builder_, operator_codes_);
      break;
    case MLOperator::OperatorKind::kRelu:
      operator_offset = SerializeRelu(operand_to_index_map, ml_operator,
                                      builder_, operator_codes_);
      break;
    case MLOperator::OperatorKind::kSoftmax:
      operator_offset = SerializeSoftmax(operand_to_index_map, ml_operator,
                                         builder_, operator_codes_);
      break;
    case MLOperator::OperatorKind::kReshape:
      operator_offset = SerializeReshape(operand_to_index_map, ml_operator,
                                         builder_, operator_codes_);
      break;
    case MLOperator::OperatorKind::kHardSwish:
    case MLOperator::OperatorKind::kReduceMean:
    case MLOperator::OperatorKind::kReduceSum:
    case MLOperator::OperatorKind::kResample2d:
    case MLOperator::OperatorKind::kSigmoid:
    case MLOperator::OperatorKind::kConcat:
    case MLOperator::OperatorKind::kTranspose:
    case MLOperator::OperatorKind::kLeakyRelu:
    case MLOperator::OperatorKind::kConvTranspose2d:
    case MLOperator::OperatorKind::kPRelu:
    case MLOperator::OperatorKind::kPad:
    case MLOperator::OperatorKind::kElu:
    case MLOperator::OperatorKind::kAbs:
    case MLOperator::OperatorKind::kCeil:
    case MLOperator::OperatorKind::kFloor:
    case MLOperator::OperatorKind::kNeg:
    case MLOperator::OperatorKind::kSlice:
    case MLOperator::OperatorKind::kSplit:
    case MLOperator::OperatorKind::kTanh:
      NOTIMPLEMENTED();
  }
  // The offset is allowed to be 0 to indicate a null object.
  if (!operator_offset.has_value()) {
    return base::unexpected(operator_offset.error());
  }
  // Add the offset to the list of operators.
  operators_.emplace_back(operator_offset.value());

  return base::ok();
}

base::expected<flatbuffers::DetachedBuffer, String>
MLGraphTfLiteConverter::FinishAndGetFlatBuffer() {
  // Create `tflite::SubGraph`, which typically represents an entire model.
  // The inputs of subgraph are the list of non-static tensors that feed into
  // the subgraph for inference. The outputs of subgraph are considered the
  // product of the subgraph's inference. The operators are in execution order.
  flatbuffers::Offset<tflite::SubGraph> subgraph = tflite::CreateSubGraph(
      builder_, builder_.CreateVector(tensors_.data(), tensors_.size()),
      builder_.CreateVector<int32_t>(graph_inputs_),
      builder_.CreateVector<int32_t>(graph_outputs_),
      builder_.CreateVector(operators_.data(), operators_.size()));
  if (subgraph.IsNull()) {
    // Value is allowed to be 0 to indicate a null object (see e.g. AddOffset).
    return base::unexpected("Failed to create graph.");
  }

  flatbuffers::Offset<flatbuffers::String> description =
      builder_.CreateString("TF-Lite model");
  if (description.IsNull()) {
    // Value is allowed to be 0 to indicate a null object (see e.g. AddOffset).
    return base::unexpected("Failed to create description for the model.");
  }

  // The operator codes used in this model are kept in order because operators
  // carry an index into this vector.
  // There is only one subgraph in the model. The buffers of the model must be
  // initialized an empty buffer.
  flatbuffers::Offset<tflite::Model> model_buffer = tflite::CreateModel(
      builder_, TFLITE_SCHEMA_VERSION,
      builder_.CreateVector(operator_codes_.data(), operator_codes_.size()),
      builder_.CreateVector(&subgraph, 1), description,
      builder_.CreateVector(buffers_.data(), buffers_.size()));
  if (model_buffer.IsNull()) {
    // Value is allowed to be 0 to indicate a null object (see e.g. AddOffset).
    return base::unexpected("Failed to create the model.");
  }

  tflite::FinishModelBuffer(builder_, model_buffer);

  return builder_.Release();
}

}  // namespace blink
