// Copyright 2023 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/tflite_converter.h"

#include <numeric>

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

int32_t GetOperatorInputIndex(const MLOperator* op,
                              const OperandTensorIndexMap& operand_index_map,
                              wtf_size_t index = 0) {
  CHECK_LE(index, op->Inputs().size());
  const auto* input = op->Inputs()[index].Get();
  CHECK_NE(op, nullptr);
  CHECK(operand_index_map.Contains(input));
  return operand_index_map.at(input);
}

int32_t GetOperatorOutputIndex(const MLOperator* op,
                               const OperandTensorIndexMap& operand_index_map,
                               wtf_size_t index = 0) {
  CHECK_LE(index, op->Outputs().size());
  const auto* output = op->Outputs()[index].Get();
  CHECK_NE(op, nullptr);
  CHECK(operand_index_map.Contains(output));
  return operand_index_map.at(output);
}

Vector<int32_t> ConvertDimensions(const Vector<uint32_t>& dimensions) {
  Vector<int32_t> new_dims;
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
bool GetTfLitePaddingMode(const OptionsType* options,
                          uint32_t input_height,
                          uint32_t input_width,
                          uint32_t filter_height,
                          uint32_t filter_width,
                          uint32_t stride_height,
                          uint32_t stride_width,
                          uint32_t dilation_height,
                          uint32_t dilation_width,
                          tflite::Padding& padding_mode,
                          String& error_message) {
  bool valid_padding;
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
        valid_padding = true;
      } else {
        // Calculate the tflite kSameUpper padding based on WebNN auto padding
        // mode and sizes.
        Vector<uint32_t> same_pads;
        auto padding_sizes_height = MLGraphBuilder::CalculateConv2dPadding(
            V8MLAutoPad::Enum::kSameUpper, input_height, filter_height,
            stride_height, dilation_height);
        CHECK(padding_sizes_height);
        same_pads.push_back(padding_sizes_height.value().begin);
        same_pads.push_back(padding_sizes_height.value().end);
        auto padding_sizes_width = MLGraphBuilder::CalculateConv2dPadding(
            V8MLAutoPad::Enum::kSameUpper, input_width, filter_width,
            stride_width, dilation_width);
        same_pads.push_back(padding_sizes_width.value().begin);
        same_pads.push_back(padding_sizes_width.value().end);

        if (explicit_pads == same_pads) {
          padding_mode = tflite::Padding_SAME;
          valid_padding = true;
        } else {
          error_message = "The explicit padding are not supported in tflite";
          valid_padding = false;
        }
      }
      break;
    }
    case V8MLAutoPad::Enum::kSameUpper: {
      padding_mode = tflite::Padding_SAME;
      valid_padding = true;
      break;
    }
    case V8MLAutoPad::Enum::kSameLower: {
      error_message = "Same lower is not supported in tflite";
      valid_padding = false;
      break;
    }
  }
  return valid_padding;
}

}  // namespace

TfLiteConverter::TfLiteConverter() {
  // It is required that the first entry in the buffers of model is always an
  // empty buffer. This is so that the default buffer index of zero in Tensor
  // will always refer to a valid empty buffer.
  model_info_.buffers.push_back(
      tflite::CreateBuffer(builder_, builder_.CreateVector({})));
}

TfLiteConverter::~TfLiteConverter() = default;

void TfLiteConverter::Trace(Visitor* visitor) const {}

bool TfLiteConverter::SerializeConstant(
    const MLOperand* constant,
    OperandTensorIndexMap& operand_index_map) {
  // There are two steps to implement the `BuildBuffer` function:
  // 1, Create `tflite::Buffer` with array buffer view.
  // 2, Create `tflite::Tensor` with the index of buffer and the shape
  // of constant operand.
  auto* array_buffer_view = constant->ArrayBufferView();
  // Create `tflite::Buffer` with raw data buffers for constant operand.
  model_info_.buffers.push_back(tflite::CreateBuffer(
      builder_,
      builder_.CreateVector(reinterpret_cast<const uint8_t*>(
                                array_buffer_view->BaseAddressMaybeShared()),
                            array_buffer_view->byteLength())));
  // The index of buffer is referenced by tensors.
  uint32_t buffer_index = static_cast<uint32_t>(model_info_.buffers.size()) - 1;
  CHECK_GT(buffer_index, uint32_t(0));
  // Build tensor for constant operand with the index of buffer.
  if (!SerializeTensor(constant, buffer_index)) {
    return false;
  }

  int32_t tensor_index =
      base::checked_cast<int32_t>(subgraph_desc_.tensors.size()) - 1;
  CHECK_GE(tensor_index, int32_t(0));
  operand_index_map.insert(constant, tensor_index);
  return true;
}

bool TfLiteConverter::SerializeInput(const MLOperand* input,
                                     OperandTensorIndexMap& operand_index_map) {
  // Serialize tensor for input operand with the name which is to identify
  // array buffer of input when computing graph. The index of tensor is
  // 0 which refers to an empty buffer.
  if (!SerializeTensor(input, 0, input->Name())) {
    return false;
  }
  // Fill subgraph inputs with the index of input.
  int32_t tensor_index =
      base::checked_cast<int32_t>(subgraph_desc_.tensors.size()) - 1;
  CHECK_GE(tensor_index, int32_t(0));
  subgraph_desc_.inputs_index.push_back(tensor_index);
  // Insert the index of tensor in the hash map.
  operand_index_map.insert(input, tensor_index);
  return true;
}

bool TfLiteConverter::SerializeOutput(
    const MLOperand* output,
    const OperandNameMap& output_operand_name_map,
    OperandTensorIndexMap& operand_index_map) {
  // Get name of output operand to build output tensor.
  String name =
      output_operand_name_map.find(output) != output_operand_name_map.end()
          ? output_operand_name_map.at(output)
          : String();
  // Create `tflite::Tensor` for intermediate operand, or model output
  // operand with output name.
  if (!SerializeTensor(output, 0, name)) {
    return false;
  }

  int32_t tensor_index =
      base::checked_cast<int32_t>(subgraph_desc_.tensors.size()) - 1;
  CHECK_GE(tensor_index, int32_t(0));
  if (output_operand_name_map.find(output) != output_operand_name_map.end()) {
    // Fill the subgraph outputs with the index of output tensor.
    subgraph_desc_.outputs_index.push_back(tensor_index);
  }
  operand_index_map.insert(output, tensor_index);
  return true;
}

bool TfLiteConverter::SerializeTensor(const MLOperand* operand,
                                      uint32_t buffer_index,
                                      String name) {
  // Create `Tensor` with operand shape, the index of buffer and the name.
  // The tensor shape is NHWC ([batch size, height, width, number of channels]).
  // The buffer index is 0 for intermediate and output operand because there is
  // no data buffer associated. Input and output operand has the name to
  // identify array buffer, there are no name for intermediate operand.
  const auto offset = tflite::CreateTensor(
      builder_,
      builder_.CreateVector<int32_t>(ConvertDimensions(operand->Dimensions())),
      BlinkOperandTypeToTFLite(operand->Type()), buffer_index,
      name.empty() ? 0 : builder_.CreateString(name.Utf8()));
  // The offset is allowed to be 0 to indicate a null object.
  if (offset.IsNull()) {
    return false;
  }

  // Add the offset to the list of |Tensor|.
  subgraph_desc_.tensors.emplace_back(offset);
  return true;
}

bool TfLiteConverter::SerializeOperations(
    const MLOperator* op,
    const OperandTensorIndexMap& operand_index_map,
    String& error_message) {
  OperatorOffset offset;
  switch (op->Kind()) {
    case MLOperator::OperatorKind::kClamp:
      NOTIMPLEMENTED();
      break;
    case MLOperator::OperatorKind::kAdd:
    case MLOperator::OperatorKind::kSub:
    case MLOperator::OperatorKind::kMul:
    case MLOperator::OperatorKind::kDiv:
    case MLOperator::OperatorKind::kMin:
    case MLOperator::OperatorKind::kMax:
      offset = SerializeElementWiseBinary(op, operand_index_map);
      break;
    case MLOperator::OperatorKind::kRelu:
      offset = SerializeRelu(op, operand_index_map);
      break;
    case MLOperator::OperatorKind::kReshape:
      offset = SerializeReshape(op, operand_index_map);
      break;
    case MLOperator::OperatorKind::kSoftmax:
      offset = SerializeSoftmax(op, operand_index_map);
      break;
    case MLOperator::OperatorKind::kConv2d:
      offset = SerializeConv2d(op, operand_index_map, error_message);
      break;
    case MLOperator::OperatorKind::kGemm:
    case MLOperator::OperatorKind::kAveragePool2d:
    case MLOperator::OperatorKind::kMaxPool2d:
      offset = SerializePool2d(op, operand_index_map, error_message);
      break;
    case MLOperator::OperatorKind::kHardSwish:
    // case MLOperator::OperatorKind::kReduceMean:
    // case MLOperator::OperatorKind::kReduceSum:
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
      break;
  }
  // The offset is allowed to be 0 to indicate a null object.
  if (offset.IsNull()) {
    return false;
  }
  // Add the offset to the list of operators.
  subgraph_desc_.operators.emplace_back(offset);

  return true;
}

uint32_t TfLiteConverter::SerializeEmptyBuffer(uint32_t output_channels) {
  Vector<int32_t> dimensions = {base::checked_cast<int32_t>(output_channels)};
  size_t buffer_size = std::accumulate(dimensions.begin(), dimensions.end(),
                                       size_t(1), std::multiplies<int32_t>());
  std::vector<float> empty_data(buffer_size);
  // Create `tflite::Buffer` with raw data buffers for constant operand.
  model_info_.buffers.push_back(tflite::CreateBuffer(
      builder_,
      builder_.CreateVector(reinterpret_cast<const uint8_t*>(empty_data.data()),
                            empty_data.size() * sizeof(float))));
  // The index of buffer is referenced by tensors.
  uint32_t buffer_index = static_cast<uint32_t>(model_info_.buffers.size()) - 1;
  // Create `Tensor` with operand shape, the index of buffer and the name.
  // The tensor shape is NHWC ([batch size, height, width, number of channels]).
  // The buffer index is 0 for intermediate and output operand because there is
  // no data buffer associated. Input and output operand has the name to
  // identify array buffer, there are no name for intermediate operand.
  subgraph_desc_.tensors.emplace_back(
      tflite::CreateTensor(builder_, builder_.CreateVector<int32_t>(dimensions),
                           tflite::TensorType_FLOAT32, buffer_index));
  return base::checked_cast<uint32_t>(subgraph_desc_.tensors.size() - 1);
}

OperatorOffset TfLiteConverter::SerializeConv2d(
    const MLOperator* conv2d,
    const OperandTensorIndexMap& operand_index_map,
    String& error_message) {
  const int32_t input_index =
      GetOperatorInputIndex(conv2d, operand_index_map, 0);
  const int32_t filter_index =
      GetOperatorInputIndex(conv2d, operand_index_map, 1);
  const int32_t output_index =
      GetOperatorOutputIndex(conv2d, operand_index_map);

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
        error_message =
            String::Format("The filter layout %s is not supported .",
                           options->filterLayout().AsCStr());
        return OperatorOffset();
      }
    } else {
      // For depthwise conv2d, TF-Lite expects weights layout in ihwo that is
      // [1, kernel_height, kernel_width, input_channels * depth_multiplier].
      // TODO(crbug.com/1273291): support other layouts by transposing the
      // filter operand.
      if (options->filterLayout().AsEnum() !=
          V8MLConv2dFilterOperandLayout::Enum::kIhwo) {
        error_message = String::Format("The filter layout %s is not supported.",
                                       options->filterLayout().AsCStr());
        return OperatorOffset();
      }
    }
    const auto* filter = conv2d->Inputs()[1].Get();
    CHECK(filter);
    filter_height = filter->Dimensions()[1];
    filter_width = filter->Dimensions()[2];
  } else {
    // TODO(crbug.com/1273291): support other layouts by transposing the input
    // operand.
    error_message = String::Format("The input layout %s is not supported.",
                                   options->inputLayout().AsCStr());
    return OperatorOffset();
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
          return OperatorOffset();
        }
        break;
      }
      case MLOperator::OperatorKind::kRelu:
        activation = tflite::ActivationFunctionType_RELU;
        break;
      default:
        error_message =
            "Only clamp and relu fused operator are supported by conv2d.";
        return OperatorOffset();
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
  tflite::Padding padding_mode = tflite::Padding_VALID;
  bool valid_padding = GetTfLitePaddingMode(
      options, input_height, input_width, filter_height, filter_width,
      stride_height, stride_width, dilation_height, dilation_width,
      padding_mode, error_message);
  if (!valid_padding) {
    return OperatorOffset();
  }

  tflite::BuiltinOperator binary_op;
  tflite::BuiltinOptions builtin_options_type = tflite::BuiltinOptions_NONE;
  flatbuffers::Offset<void> builtin_options = 0;
  if (depthwise) {
    const uint32_t depth_multiplier = 1;
    binary_op = tflite::BuiltinOperator_DEPTHWISE_CONV_2D;
    builtin_options =
        tflite::CreateDepthwiseConv2DOptions(
            builder_, padding_mode, stride_width, stride_height,
            depth_multiplier, activation, dilation_width, dilation_height)
            .Union();
    builtin_options_type = tflite::BuiltinOptions_DepthwiseConv2DOptions;
  } else {
    binary_op = tflite::BuiltinOperator_CONV_2D;
    builtin_options = tflite::CreateConv2DOptions(
                          builder_, padding_mode, stride_width, stride_height,
                          activation, dilation_width, dilation_height)
                          .Union();
    builtin_options_type = tflite::BuiltinOptions_Conv2DOptions;
  }

  model_info_.operator_codes.push_back(
      tflite::CreateOperatorCode(builder_, binary_op));

  // Create `tflite::Operator` with the tensor index of inputs and outputs
  // operand. The type of operation is determined by an index into the list of
  // valid OperatorCodes.
  uint32_t opcode_index =
      base::checked_cast<uint32_t>(model_info_.operator_codes.size()) - 1;
  // If there is no bias operand, serialize a empty buffer with the size of
  // output channel.
  const int32_t bias_index =
      conv2d->Inputs().size() == 3
          ? GetOperatorInputIndex(conv2d, operand_index_map, 2)
          : SerializeEmptyBuffer(output_channels);
  const std::vector<int32_t> op_inputs = {input_index, filter_index,
                                          bias_index};
  const std::vector<int32_t> op_outputs = {output_index};

  return tflite::CreateOperator(builder_, opcode_index,
                                builder_.CreateVector<int32_t>(op_inputs),
                                builder_.CreateVector<int32_t>(op_outputs),
                                builtin_options_type, builtin_options);
}

OperatorOffset TfLiteConverter::SerializePool2d(
    const MLOperator* pool2d,
    const OperandTensorIndexMap& operand_index_map,
    String& error_message) {
  const int32_t input_index = GetOperatorInputIndex(pool2d, operand_index_map);
  const int32_t output_index =
      GetOperatorOutputIndex(pool2d, operand_index_map);

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
    error_message = "Pool2d in TF-Lite schema doesn't support dilations.";
    return OperatorOffset();
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
      error_message = "The nchw input layout is not supported.";
      return OperatorOffset();
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
  // Create `tflite::OperatorCode` for pool2d operations.
  model_info_.operator_codes.push_back(
      tflite::CreateOperatorCode(builder_, pool_op));

  // Set tflite padding mode.
  tflite::Padding padding_mode = tflite::Padding_VALID;
  bool valid_padding = GetTfLitePaddingMode(
      options, input_height, input_width, filter_height, filter_width,
      stride_height, stride_width, dilation_height, dilation_width,
      padding_mode, error_message);
  if (!valid_padding) {
    return OperatorOffset();
  }
  flatbuffers::Offset<tflite::Pool2DOptions> pool_2d_options =
      CreatePool2DOptions(builder_, tflite::Padding_VALID, stride_width,
                          stride_height, filter_width, filter_height,
                          tflite::ActivationFunctionType_NONE);

  // Create `tflite::Operator` with the tensor index of inputs and outputs
  // operand. The type of operation is determined by an index into the list of
  // valid OperatorCodes, while the specifics of each operations is configured
  // using builtin_options or custom_options.
  uint32_t opcode_index =
      base::checked_cast<uint32_t>(model_info_.operator_codes.size()) - 1;
  const std::vector<int32_t> op_inputs = {input_index};
  const std::vector<int32_t> op_outputs = {output_index};

  return tflite::CreateOperator(
      builder_, opcode_index, builder_.CreateVector<int32_t>(op_inputs),
      builder_.CreateVector<int32_t>(op_outputs),
      tflite::BuiltinOptions_Pool2DOptions, pool_2d_options.Union());
}

OperatorOffset TfLiteConverter::SerializeElementWiseBinary(
    const MLOperator* binary,
    const OperandTensorIndexMap& operand_index_map) {
  const int32_t lhs_index = GetOperatorInputIndex(binary, operand_index_map, 0);
  const int32_t rhs_index = GetOperatorInputIndex(binary, operand_index_map, 1);
  const int32_t output_index =
      GetOperatorOutputIndex(binary, operand_index_map);
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
    default:
      NOTREACHED();
      return OperatorOffset();
  }

  // Create `tflite::OperatorCode` for elementwise binary operations.
  model_info_.operator_codes.push_back(
      tflite::CreateOperatorCode(builder_, binary_op));
  // TF-Lite support activation in elementwise binary operations, but WebNN Spec
  // doesn't define the feature, so the options of elementwise binary doesn't
  // need to be configured.
  tflite::BuiltinOptions builtin_options_type = tflite::BuiltinOptions_NONE;
  flatbuffers::Offset<void> builtin_options = 0;

  // Create `tflite::Operator` with the tensor index of inputs and outputs
  // operand. The type of operation is determined by an index into the list of
  // valid OperatorCodes, while the specifics of each operations is configured
  // using builtin_options or custom_options.
  uint32_t opcode_index =
      base::checked_cast<uint32_t>(model_info_.operator_codes.size()) - 1;
  const std::vector<int32_t> op_inputs = {lhs_index, rhs_index};
  const std::vector<int32_t> op_outputs = {output_index};

  return tflite::CreateOperator(builder_, opcode_index,
                                builder_.CreateVector<int32_t>(op_inputs),
                                builder_.CreateVector<int32_t>(op_outputs),
                                builtin_options_type, builtin_options);
}

OperatorOffset TfLiteConverter::SerializeRelu(
    const MLOperator* relu,
    const OperandTensorIndexMap& operand_index_map) {
  const int32_t input_index = GetOperatorInputIndex(relu, operand_index_map);
  const int32_t output_index = GetOperatorOutputIndex(relu, operand_index_map);

  // Create `tflite::OperatorCode` for Relu operation.
  model_info_.operator_codes.push_back(
      tflite::CreateOperatorCode(builder_, tflite::BuiltinOperator_RELU));

  // Create `tflite::Operator` with the tensor index of inputs and outputs
  // operand. The type of operation is determined by an index into the list of
  // valid OperatorCodes.
  uint32_t opcode_index =
      base::checked_cast<uint32_t>(model_info_.operator_codes.size()) - 1;
  const std::vector<int32_t> op_inputs = {input_index};
  const std::vector<int32_t> op_outputs = {output_index};

  return tflite::CreateOperator(builder_, opcode_index,
                                builder_.CreateVector<int32_t>(op_inputs),
                                builder_.CreateVector<int32_t>(op_outputs));
}

OperatorOffset TfLiteConverter::SerializeSoftmax(
    const MLOperator* softmax,
    const OperandTensorIndexMap& operand_index_map) {
  const int32_t input_index = GetOperatorInputIndex(softmax, operand_index_map);
  const int32_t output_index =
      GetOperatorOutputIndex(softmax, operand_index_map);

  // Create `tflite::OperatorCode` for Softmax operation.
  model_info_.operator_codes.push_back(
      tflite::CreateOperatorCode(builder_, tflite::BuiltinOperator_SOFTMAX));

  flatbuffers::Offset<tflite::SoftmaxOptions> softmax_options =
      tflite::CreateSoftmaxOptions(builder_, 1.0);
  // Create `tflite::Operator` with the tensor index of inputs and outputs
  // operand. The type of operation is determined by an index into the list of
  // valid OperatorCodes.
  uint32_t opcode_index =
      base::checked_cast<uint32_t>(model_info_.operator_codes.size()) - 1;
  const std::vector<int32_t> op_inputs = {input_index};
  const std::vector<int32_t> op_outputs = {output_index};

  return tflite::CreateOperator(
      builder_, opcode_index, builder_.CreateVector<int32_t>(op_inputs),
      builder_.CreateVector<int32_t>(op_outputs),
      tflite::BuiltinOptions_SoftmaxOptions, softmax_options.Union());
}

OperatorOffset TfLiteConverter::SerializeReshape(
    const MLOperator* reshape,
    const OperandTensorIndexMap& operand_index_map) {
  const int32_t input_index = GetOperatorInputIndex(reshape, operand_index_map);
  const int32_t output_index =
      GetOperatorOutputIndex(reshape, operand_index_map);

  // Create `tflite::OperatorCode` for Softmax operation.
  model_info_.operator_codes.push_back(
      tflite::CreateOperatorCode(builder_, tflite::BuiltinOperator_RESHAPE));

  auto* output = reshape->Outputs()[0].Get();
  CHECK(output);
  flatbuffers::Offset<tflite::ReshapeOptions> reshape_options =
      tflite::CreateReshapeOptions(
          builder_, builder_.CreateVector<int32_t>(
                        ConvertDimensions(output->Dimensions())));
  // Create `tflite::Operator` with the tensor index of inputs and outputs
  // operand. The type of operation is determined by an index into the list of
  // valid OperatorCodes.
  uint32_t opcode_index =
      base::checked_cast<uint32_t>(model_info_.operator_codes.size()) - 1;
  const std::vector<int32_t> op_inputs = {input_index};
  const std::vector<int32_t> op_outputs = {output_index};

  return tflite::CreateOperator(
      builder_, opcode_index, builder_.CreateVector<int32_t>(op_inputs),
      builder_.CreateVector<int32_t>(op_outputs),
      tflite::BuiltinOptions_ReshapeOptions, reshape_options.Union());
}

bool TfLiteConverter::BuildModel() {
  // Create `tflite::SubGraph`, which typically represents an entire model.
  // The inputs of subgraph are the list of non-static tensors that feed into
  // the subgraph for inference. The outputs of subgraph are considered the
  // product of the subgraph's inference. The operators are in execution order.
  flatbuffers::Offset<tflite::SubGraph> subgraph = tflite::CreateSubGraph(
      builder_,
      builder_.CreateVector(subgraph_desc_.tensors.data(),
                            subgraph_desc_.tensors.size()),
      builder_.CreateVector<int32_t>(subgraph_desc_.inputs_index),
      builder_.CreateVector<int32_t>(subgraph_desc_.outputs_index),
      builder_.CreateVector(subgraph_desc_.operators.data(),
                            subgraph_desc_.operators.size()));
  if (subgraph.IsNull()) {
    // Value is allowed to be 0 to indicate a null object (see e.g. AddOffset).
    return false;
  }

  flatbuffers::Offset<flatbuffers::String> description =
      builder_.CreateString("TF-Lite model");
  if (description.IsNull()) {
    // Value is allowed to be 0 to indicate a null object (see e.g. AddOffset).
    return false;
  }

  // The operator codes used in this model are kept in order because operators
  // carry an index into this vector.
  // There is only one subgraph in the model. The buffers of the model must be
  // initialized an empty buffer.
  flatbuffers::Offset<tflite::Model> model_buffer = tflite::CreateModel(
      builder_, TFLITE_SCHEMA_VERSION,
      builder_.CreateVector(model_info_.operator_codes.data(),
                            model_info_.operator_codes.size()),
      builder_.CreateVector(&subgraph, 1), description,
      builder_.CreateVector(model_info_.buffers.data(),
                            model_info_.buffers.size()));
  if (model_buffer.IsNull()) {
    // Value is allowed to be 0 to indicate a null object (see e.g. AddOffset).
    return false;
  }

  tflite::FinishModelBuffer(builder_, model_buffer);

  return true;
}

flatbuffers::FlatBufferBuilder& TfLiteConverter::GetFlatBufferBuilder() {
  return builder_;
}

}  // namespace blink
