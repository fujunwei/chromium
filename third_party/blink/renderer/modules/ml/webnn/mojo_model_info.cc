// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/mojo_model_info.h"

#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_clamp_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_conv_2d_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_gemm_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_operand_descriptor.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_pool_2d_options.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_builder.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operand.h"
#include "third_party/blink/renderer/modules/ml/webnn/mojo_graph.h"
#include "third_party/blink/renderer/platform/bindings/exception_code.h"
#include "third_party/blink/renderer/platform/bindings/exception_state.h"

namespace blink {

base::CheckedNumeric<size_t> Align(size_t value, uint32_t aligment) {
  size_t remainder = value % aligment;
  if (remainder != 0) {
    value += aligment - remainder;
  }

  return value;
}

namespace {

using ml::webnn::mojom::blink::AutoPad;
using ml::webnn::mojom::blink::Conv2dFilterOperandLayout;
using ml::webnn::mojom::blink::ElementWiseBinaryType;
using ml::webnn::mojom::blink::InputOperandLayout;
using ml::webnn::mojom::blink::OperandType;
using ml::webnn::mojom::blink::OperationInfo;
using ml::webnn::mojom::blink::OperationInfoPtr;
using ml::webnn::mojom::blink::Pool2dType;
using ml::webnn::mojom::blink::RoundingType;

OperandType BlinkOperandTypeToMojo(V8MLOperandType::Enum type) {
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

InputOperandLayout BlinkInputOperandLayoutToMojo(
    V8MLInputOperandLayout::Enum type) {
  switch (type) {
    case V8MLInputOperandLayout::Enum::kNchw:
      return InputOperandLayout::kNchw;
    case V8MLInputOperandLayout::Enum::kNhwc:
      return InputOperandLayout::kNhwc;
  }
}

Conv2dFilterOperandLayout BlinkConv2dFilterOperandLayoutToMojo(
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

AutoPad BlinkAutoPadToMojo(V8MLAutoPad::Enum type) {
  switch (type) {
    case V8MLAutoPad::Enum::kExplicit:
      return AutoPad::kExplicit;
    case V8MLAutoPad::Enum::kSameUpper:
      return AutoPad::kSameUpper;
    case V8MLAutoPad::Enum::kSameLower:
      return AutoPad::kSameLower;
  }
}

RoundingType BlinkRoundingTypeToMojo(V8MLRoundingType::Enum type) {
  switch (type) {
    case V8MLRoundingType::Enum::kFloor:
      return RoundingType::kFloor;
    case V8MLRoundingType::Enum::kCeil:
      return RoundingType::kCeil;
  }
}

Pool2dType BlinkPool2dTypeToMojo(MLOperator::OperatorKind type) {
  switch (type) {
    case MLOperator::OperatorKind::kAveragePool2d:
      return Pool2dType::kAveragePool2d;
    default:
      NOTREACHED();
      return Pool2dType::kUnknown;
  }
}

ElementWiseBinaryType BlinkElementWiseBinaryTypeToMojo(
    MLOperator::OperatorKind type) {
  switch (type) {
    case MLOperator::OperatorKind::kAdd:
      return ElementWiseBinaryType::kAdd;
    default:
      NOTREACHED();
      return ElementWiseBinaryType::kUnknown;
  }
}

ml::webnn::mojom::blink::ClampOptionsPtr BlinkClampOptioinToMojo(
    const MLClampOptions* ml_options) {
  const float min = ml_options->hasMinValue()
                        ? ml_options->minValue()
                        : -std::numeric_limits<float>::infinity();
  const float max = ml_options->hasMaxValue()
                        ? ml_options->maxValue()
                        : +std::numeric_limits<float>::infinity();
  auto options = ml::webnn::mojom::blink::ClampOptions::New();
  options->minValue = min;
  options->maxValue = max;

  return options;
}

OperationInfoPtr FusionOperation(const MLOperator* activation) {
  switch (activation->Kind()) {
    case MLOperator::OperatorKind::kClamp: {
      auto clamp = ml::webnn::mojom::blink::Clamp::New();
      clamp->input_index = std::numeric_limits<uint64_t>::max();
      clamp->options = BlinkClampOptioinToMojo(
          static_cast<const MLClampOptions*>(activation->Options()));
      clamp->output_index = std::numeric_limits<uint64_t>::max();
      auto operation = OperationInfo::NewClamp(std::move(clamp));
      return operation;
    }
    case MLOperator::OperatorKind::kRelu: {
      auto relu = ml::webnn::mojom::blink::Relu::New();
      relu->input_index = std::numeric_limits<uint64_t>::max();
      relu->output_index = std::numeric_limits<uint64_t>::max();
      auto operation = OperationInfo::NewRelu(std::move(relu));
      return operation;
    }
    default: {
      NOTREACHED();
      return nullptr;
    }
  }
}

ml::webnn::mojom::blink::Conv2dOptionsPtr BlinkConv2dOptioinToMojo(
    const MLConv2dOptions* ml_options,
    const HeapHashMap<Member<const MLOperand>, size_t>& operand_index_map) {
  auto options = ml::webnn::mojom::blink::Conv2dOptions::New();
  options->padding =
      ml_options->hasPadding() ? ml_options->padding() : Vector<int32_t>(4, 0);
  options->strides = ml_options->hasStrides()
                         ? ml_options->strides()
                         : Vector<int32_t>(2, static_cast<uint32_t>(1));
  options->dilations = ml_options->hasDilations()
                           ? ml_options->dilations()
                           : Vector<int32_t>(2, static_cast<uint32_t>(1));
  options->auto_pad = BlinkAutoPadToMojo(ml_options->autoPad().AsEnum());
  options->groups = ml_options->groups();
  options->inputLayout =
      BlinkInputOperandLayoutToMojo(ml_options->inputLayout().AsEnum());
  options->filterLayout =
      BlinkConv2dFilterOperandLayoutToMojo(ml_options->filterLayout().AsEnum());
  options->bias_index = ml_options->hasBias()
                            ? operand_index_map.at(ml_options->bias())
                            : std::numeric_limits<uint64_t>::max();
  if (ml_options->hasActivation()) {
    options->activation = FusionOperation(ml_options->activation());
  }

  return options;
}

ml::webnn::mojom::blink::Pool2dOptionsPtr BlinkPool2dOptioinToMojo(
    const MLPool2dOptions* ml_options) {
  auto options = ml::webnn::mojom::blink::Pool2dOptions::New();
  options->window_dimensions = ml_options->hasWindowDimensions()
                                   ? ml_options->windowDimensions()
                                   : Vector<int32_t>();
  options->padding =
      ml_options->hasPadding() ? ml_options->padding() : Vector<int32_t>(4, 0);
  options->strides = ml_options->hasStrides()
                         ? ml_options->strides()
                         : Vector<int32_t>(2, static_cast<uint32_t>(1));
  options->dilations = ml_options->hasDilations()
                           ? ml_options->dilations()
                           : Vector<int32_t>(2, static_cast<uint32_t>(1));
  options->auto_pad = BlinkAutoPadToMojo(ml_options->autoPad().AsEnum());
  options->layout =
      BlinkInputOperandLayoutToMojo(ml_options->layout().AsEnum());
  options->rounding_type =
      BlinkRoundingTypeToMojo(ml_options->roundingType().AsEnum());
  options->output_sizes = ml_options->hasOutputSizes()
                              ? ml_options->outputSizes()
                              : Vector<int32_t>();
  return options;
}

ml::webnn::mojom::blink::GemmOptionsPtr BlinkGemmOptioinToMojo(
    const MLGemmOptions* ml_options,
    const HeapHashMap<Member<const MLOperand>, size_t>& operand_index_map) {
  auto options = ml::webnn::mojom::blink::GemmOptions::New();
  options->c_index = ml_options->hasC() ? operand_index_map.at(ml_options->c())
                                        : std::numeric_limits<uint64_t>::max();
  options->alpha = ml_options->hasAlpha() ? ml_options->alpha() : 1.0;
  options->beta = ml_options->hasBeta() ? ml_options->beta() : 1.0;
  options->a_transpose =
      ml_options->hasATranspose() ? ml_options->aTranspose() : false;
  options->b_transpose =
      ml_options->hasBTranspose() ? ml_options->bTranspose() : false;
  return options;
}

}  // namespace

MojoModelInfo::MojoModelInfo() {
  model_info_ = ml::webnn::mojom::blink::ModelInfo::New();
}

MojoModelInfo::~MojoModelInfo() = default;

void MojoModelInfo::Trace(Visitor* visitor) const {
  visitor->Trace(operand_index_map_);
  visitor->Trace(constant_index_map_);
}

void MojoModelInfo::AddInput(const MLOperand* input) {
  // Add input operand descriptor to the model.
  size_t index = AddOperandToModel(input);
  // Add the input to the model.
  auto named_input = ml::webnn::mojom::blink::NamedOperand::New();
  named_input->name = input->Name();
  named_input->index = index;
  model_info_->inputs.push_back(std::move(named_input));
}

void MojoModelInfo::AddConstant(const MLOperand* constant) {
  // Add const operand descriptor to the model.
  size_t index = AddOperandToModel(constant);
  // All constant data will share a big shared memory, so hold the index of
  // constant temporarily in member variable.
  constant_index_map_.insert(constant, index);
}

void MojoModelInfo::AddOutput(String name, const MLOperand* output) {
  if (operand_index_map_.find(output) == operand_index_map_.end()) {
    return;
  }
  auto named_output = ml::webnn::mojom::blink::NamedOperand::New();
  named_output->name = std::move(name);
  named_output->index = operand_index_map_.at(output);
  model_info_->outputs.push_back(std::move(named_output));
}

void MojoModelInfo::AddClamp(const MLOperator* ml_clamp) {
  DCHECK_EQ(ml_clamp->Inputs().size(), static_cast<uint32_t>(1));
  auto* input = ml_clamp->Inputs()[0].Get();
  if (operand_index_map_.find(input) == operand_index_map_.end()) {
    return;
  }
  DCHECK_EQ(ml_clamp->Outputs().size(), static_cast<uint32_t>(1));
  auto* output = ml_clamp->Outputs()[0].Get();
  DCHECK(operand_index_map_.find(output) == operand_index_map_.end());
  // Add operand descriptor to the model.
  size_t output_index = AddOperandToModel(output);
  // Add clamp operation to the model.
  auto clamp = ml::webnn::mojom::blink::Clamp::New();
  clamp->input_index = operand_index_map_.at(input);
  clamp->options = BlinkClampOptioinToMojo(
      static_cast<const MLClampOptions*>(ml_clamp->Options()));
  clamp->output_index = output_index;
  auto operation = OperationInfo::NewClamp(std::move(clamp));
  model_info_->operations.push_back(std::move(operation));
}

void MojoModelInfo::AddConv2d(const MLOperator* ml_conv2d) {
  DCHECK_GE(ml_conv2d->Inputs().size(), static_cast<uint32_t>(2));
  auto* input = ml_conv2d->Inputs()[0].Get();
  auto* filter = ml_conv2d->Inputs()[1].Get();
  if (operand_index_map_.find(input) == operand_index_map_.end() ||
      operand_index_map_.find(filter) == operand_index_map_.end()) {
    return;
  }
  // Add operand descriptor to the model.
  DCHECK_EQ(ml_conv2d->Outputs().size(), static_cast<uint32_t>(1));
  auto* output = ml_conv2d->Outputs()[0].Get();
  DCHECK(operand_index_map_.find(output) == operand_index_map_.end());
  size_t output_index = AddOperandToModel(output);
  // Add clamp operation to the model.
  auto conv2d = ml::webnn::mojom::blink::Conv2d::New();
  conv2d->input_index = operand_index_map_.at(input);
  conv2d->filter_index = operand_index_map_.at(filter);
  const MLConv2dOptions* ml_options =
      static_cast<const MLConv2dOptions*>(ml_conv2d->Options());
  conv2d->options = BlinkConv2dOptioinToMojo(ml_options, operand_index_map_);
  conv2d->output_index = output_index;
  auto operation = OperationInfo::NewConv2d(std::move(conv2d));
  model_info_->operations.push_back(std::move(operation));
}

void MojoModelInfo::AddElementWiseBinary(const MLOperator* ml_binary) {
  DCHECK_EQ(ml_binary->Inputs().size(), static_cast<uint32_t>(2));
  auto* a = ml_binary->Inputs()[0].Get();
  auto* b = ml_binary->Inputs()[1].Get();
  if (operand_index_map_.find(a) == operand_index_map_.end() ||
      operand_index_map_.find(b) == operand_index_map_.end()) {
    return;
  }
  DCHECK_EQ(ml_binary->Outputs().size(), static_cast<uint32_t>(1));
  auto* output = ml_binary->Outputs()[0].Get();
  DCHECK(operand_index_map_.find(output) == operand_index_map_.end());
  size_t output_index = AddOperandToModel(output);
  auto binary = ml::webnn::mojom::blink::ElementWiseBinary::New();
  binary->type = BlinkElementWiseBinaryTypeToMojo(ml_binary->Kind());
  binary->a_index = operand_index_map_.at(a);
  binary->b_index = operand_index_map_.at(b);
  binary->output_index = output_index;
  auto operation = OperationInfo::NewElementWiseBinary(std::move(binary));
  model_info_->operations.push_back(std::move(operation));
}

void MojoModelInfo::AddGemm(const MLOperator* ml_gemm) {
  DCHECK_GE(ml_gemm->Inputs().size(), static_cast<uint32_t>(2));
  auto* a = ml_gemm->Inputs()[0].Get();
  auto* b = ml_gemm->Inputs()[1].Get();
  if (operand_index_map_.find(a) == operand_index_map_.end() ||
      operand_index_map_.find(b) == operand_index_map_.end()) {
    return;
  }
  // Add operand descriptor to the model.
  DCHECK_EQ(ml_gemm->Outputs().size(), static_cast<uint32_t>(1));
  auto* output = ml_gemm->Outputs()[0].Get();
  DCHECK(operand_index_map_.find(output) == operand_index_map_.end());
  size_t output_index = AddOperandToModel(output);
  // Add clamp operation to the model.
  auto gemm = ml::webnn::mojom::blink::Gemm::New();
  gemm->a_index = operand_index_map_.at(a);
  gemm->b_index = operand_index_map_.at(b);
  const MLGemmOptions* ml_options =
      static_cast<const MLGemmOptions*>(ml_gemm->Options());
  gemm->options = BlinkGemmOptioinToMojo(ml_options, operand_index_map_);
  gemm->output_index = output_index;
  auto operation = OperationInfo::NewGemm(std::move(gemm));
  model_info_->operations.push_back(std::move(operation));
}

void MojoModelInfo::AddPool2d(const MLOperator* ml_pool2d) {
  DCHECK_EQ(ml_pool2d->Inputs().size(), static_cast<uint32_t>(1));
  auto* input = ml_pool2d->Inputs()[0].Get();
  if (operand_index_map_.find(input) == operand_index_map_.end()) {
    return;
  }
  // Add operand descriptor to the model.
  DCHECK_EQ(ml_pool2d->Outputs().size(), static_cast<uint32_t>(1));
  auto* output = ml_pool2d->Outputs()[0].Get();
  DCHECK(operand_index_map_.find(output) == operand_index_map_.end());
  size_t output_index = AddOperandToModel(output);
  // Add averagePool2d operation to the model.
  auto pool2d = ml::webnn::mojom::blink::Pool2d::New();
  pool2d->type = BlinkPool2dTypeToMojo(ml_pool2d->Kind());
  pool2d->input_index = operand_index_map_.at(input);
  const MLPool2dOptions* ml_options =
      static_cast<const MLPool2dOptions*>(ml_pool2d->Options());
  pool2d->options = BlinkPool2dOptioinToMojo(ml_options);
  pool2d->output_index = output_index;
  auto operation = OperationInfo::NewPool2d(std::move(pool2d));
  model_info_->operations.push_back(std::move(operation));
}

void MojoModelInfo::AddRelu(const MLOperator* ml_relu) {
  DCHECK_EQ(ml_relu->Inputs().size(), static_cast<uint32_t>(1));
  auto* input = ml_relu->Inputs()[0].Get();
  if (operand_index_map_.find(input) == operand_index_map_.end()) {
    return;
  }
  // Add operand descriptor to the model.
  DCHECK_EQ(ml_relu->Outputs().size(), static_cast<uint32_t>(1));
  auto* output = ml_relu->Outputs()[0].Get();
  DCHECK(operand_index_map_.find(output) == operand_index_map_.end());
  size_t output_index = AddOperandToModel(output);
  // Add averagePool2d operation to the model.
  auto relu = ml::webnn::mojom::blink::Relu::New();
  relu->input_index = operand_index_map_.at(input);
  relu->output_index = output_index;
  auto operation = OperationInfo::NewRelu(std::move(relu));
  model_info_->operations.push_back(std::move(operation));
}

void MojoModelInfo::AddReshape(const MLOperator* ml_reshape) {
  DCHECK_EQ(ml_reshape->Inputs().size(), static_cast<uint32_t>(1));
  auto* input = ml_reshape->Inputs()[0].Get();
  if (operand_index_map_.find(input) == operand_index_map_.end()) {
    return;
  }
  // Add operand descriptor to the model.
  DCHECK_EQ(ml_reshape->Outputs().size(), static_cast<uint32_t>(1));
  auto* output = ml_reshape->Outputs()[0].Get();
  DCHECK(operand_index_map_.find(output) == operand_index_map_.end());
  size_t output_index = AddOperandToModel(output);
  // Add averagePool2d operation to the model.
  auto reshape = ml::webnn::mojom::blink::Reshape::New();
  reshape->input_index = operand_index_map_.at(input);
  reshape->output_index = output_index;
  auto operation = OperationInfo::NewReshape(std::move(reshape));
  model_info_->operations.push_back(std::move(operation));
}

void MojoModelInfo::AddSoftmax(const MLOperator* ml_softmax) {
  DCHECK_EQ(ml_softmax->Inputs().size(), static_cast<uint32_t>(1));
  auto* input = ml_softmax->Inputs()[0].Get();
  if (operand_index_map_.find(input) == operand_index_map_.end()) {
    return;
  }
  // Add operand descriptor to the model.
  DCHECK_EQ(ml_softmax->Outputs().size(), static_cast<uint32_t>(1));
  auto* output = ml_softmax->Outputs()[0].Get();
  DCHECK(operand_index_map_.find(output) == operand_index_map_.end());
  size_t output_index = AddOperandToModel(output);
  // Add averagePool2d operation to the model.
  auto softmax = ml::webnn::mojom::blink::Softmax::New();
  softmax->input_index = operand_index_map_.at(input);
  softmax->output_index = output_index;
  auto operation = OperationInfo::NewSoftmax(std::move(softmax));
  model_info_->operations.push_back(std::move(operation));
}

void MojoModelInfo::FillConstantsWithArrayBuffer() {
  // Copy constant data to shared memory.
  base::CheckedNumeric<size_t> constants_buffer_length(0);
  for (const auto& [constant, index] : constant_index_map_) {
    wtf_size_t size = base::checked_cast<wtf_size_t>(
        constant->ArrayBufferView()->byteLength());
    constants_buffer_length += Align(size, kBufferAlignment);
  }
  base::MappedReadOnlyRegion constants_shm_region =
      base::ReadOnlySharedMemoryRegion::Create(
          constants_buffer_length.ValueOrDie());
  auto constants_info = ml::webnn::mojom::blink::ConstantsInfo::New();
  base::CheckedNumeric<size_t> aligned_constant_offset(0);
  for (const auto& [constant, index] : constant_index_map_) {
    auto memory_info = ml::webnn::mojom::blink::MemoryInfo::New();
    auto* array_buffer_view = constant->ArrayBufferView();
    memory_info->byte_offset = aligned_constant_offset.ValueOrDie();
    memory_info->byte_length = array_buffer_view->byteLength();
    aligned_constant_offset +=
        Align(memory_info->byte_length, kBufferAlignment);

    uint8_t* address = constants_shm_region.mapping.GetMemoryAs<uint8_t>() +
                       memory_info->byte_offset;
    memcpy(address, array_buffer_view->BaseAddressMaybeShared(),
           array_buffer_view->byteLength());
    constants_info->memory_info.insert(index, std::move(memory_info));
  }
  constants_info->shared_memory = constants_shm_region.region.Duplicate();
  model_info_->constants =
      constant_index_map_.size() == 0 ? nullptr : std::move(constants_info);
}

ModelInfoPtr MojoModelInfo::GetModelInfo() {
  return std::move(model_info_);
}

size_t MojoModelInfo::AddOperandToModel(const MLOperand* output) {
  // Create the operand descriptor for mojo.
  auto desc = ml::webnn::mojom::blink::OperandDescriptor::New();
  desc->data_type = BlinkOperandTypeToMojo(output->Type());
  desc->dimensions = output->Dimensions();
  // Add operand descriptor into model.
  model_info_->operands.push_back(std::move(desc));
  // The index used to identify operand on the server side, each operation
  // generate a output operand that will be inserted in a hash map with the
  // MLOperand and index, the index is incremented by one.
  size_t output_index = model_info_->operands.size() - 1;
  operand_index_map_.insert(output, output_index);
  return output_index;
}

}  // namespace blink
