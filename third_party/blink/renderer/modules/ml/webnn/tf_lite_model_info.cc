// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/tf_lite_model_info.h"

#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_clamp_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_conv_2d_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_gemm_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_operand_descriptor.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_pool_2d_options.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_builder.h"
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

Vector<int32_t> ConvertType(const Vector<uint32_t>& dimensions) {
  Vector<int32_t> new_dims;
  // new_dims.reserve(dimensions.size());
  for (auto dim : dimensions) {
    new_dims.push_back(base::checked_cast<int32_t>(dim));
  }
  return new_dims;
}

// OperandType BlinkOperandTypeToMojo(V8MLOperandType::Enum type) {
//   switch (type) {
//     case V8MLOperandType::Enum::kFloat32:
//       return OperandType::kFloat32;
//     case V8MLOperandType::Enum::kFloat16:
//       return OperandType::kFloat16;
//     case V8MLOperandType::Enum::kInt32:
//       return OperandType::kInt32;
//     case V8MLOperandType::Enum::kUint32:
//       return OperandType::kUint32;
//     case V8MLOperandType::Enum::kInt8:
//       return OperandType::kInt8;
//     case V8MLOperandType::Enum::kUint8:
//       return OperandType::kUint8;
//   }
// }

// InputOperandLayout BlinkInputOperandLayoutToMojo(
//     V8MLInputOperandLayout::Enum type) {
//   switch (type) {
//     case V8MLInputOperandLayout::Enum::kNchw:
//       return InputOperandLayout::kNchw;
//     case V8MLInputOperandLayout::Enum::kNhwc:
//       return InputOperandLayout::kNhwc;
//   }
// }

// Conv2dFilterOperandLayout BlinkConv2dFilterOperandLayoutToMojo(
//     V8MLConv2dFilterOperandLayout::Enum type) {
//   switch (type) {
//     case V8MLConv2dFilterOperandLayout::Enum::kOihw:
//       return Conv2dFilterOperandLayout::kOihw;
//     case V8MLConv2dFilterOperandLayout::Enum::kHwio:
//       return Conv2dFilterOperandLayout::kHwio;
//     case V8MLConv2dFilterOperandLayout::Enum::kOhwi:
//       return Conv2dFilterOperandLayout::kOhwi;
//     case V8MLConv2dFilterOperandLayout::Enum::kIhwo:
//       return Conv2dFilterOperandLayout::kIhwo;
//   }
// }

// AutoPad BlinkAutoPadToMojo(V8MLAutoPad::Enum type) {
//   switch (type) {
//     case V8MLAutoPad::Enum::kExplicit:
//       return AutoPad::kExplicit;
//     case V8MLAutoPad::Enum::kSameUpper:
//       return AutoPad::kSameUpper;
//     case V8MLAutoPad::Enum::kSameLower:
//       return AutoPad::kSameLower;
//   }
// }

// RoundingType BlinkRoundingTypeToMojo(V8MLRoundingType::Enum type) {
//   switch (type) {
//     case V8MLRoundingType::Enum::kFloor:
//       return RoundingType::kFloor;
//     case V8MLRoundingType::Enum::kCeil:
//       return RoundingType::kCeil;
//   }
// }

// Pool2dType BlinkPool2dTypeToMojo(MLOperator::OperatorKind type) {
//   switch (type) {
//     case MLOperator::OperatorKind::kAveragePool2d:
//       return Pool2dType::kAveragePool2d;
//     default:
//       NOTREACHED();
//       return Pool2dType::kUnknown;
//   }
// }

// ml::webnn::mojom::blink::ClampOptionsPtr BlinkClampOptioinToMojo(
//     const MLClampOptions* ml_options) {
//   const float min = ml_options->hasMinValue()
//                         ? ml_options->minValue()
//                         : -std::numeric_limits<float>::infinity();
//   const float max = ml_options->hasMaxValue()
//                         ? ml_options->maxValue()
//                         : +std::numeric_limits<float>::infinity();
//   auto options = ml::webnn::mojom::blink::ClampOptions::New();
//   options->minValue = min;
//   options->maxValue = max;

//   return options;
// }

// OperationInfoPtr FusionOperation(const MLOperator* activation) {
//   switch (activation->Kind()) {
//     case MLOperator::OperatorKind::kClamp: {
//       auto clamp = ml::webnn::mojom::blink::Clamp::New();
//       clamp->input_index = std::numeric_limits<uint64_t>::max();
//       clamp->options = BlinkClampOptioinToMojo(
//           static_cast<const MLClampOptions*>(activation->Options()));
//       clamp->output_index = std::numeric_limits<uint64_t>::max();
//       auto operation = OperationInfo::NewClamp(std::move(clamp));
//       return operation;
//     }
//     case MLOperator::OperatorKind::kRelu: {
//       auto relu = ml::webnn::mojom::blink::Relu::New();
//       relu->input_index = std::numeric_limits<uint64_t>::max();
//       relu->output_index = std::numeric_limits<uint64_t>::max();
//       auto operation = OperationInfo::NewRelu(std::move(relu));
//       return operation;
//     }
//     default: {
//       NOTREACHED();
//       return nullptr;
//     }
//   }
// }

// ml::webnn::mojom::blink::Conv2dOptionsPtr BlinkConv2dOptioinToMojo(
//     const MLConv2dOptions* ml_options,
//     const HeapHashMap<Member<const MLOperand>, size_t>& operand_index_map) {
//   auto options = ml::webnn::mojom::blink::Conv2dOptions::New();
//   options->padding =
//       ml_options->hasPadding() ? ml_options->padding() : Vector<int32_t>(4,
//       0);
//   options->strides = ml_options->hasStrides()
//                          ? ml_options->strides()
//                          : Vector<int32_t>(2, 1u);
//   options->dilations = ml_options->hasDilations()
//                            ? ml_options->dilations()
//                            : Vector<int32_t>(2, 1u);
//   options->auto_pad = BlinkAutoPadToMojo(ml_options->autoPad().AsEnum());
//   options->groups = ml_options->groups();
//   options->inputLayout =
//       BlinkInputOperandLayoutToMojo(ml_options->inputLayout().AsEnum());
//   options->filterLayout =
//       BlinkConv2dFilterOperandLayoutToMojo(ml_options->filterLayout().AsEnum());
//   options->bias_index = ml_options->hasBias()
//                             ? operand_index_map.at(ml_options->bias())
//                             : std::numeric_limits<uint64_t>::max();
//   if (ml_options->hasActivation()) {
//     options->activation = FusionOperation(ml_options->activation());
//   }

//   return options;
// }

// ml::webnn::mojom::blink::Pool2dOptionsPtr BlinkPool2dOptioinToMojo(
//     const MLPool2dOptions* ml_options) {
//   auto options = ml::webnn::mojom::blink::Pool2dOptions::New();
//   options->window_dimensions = ml_options->hasWindowDimensions()
//                                    ? ml_options->windowDimensions()
//                                    : Vector<int32_t>();
//   options->padding =
//       ml_options->hasPadding() ? ml_options->padding() : Vector<int32_t>(4,
//       0);
//   options->strides = ml_options->hasStrides()
//                          ? ml_options->strides()
//                          : Vector<int32_t>(2, 1u);
//   options->dilations = ml_options->hasDilations()
//                            ? ml_options->dilations()
//                            : Vector<int32_t>(2, 1u);
//   options->auto_pad = BlinkAutoPadToMojo(ml_options->autoPad().AsEnum());
//   options->layout =
//       BlinkInputOperandLayoutToMojo(ml_options->layout().AsEnum());
//   options->rounding_type =
//       BlinkRoundingTypeToMojo(ml_options->roundingType().AsEnum());
//   options->output_sizes = ml_options->hasOutputSizes()
//                               ? ml_options->outputSizes()
//                               : Vector<int32_t>();
//   return options;
// }

// ml::webnn::mojom::blink::GemmOptionsPtr BlinkGemmOptioinToMojo(
//     const MLGemmOptions* ml_options,
//     const HeapHashMap<Member<const MLOperand>, size_t>& operand_index_map) {
//   auto options = ml::webnn::mojom::blink::GemmOptions::New();
//   options->c_index = ml_options->hasC() ?
//   operand_index_map.at(ml_options->c())
//                                         :
//                                         std::numeric_limits<uint64_t>::max();
//   options->alpha = ml_options->hasAlpha() ? ml_options->alpha() : 1.0;
//   options->beta = ml_options->hasBeta() ? ml_options->beta() : 1.0;
//   options->a_transpose =
//       ml_options->hasATranspose() ? ml_options->aTranspose() : false;
//   options->b_transpose =
//       ml_options->hasBTranspose() ? ml_options->bTranspose() : false;
//   return options;
// }

}  // namespace

TFLiteModelInfo::TFLiteModelInfo() {
  // Initialize a empty buffer to the model.
  buffers_.push_back(tflite::CreateBuffer(builder_, builder_.CreateVector({})));
}

TFLiteModelInfo::~TFLiteModelInfo() = default;

void TFLiteModelInfo::Trace(Visitor* visitor) const {
  visitor->Trace(operand_index_map_);
}

void TFLiteModelInfo::BuildBuffer(const MLOperand* constant) {
  auto* array_buffer_view = constant->ArrayBufferView();
  buffers_.push_back(tflite::CreateBuffer(
      builder_,
      builder_.CreateVector(reinterpret_cast<const uint8_t*>(
                                array_buffer_view->BaseAddressMaybeShared()),
                            array_buffer_view->byteLength())));
  uint32_t buffer_index = static_cast<uint32_t>(buffers_.size()) - 1;
  BuildTensor(constant, buffer_index);
}

int32_t TFLiteModelInfo::BuildTensor(const MLOperand* operand,
                                     uint32_t buffer_index) {
  // Create `Tensor` with the index of buffer. this buffer index is 0 for
  // intermediate output operand because there is no data buffer associated.
  // TODO: convert TensorType_FLOAT32 from WebNN Spec.
  LOG(ERROR) << "========== BuildTensor buffer_index " << buffer_index;
  tensors_.emplace_back(tflite::CreateTensor(
      builder_,
      builder_.CreateVector<int32_t>(ConvertType(operand->Dimensions())),
      tflite::TensorType_FLOAT32, buffer_index));
  // The index of buffer is used to identify constant operand,  the index of
  // tensor is used to create `Operator` and `SubGraph`. each operation generate
  // a output operand that will be inserted in a hash map with the MLOperand and
  // index, the index is incremented by one.
  int32_t tensor_index = static_cast<int>(tensors_.size()) - 1;
  LOG(ERROR) << "========== BuildTensor tensor_index " << tensor_index;
  operand_index_map_.insert(operand, tensor_index);
  return tensor_index;
}

void TFLiteModelInfo::BuildOperator(const MLOperator* op) {
  switch (op->Kind()) {
    case MLOperator::OperatorKind::kClamp:
      //   model_info->AddClamp(op);
      break;
    case MLOperator::OperatorKind::kConv2d:
      //   model_info->AddConv2d(op);
      break;
    case MLOperator::OperatorKind::kAdd:
    case MLOperator::OperatorKind::kSub:
    case MLOperator::OperatorKind::kMul:
    case MLOperator::OperatorKind::kDiv:
    case MLOperator::OperatorKind::kMin:
    case MLOperator::OperatorKind::kMax:
      AddElementWiseBinary(op);
      break;
    case MLOperator::OperatorKind::kGemm:
      //   model_info->AddGemm(op);
      break;
    case MLOperator::OperatorKind::kAveragePool2d:
    case MLOperator::OperatorKind::kMaxPool2d:
      //   model_info->AddPool2d(op);
      break;
    case MLOperator::OperatorKind::kRelu:
      //   model_info->AddRelu(op);
      break;
    case MLOperator::OperatorKind::kSoftmax:
      // model_info->AddSoftmax(op);
      break;
    case MLOperator::OperatorKind::kReshape:
      // model_info->AddReshape(op);
      break;
    case MLOperator::OperatorKind::kHardSwish:
      NOTIMPLEMENTED();
      break;
  }
}

// void TFLiteModelInfo::AddClamp(const MLOperator* ml_clamp) {
//   DCHECK_EQ(ml_clamp->Inputs().size(), 1u);
//   auto* input = ml_clamp->Inputs()[0].Get();
//   if (operand_index_map_.find(input) == operand_index_map_.end()) {
//     return;
//   }
//   DCHECK_EQ(ml_clamp->Outputs().size(), 1u);
//   auto* output = ml_clamp->Outputs()[0].Get();
//   DCHECK(operand_index_map_.find(output) == operand_index_map_.end());
//   // Add operand descriptor to the model.
//   size_t output_index = BuildTensor(output);
//   // Add clamp operation to the model.
//   auto clamp = ml::webnn::mojom::blink::Clamp::New();
//   clamp->input_index = operand_index_map_.at(input);
//   clamp->options = BlinkClampOptioinToMojo(
//       static_cast<const MLClampOptions*>(ml_clamp->Options()));
//   clamp->output_index = output_index;
//   auto operation = OperationInfo::NewClamp(std::move(clamp));
//   model_info_->operations.push_back(std::move(operation));
// }

// void TFLiteModelInfo::AddConv2d(const MLOperator* ml_conv2d) {
//   DCHECK_GE(ml_conv2d->Inputs().size(), 2u);
//   auto* input = ml_conv2d->Inputs()[0].Get();
//   auto* filter = ml_conv2d->Inputs()[1].Get();
//   if (operand_index_map_.find(input) == operand_index_map_.end() ||
//       operand_index_map_.find(filter) == operand_index_map_.end()) {
//     return;
//   }
//   // Add operand descriptor to the model.
//   DCHECK_EQ(ml_conv2d->Outputs().size(), 1u);
//   auto* output = ml_conv2d->Outputs()[0].Get();
//   DCHECK(operand_index_map_.find(output) == operand_index_map_.end());
//   size_t output_index = BuildTensor(output);
//   // Add clamp operation to the model.
//   auto conv2d = ml::webnn::mojom::blink::Conv2d::New();
//   conv2d->input_index = operand_index_map_.at(input);
//   conv2d->filter_index = operand_index_map_.at(filter);
//   const MLConv2dOptions* ml_options =
//       static_cast<const MLConv2dOptions*>(ml_conv2d->Options());
//   conv2d->options = BlinkConv2dOptioinToMojo(ml_options, operand_index_map_);
//   conv2d->output_index = output_index;
//   auto operation = OperationInfo::NewConv2d(std::move(conv2d));
//   model_info_->operations.push_back(std::move(operation));
// }

void TFLiteModelInfo::AddElementWiseBinary(const MLOperator* ml_binary) {
  DCHECK_EQ(ml_binary->Inputs().size(), 2u);
  auto* a = ml_binary->Inputs()[0].Get();
  auto* b = ml_binary->Inputs()[1].Get();
  if (operand_index_map_.find(a) == operand_index_map_.end() ||
      operand_index_map_.find(b) == operand_index_map_.end()) {
    return;
  }
  DCHECK_EQ(ml_binary->Outputs().size(), 1u);
  auto* output = ml_binary->Outputs()[0].Get();
  DCHECK(operand_index_map_.find(output) == operand_index_map_.end());
  tflite::BuiltinOperator binary_op;
  switch (ml_binary->Kind()) {
    case MLOperator::OperatorKind::kAdd:
      binary_op = tflite::BuiltinOperator_ADD;
      break;
    default:
      NOTREACHED();
      return;
  }

  operator_codes_.push_back(tflite::CreateOperatorCode(builder_, binary_op));
  BuildTensor(output);
  // TF-Lite support activation in elementwise binary operations, but WebNN Spec
  // doesn't define the feature, so the options of elementwise binary doesn't
  // need to be configured.
  tflite::BuiltinOptions builtin_options_type = tflite::BuiltinOptions_NONE;
  flatbuffers::Offset<void> builtin_options = 0;

  const std::array<int32_t, 2> op_inputs{
      {operand_index_map_.at(a), operand_index_map_.at(b)}};
  const std::array<int32_t, 1> op_outputs{{operand_index_map_.at(output)}};
  operators_.emplace_back(tflite::CreateOperator(
      builder_, /*opcode_index=*/0, builder_.CreateVector<int32_t>(op_inputs),
      builder_.CreateVector<int32_t>(op_outputs), builtin_options_type,
      builtin_options));
}

// void TFLiteModelInfo::AddGemm(const MLOperator* ml_gemm) {
//   DCHECK_GE(ml_gemm->Inputs().size(), 2u);
//   auto* a = ml_gemm->Inputs()[0].Get();
//   auto* b = ml_gemm->Inputs()[1].Get();
//   if (operand_index_map_.find(a) == operand_index_map_.end() ||
//       operand_index_map_.find(b) == operand_index_map_.end()) {
//     return;
//   }
//   // Add operand descriptor to the model.
//   DCHECK_EQ(ml_gemm->Outputs().size(), 1u);
//   auto* output = ml_gemm->Outputs()[0].Get();
//   DCHECK(operand_index_map_.find(output) == operand_index_map_.end());
//   size_t output_index = BuildTensor(output);
//   // Add clamp operation to the model.
//   auto gemm = ml::webnn::mojom::blink::Gemm::New();
//   gemm->a_index = operand_index_map_.at(a);
//   gemm->b_index = operand_index_map_.at(b);
//   const MLGemmOptions* ml_options =
//       static_cast<const MLGemmOptions*>(ml_gemm->Options());
//   gemm->options = BlinkGemmOptioinToMojo(ml_options, operand_index_map_);
//   gemm->output_index = output_index;
//   auto operation = OperationInfo::NewGemm(std::move(gemm));
//   model_info_->operations.push_back(std::move(operation));
// }

// void TFLiteModelInfo::AddPool2d(const MLOperator* ml_pool2d) {
//   DCHECK_EQ(ml_pool2d->Inputs().size(), 1u);
//   auto* input = ml_pool2d->Inputs()[0].Get();
//   if (operand_index_map_.find(input) == operand_index_map_.end()) {
//     return;
//   }
//   // Add operand descriptor to the model.
//   DCHECK_EQ(ml_pool2d->Outputs().size(), 1u);
//   auto* output = ml_pool2d->Outputs()[0].Get();
//   DCHECK(operand_index_map_.find(output) == operand_index_map_.end());
//   size_t output_index = BuildTensor(output);
//   // Add averagePool2d operation to the model.
//   auto pool2d = ml::webnn::mojom::blink::Pool2d::New();
//   pool2d->type = BlinkPool2dTypeToMojo(ml_pool2d->Kind());
//   pool2d->input_index = operand_index_map_.at(input);
//   const MLPool2dOptions* ml_options =
//       static_cast<const MLPool2dOptions*>(ml_pool2d->Options());
//   pool2d->options = BlinkPool2dOptioinToMojo(ml_options);
//   pool2d->output_index = output_index;
//   auto operation = OperationInfo::NewPool2d(std::move(pool2d));
//   model_info_->operations.push_back(std::move(operation));
// }

// void TFLiteModelInfo::AddRelu(const MLOperator* ml_relu) {
//   DCHECK_EQ(ml_relu->Inputs().size(), 1u);
//   auto* input = ml_relu->Inputs()[0].Get();
//   if (operand_index_map_.find(input) == operand_index_map_.end()) {
//     return;
//   }
//   // Add operand descriptor to the model.
//   DCHECK_EQ(ml_relu->Outputs().size(), 1u);
//   auto* output = ml_relu->Outputs()[0].Get();
//   DCHECK(operand_index_map_.find(output) == operand_index_map_.end());
//   size_t output_index = BuildTensor(output);
//   // Add averagePool2d operation to the model.
//   auto relu = ml::webnn::mojom::blink::Relu::New();
//   relu->input_index = operand_index_map_.at(input);
//   relu->output_index = output_index;
//   auto operation = OperationInfo::NewRelu(std::move(relu));
//   model_info_->operations.push_back(std::move(operation));
// }

// void TFLiteModelInfo::AddReshape(const MLOperator* ml_reshape) {
//   DCHECK_EQ(ml_reshape->Inputs().size(), 1u);
//   auto* input = ml_reshape->Inputs()[0].Get();
//   if (operand_index_map_.find(input) == operand_index_map_.end()) {
//     return;
//   }
//   // Add operand descriptor to the model.
//   DCHECK_EQ(ml_reshape->Outputs().size(), 1u);
//   auto* output = ml_reshape->Outputs()[0].Get();
//   DCHECK(operand_index_map_.find(output) == operand_index_map_.end());
//   size_t output_index = BuildTensor(output);
//   // Add averagePool2d operation to the model.
//   auto reshape = ml::webnn::mojom::blink::Reshape::New();
//   reshape->input_index = operand_index_map_.at(input);
//   reshape->output_index = output_index;
//   auto operation = OperationInfo::NewReshape(std::move(reshape));
//   model_info_->operations.push_back(std::move(operation));
// }

// void TFLiteModelInfo::AddSoftmax(const MLOperator* ml_softmax) {
//   DCHECK_EQ(ml_softmax->Inputs().size(), 1u);
//   auto* input = ml_softmax->Inputs()[0].Get();
//   if (operand_index_map_.find(input) == operand_index_map_.end()) {
//     return;
//   }
//   // Add operand descriptor to the model.
//   DCHECK_EQ(ml_softmax->Outputs().size(), 1u);
//   auto* output = ml_softmax->Outputs()[0].Get();
//   DCHECK(operand_index_map_.find(output) == operand_index_map_.end());
//   size_t output_index = BuildTensor(output);
//   // Add averagePool2d operation to the model.
//   auto softmax = ml::webnn::mojom::blink::Softmax::New();
//   softmax->input_index = operand_index_map_.at(input);
//   softmax->output_index = output_index;
//   auto operation = OperationInfo::NewSoftmax(std::move(softmax));
//   model_info_->operations.push_back(std::move(operation));
// }

void TFLiteModelInfo::BuildModel(const std::vector<int32_t>& subgraph_inputs,
                                 const std::vector<int32_t>& subgraph_outputs) {
  flatbuffers::Offset<tflite::SubGraph> subgraph =
      tflite::CreateSubGraph(builder_, builder_.CreateVector(tensors_),
                             builder_.CreateVector<int32_t>(subgraph_inputs),
                             builder_.CreateVector<int32_t>(subgraph_outputs),
                             builder_.CreateVector(operators_));

  flatbuffers::Offset<flatbuffers::String> description =
      builder_.CreateString("TF-Lite model");

  flatbuffers::Offset<tflite::Model> model_buffer = tflite::CreateModel(
      builder_, TFLITE_SCHEMA_VERSION, builder_.CreateVector(operator_codes_),
      builder_.CreateVector(&subgraph, 1), description,
      builder_.CreateVector(buffers_));

  tflite::FinishModelBuffer(builder_, model_buffer);
}

int32_t TFLiteModelInfo::GetTensorIndex(const MLOperand* operand) {
  DCHECK(operand_index_map_.find(operand) != operand_index_map_.end());
  return operand_index_map_.at(operand);
}

flatbuffers::FlatBufferBuilder& TFLiteModelInfo::GetFlatBufferBuilder() {
  return builder_;
}

}  // namespace blink
