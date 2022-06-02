// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_builder.h"

#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_clamp_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_conv_2d_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_gemm_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_operand_descriptor.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_pool_2d_options.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"
#include "third_party/blink/renderer/modules/ml/ml_context.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_xnnpack.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operand.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operator.h"
#include "third_party/blink/renderer/platform/bindings/exception_state.h"
#include "third_party/blink/renderer/platform/heap/collection_support/heap_deque.h"
#include "third_party/blink/renderer/platform/heap/collection_support/heap_hash_set.h"

namespace blink {

namespace {

bool BroadcastShape(Vector<int32_t> dims_input0,
                    Vector<int32_t> dims_input1,
                    Vector<int32_t>& dims_output,
                    wtf_size_t skip_axes = 0) {
  // The rank of the output tensor is the maximum rank of the input tensors.
  auto rank_a = dims_input0.size(), rank_b = dims_input1.size();
  auto rank_c = rank_a >= rank_b ? rank_a : rank_b;
  dims_output.resize(rank_c);
  DCHECK(rank_a >= skip_axes && rank_b >= skip_axes);
  // For each dimension of the output tensor, its size is the maximum size along
  // that dimension of the input tensors.
  for (wtf_size_t i = 0; i < rank_c; ++i) {
    // Skip some axes from the right side when broadcasting.
    if (i >= skip_axes) {
      auto dim_a = i < rank_a ? dims_input0[rank_a - i - 1] : 1;
      auto dim_b = i < rank_b ? dims_input1[rank_b - i - 1] : 1;
      if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
        return false;
      }
      dims_output[rank_c - i - 1] = dim_a > dim_b ? dim_a : dim_b;
    }
  }
  return true;
}

void CalculatePaddingForAutoPad(V8MLAutoPad::Enum autoPad,
                                int32_t input_size,
                                int32_t filter_size,
                                int32_t stride,
                                int32_t dilation,
                                int32_t& padding_begin,
                                int32_t& padding_end) {
  int32_t out_size = (input_size + stride - 1) / stride;
  int32_t dilated_filter = (filter_size - 1) * dilation + 1;
  int32_t needed_input = (out_size - 1) * stride + dilated_filter;
  int32_t total_padding =
      needed_input > input_size ? needed_input - input_size : 0;
  switch (autoPad) {
    case V8MLAutoPad::Enum::kSameUpper:
      padding_begin = total_padding / 2;
      padding_end = (total_padding + 1) / 2;
      break;
    case V8MLAutoPad::Enum::kSameLower:
      padding_begin = (total_padding + 1) / 2;
      padding_end = total_padding / 2;
      break;
    default:
      NOTREACHED();
  }
}

}  // namespace

// static
MLGraphBuilder* MLGraphBuilder::Create(MLContext* context) {
  return MakeGarbageCollected<MLGraphBuilder>(context);
}

MLGraphBuilder::MLGraphBuilder(MLContext* context) : ml_context_(context) {}

MLGraphBuilder::~MLGraphBuilder() = default;

void MLGraphBuilder::Trace(Visitor* visitor) const {
  visitor->Trace(ml_context_);
  ScriptWrappable::Trace(visitor);
}

MLOperand* MLGraphBuilder::input(String name,
                                 const MLOperandDescriptor* desc,
                                 ExceptionState& exception_state) {
  auto* input =
      MakeGarbageCollected<MLOperand>(this, MLOperand::KindEnum::kInput);
  input->SetName(name);
  input->SetType(desc->type().AsEnum());
  if (desc->hasDimensions()) {
    input->SetDimensions(desc->dimensions());
  }
  return input;
}

MLOperand* MLGraphBuilder::constant(const MLOperandDescriptor* desc,
                                    MaybeShared<DOMArrayBufferView> buffer_view,
                                    ExceptionState& exception_state) {
  auto* constant =
      MakeGarbageCollected<MLOperand>(this, MLOperand::KindEnum::kConstant);
  constant->SetType(desc->type().AsEnum());
  if (desc->hasDimensions()) {
    constant->SetDimensions(desc->dimensions());
  }
  constant->SetArrayBufferView(buffer_view.Get());
  return constant;
}

MLOperand* MLGraphBuilder::add(const MLOperand* a,
                               const MLOperand* b,
                               ExceptionState& exception_state) {
  return BuildElementWiseBinary(MLOperator::OpKind::kAdd, a, b,
                                exception_state);
}

MLOperand* MLGraphBuilder::clamp(const MLOperand* input,
                                 const MLClampOptions* options,
                                 ExceptionState& exception_state) {
  auto* clamp =
      MakeGarbageCollected<MLOperator>(this, MLOperator::OpKind::kClamp);
  clamp->Inputs().resize(1);
  clamp->Inputs()[0] = input;
  clamp->SetOptions(options);
  auto* output = MakeGarbageCollected<MLOperand>(this);
  output->SetType(input->Type());
  output->SetDimensions(input->Dimensions());
  output->SetOperator(clamp);
  clamp->Outputs().resize(1);
  clamp->Outputs()[0] = output;
  return output;
}

MLOperator* MLGraphBuilder::clamp(const MLClampOptions* options,
                                  ExceptionState& exception_state) {
  auto* clamp =
      MakeGarbageCollected<MLOperator>(this, MLOperator::OpKind::kClamp);
  clamp->SetOptions(options);
  return clamp;
}

MLOperand* MLGraphBuilder::conv2d(const MLOperand* input,
                                  const MLOperand* filter,
                                  const MLConv2dOptions* options,
                                  ExceptionState& exception_state) {
  // Validate inputs and options
  if (input->Type() != filter->Type()) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "operand type is not consistent");
    return nullptr;
  }
  auto input_shape = input->Dimensions();
  if (input_shape.size() != 4) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "input is not a 4-D tensor");
    return nullptr;
  }
  auto filter_shape = filter->Dimensions();
  if (filter_shape.size() != 4) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "filter is not a 4-D tensor");
    return nullptr;
  }
  if (options->hasBias() && options->bias()->Dimensions().size() != 1) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "bias is not a 1-D tensor");
    return nullptr;
  }
  if (options->hasPadding() && options->padding().size() != 4) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "the length of padding is not 4");
    return nullptr;
  }
  auto padding = options->getPaddingOr({0, 0, 0, 0});
  if (options->hasStrides() && options->strides().size() != 2) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "the length of strides is not 2");
    return nullptr;
  }
  auto strides = options->getStridesOr({1, 1});
  if (options->hasDilations() && options->dilations().size() != 2) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "the length of dilations is not 2");
    return nullptr;
  }
  auto dilations = options->getDilationsOr({1, 1});
  bool nchw = options->inputLayout() == V8MLInputOperandLayout::Enum::kNchw;
  int32_t batch_size = input_shape[0];
  int32_t input_height = nchw ? input_shape[2] : input_shape[1];
  int32_t input_width = nchw ? input_shape[3] : input_shape[2];
  int32_t input_channels = nchw ? input_shape[1] : input_shape[3];
  int32_t filter_height = 0, filter_width = 0, output_channels = 0,
          filter_depth_in = 0;
  switch (options->filterLayout().AsEnum()) {
    case V8MLConv2dFilterOperandLayout::Enum::kHwio:
      filter_height = filter_shape[0];
      filter_width = filter_shape[1];
      output_channels = filter_shape[3];
      filter_depth_in = filter_shape[2];
      break;
    case V8MLConv2dFilterOperandLayout::Enum::kOhwi:
      filter_height = filter_shape[1];
      filter_width = filter_shape[2];
      output_channels = filter_shape[0];
      filter_depth_in = filter_shape[3];
      break;
    case V8MLConv2dFilterOperandLayout::Enum::kIhwo:
      filter_height = filter_shape[1];
      filter_width = filter_shape[2];
      output_channels = filter_shape[3];
      filter_depth_in = filter_shape[0];
      break;
    case V8MLConv2dFilterOperandLayout::Enum::kOihw:
      filter_height = filter_shape[2];
      filter_width = filter_shape[3];
      output_channels = filter_shape[0];
      filter_depth_in = filter_shape[1];
      break;
  }
  if (filter_depth_in != input_channels / options->groups()) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kDataError,
        "The groups is invalid, it must evenly divides the input channels.");
    return nullptr;
  }

  // Calculate output shape
  int32_t padding_begin_height = padding[0], padding_end_height = padding[1],
          padding_begin_width = padding[2], padding_end_width = padding[3];
  int32_t stride_height = strides[0], stride_width = strides[1];
  int32_t dilation_height = dilations[0], dilation_width = dilations[1];
  if (options->autoPad().AsEnum() != V8MLAutoPad::Enum::kExplicit) {
    CalculatePaddingForAutoPad(options->autoPad().AsEnum(), input_height,
                               filter_height, stride_height, dilation_height,
                               padding_begin_height, padding_end_height);
    CalculatePaddingForAutoPad(options->autoPad().AsEnum(), input_width,
                               filter_width, stride_width, dilation_width,
                               padding_begin_width, padding_end_width);
  }

  int32_t dilated_filter_height = dilation_height * (filter_height - 1) + 1;
  int32_t dilated_filter_width = dilation_width * (filter_width - 1) + 1;
  int32_t output_height = 1 + (input_height - dilated_filter_height +
                               padding_begin_height + padding_end_height) /
                                  stride_height;
  int32_t output_width = 1 + (input_width - dilated_filter_width +
                              padding_begin_width + padding_end_width) /
                                 stride_width;
  Vector<int32_t> output_shape;
  if (nchw) {
    output_shape = {batch_size, output_channels, output_height, output_width};
  } else {
    output_shape = {batch_size, output_height, output_width, output_channels};
  }
  auto* conv2d =
      MakeGarbageCollected<MLOperator>(this, MLOperator::OpKind::kConv2d);
  conv2d->Inputs().resize(2);
  conv2d->Inputs()[0] = input;
  conv2d->Inputs()[1] = filter;
  if (options->hasBias()) {
    conv2d->Inputs().push_back(options->bias());
  }
  conv2d->SetOptions(options);
  auto* output = MakeGarbageCollected<MLOperand>(this);
  output->SetType(input->Type());
  output->SetDimensions(std::move(output_shape));
  output->SetOperator(conv2d);
  conv2d->Outputs().resize(1);
  conv2d->Outputs()[0] = output;
  return output;
}

MLOperand* MLGraphBuilder::gemm(const MLOperand* a,
                                const MLOperand* b,
                                const MLGemmOptions* options,
                                ExceptionState& exception_state) {
  if (a->Type() != b->Type()) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "Input types are inconsistent.");
    return nullptr;
  }
  // The first input 2-D tensor with shape [M, K] if aTranspose is false, or [K,
  // M] if aTranspose is true. The second input 2-D tensor with shape [K, N] if
  // bTranspose is false, or [N, K] if bTranspose is true.
  auto shape_a = a->Dimensions();
  if (shape_a.size() != 2) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "input a is not a 2-D tensor");
    return nullptr;
  }
  auto shape_b = b->Dimensions();
  if (shape_b.size() != 2) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "input b is not a 2-D tensor");
    return nullptr;
  }
  bool is_valid_shape = (options->aTranspose() ? shape_a[0] : shape_a[1]) ==
                        (options->bTranspose() ? shape_b[1] : shape_b[0]);
  if (!is_valid_shape) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kDataError,
        "The shape of two inputs are invalid for matrix multiplication.");
    return nullptr;
  }
  Vector<int32_t> shape_output = {
      options->aTranspose() ? shape_a[1] : shape_a[0],
      options->bTranspose() ? shape_b[0] : shape_b[1]};
  // The third input tensor c is either a scalar, or of the shape that is
  // unidirectionally broadcastable to the shape [M, N].
  if (options->hasC()) {
    auto* c = options->c();
    if (c->Type() != a->Type()) {
      exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                        "The type of input c is invalid.");
      return nullptr;
    }
    auto shape_c = options->c()->Dimensions();
    if (shape_c.size() > 2) {
      exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                        "The shape of input c is invalid.");
      return nullptr;
    }

    for (int32_t i = shape_c.size() - 1, j = shape_output.size() - 1;
         i >= 0 && j >= 0; --i, --j) {
      if (shape_c[i] != shape_output[j] && shape_c[i] != 1) {
        exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                          "The shape of input c is invalid.");
        return nullptr;
      }
    }
  }

  auto* gemm =
      MakeGarbageCollected<MLOperator>(this, MLOperator::OpKind::kGemm);
  gemm->Inputs().resize(2);
  gemm->Inputs()[0] = a;
  gemm->Inputs()[1] = b;
  if (options->hasC()) {
    gemm->Inputs().push_back(options->c());
  }
  gemm->SetOptions(options);
  auto* output = MakeGarbageCollected<MLOperand>(this);
  output->SetType(a->Type());
  output->SetDimensions(std::move(shape_output));
  output->SetOperator(gemm);
  gemm->Outputs().resize(1);
  gemm->Outputs()[0] = output;
  return output;
}

MLOperand* MLGraphBuilder::averagePool2d(const MLOperand* input,
                                         const MLPool2dOptions* options,
                                         ExceptionState& exception_state) {
  return BuildPool2d(MLOperator::OpKind::kAveragePool2d, input, options,
                     exception_state);
}

MLOperand* MLGraphBuilder::relu(const MLOperand* input,
                                ExceptionState& exception_state) {
  return BuildElementWiseUnary(MLOperator::OpKind::kRelu, input,
                               exception_state);
}

MLOperator* MLGraphBuilder::relu(ExceptionState& exception_state) {
  return MakeGarbageCollected<MLOperator>(this, MLOperator::OpKind::kRelu);
}

MLOperand* MLGraphBuilder::reshape(const MLOperand* input,
                                   const Vector<int32_t>& new_shape,
                                   ExceptionState& exception_state) {
  bool has_minus1 = false;
  // Only one component of newShape can be the special value of -1
  for (auto i : new_shape) {
    if (i < -1 || i == 0) {
      exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                        "new shape is invalid");
      return nullptr;
    } else if (i == -1) {
      if (has_minus1) {
        exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                          "new shape is invalid");
        return nullptr;
      }
      has_minus1 = true;
    }
  }
  auto input_shape = input->Dimensions();
  uint32_t input_size = 1, capacity = 1;
  for (auto dim : input_shape) {
    input_size *= dim;
  }
  int minus1_dim_idx = -1;
  has_minus1 = false;
  Vector<int32_t> output_shape(new_shape.size());
  for (wtf_size_t i = 0; i < new_shape.size(); ++i) {
    int32_t dim = new_shape[i];
    if (dim == -1) {
      minus1_dim_idx = i;
      has_minus1 = true;
    } else {
      capacity *= dim;
      output_shape[i] = dim;
    }
  }

  // The size of the dimension with the value -1 is computed so that the total
  // size remains constant.
  if (has_minus1) {
    output_shape[minus1_dim_idx] = input_size / capacity;
  } else {
    // The number of elements implied by newShape must be the same as the number
    // of elements in the input tensor.
    if (input_size != capacity) {
      exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                        "new shape is invalid");
      return nullptr;
    }
  }
  auto* reshape =
      MakeGarbageCollected<MLOperator>(this, MLOperator::OpKind::kReshape);
  reshape->Inputs().resize(1);
  reshape->Inputs()[0] = input;
  auto* output = MakeGarbageCollected<MLOperand>(this);
  output->SetType(input->Type());
  output->SetDimensions(std::move(output_shape));
  output->SetOperator(reshape);
  reshape->Outputs().resize(1);
  reshape->Outputs()[0] = output;
  return output;
}

MLOperand* MLGraphBuilder::softmax(const MLOperand* input,
                                   ExceptionState& exception_state) {
  return BuildElementWiseUnary(MLOperator::OpKind::kSoftmax, input,
                               exception_state);
}

MLOperand* MLGraphBuilder::BuildElementWiseBinary(
    MLOperator::OpKind kind,
    const MLOperand* a,
    const MLOperand* b,
    ExceptionState& exception_state) {
  if (a->Type() != b->Type()) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "Input types are inconsistent.");
    return nullptr;
  }
  Vector<int32_t> dims_output;
  if (!BroadcastShape(a->Dimensions(), b->Dimensions(), dims_output)) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kDataError,
        "Input shapes are not broadcast compatible.");
    return nullptr;
  }
  auto* binary = MakeGarbageCollected<MLOperator>(this, kind);
  binary->Inputs().resize(2);
  binary->Inputs()[0] = a;
  binary->Inputs()[1] = b;
  auto* c = MakeGarbageCollected<MLOperand>(this);
  c->SetType(a->Type());
  c->SetDimensions(std::move(dims_output));
  c->SetOperator(binary);
  binary->Outputs().resize(1);
  binary->Outputs()[0] = c;
  return c;
}

MLOperand* MLGraphBuilder::BuildElementWiseUnary(
    MLOperator::OpKind kind,
    const MLOperand* input,
    ExceptionState& exception_state) {
  auto* unary = MakeGarbageCollected<MLOperator>(this, kind);
  unary->Inputs().resize(1);
  unary->Inputs()[0] = input;
  auto* output = MakeGarbageCollected<MLOperand>(this);
  output->SetType(input->Type());
  output->SetDimensions(input->Dimensions());
  output->SetOperator(unary);
  unary->Outputs().resize(1);
  unary->Outputs()[0] = output;
  return output;
}

MLOperand* MLGraphBuilder::BuildPool2d(MLOperator::OpKind kind,
                                       const MLOperand* input,
                                       const MLPool2dOptions* options,
                                       ExceptionState& exception_state) {
  auto input_shape = input->Dimensions();
  if (input_shape.size() != 4) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "input is not a 4-D tensor");
    return nullptr;
  }
  if (options->hasWindowDimensions() &&
      options->windowDimensions().size() != 2) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kDataError,
        "the length of windowDimensions is not 2");
    return nullptr;
  }
  if (options->hasOutputSizes() && options->outputSizes().size() != 2) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "the length of outputSizes is not 2");
    return nullptr;
  }
  if (options->hasPadding() && options->padding().size() != 4) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "the length of padding is not 4");
    return nullptr;
  }
  auto padding = options->getPaddingOr({0, 0, 0, 0});
  if (options->hasStrides() && options->strides().size() != 2) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "the length of strides is not 2");
    return nullptr;
  }
  auto strides = options->getStridesOr({1, 1});
  if (options->hasDilations() && options->dilations().size() != 2) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "the length of dilations is not 2");
    return nullptr;
  }
  auto dilations = options->getDilationsOr({1, 1});
  bool nchw = options->layout() == V8MLInputOperandLayout::Enum::kNchw;
  int32_t batches = input_shape[0];
  int32_t input_height = nchw ? input_shape[2] : input_shape[1];
  int32_t input_width = nchw ? input_shape[3] : input_shape[2];
  int32_t channels = nchw ? input_shape[1] : input_shape[3];
  int32_t window_height = options->hasWindowDimensions()
                              ? options->windowDimensions()[0]
                              : input_height;
  int32_t window_width = options->hasWindowDimensions()
                             ? options->windowDimensions()[1]
                             : input_width;

  int32_t padding_begin_height = padding[0], padding_end_height = padding[1],
          padding_begin_width = padding[2], padding_end_width = padding[3];
  int32_t stride_height = strides[0], stride_width = strides[1];
  int32_t dilation_height = dilations[0], dilation_width = dilations[1];
  if (options->autoPad().AsEnum() != V8MLAutoPad::Enum::kExplicit) {
    CalculatePaddingForAutoPad(options->autoPad().AsEnum(), dilation_height,
                               input_height, window_height, stride_height,
                               padding_begin_height, padding_end_height);
    CalculatePaddingForAutoPad(options->autoPad().AsEnum(), dilation_width,
                               input_width, window_width, stride_width,
                               padding_begin_width, padding_end_width);
  }

  // TODO: We may need to consider dilations when calculating output sizes.
  int32_t output_height, output_width;
  if (!options->hasOutputSizes()) {
    float float_output_height =
        1.0 + static_cast<float>(input_height - window_height +
                                 padding_begin_height + padding_end_height) /
                  static_cast<float>(stride_height);
    float float_output_width =
        1.0 + static_cast<float>(input_width - window_width +
                                 padding_begin_width + padding_end_width) /
                  static_cast<float>(stride_width);
    output_height =
        options->roundingType().AsEnum() == V8MLRoundingType::Enum::kFloor
            ? floor(float_output_height)
            : ceil(float_output_height);
    output_width =
        options->roundingType().AsEnum() == V8MLRoundingType::Enum::kFloor
            ? floor(float_output_width)
            : ceil(float_output_width);
  } else {
    output_height = options->outputSizes()[0];
    output_width = options->outputSizes()[1];
  }

  Vector<int32_t> output_shape;
  if (nchw) {
    output_shape = {batches, channels, output_height, output_width};
  } else {
    output_shape = {batches, output_height, output_width, channels};
  }

  auto* pool2d = MakeGarbageCollected<MLOperator>(this, kind);
  pool2d->Inputs().resize(1);
  pool2d->Inputs()[0] = input;
  pool2d->SetOptions(options);
  auto* output = MakeGarbageCollected<MLOperand>(this);
  output->SetType(input->Type());
  output->SetDimensions(std::move(output_shape));
  output->SetOperator(pool2d);
  pool2d->Outputs().resize(1);
  pool2d->Outputs()[0] = output;
  return output;
}

MLGraph* MLGraphBuilder::build(const MLNamedOperands& named_outputs,
                               ExceptionState& exception_state) {
  HeapVector<Member<const MLOperand>> inputs;
  HeapVector<Member<const MLOperand>> constants;
  HeapVector<Member<const MLOperator>> sorted_operators;

  HeapDeque<Member<const MLOperator>> nodes_to_do;
  HeapHashSet<Member<const MLOperator>> nodes_done;
  for (const auto& output : named_outputs) {
    nodes_to_do.push_back(output.second->Operator());
  }
  while (nodes_to_do.size() > 0) {
    const auto& node = nodes_to_do.back();
    if (!nodes_done.Contains(node.Get())) {
      bool can_add = true;
      for (const auto& input : node->Inputs()) {
        if (input->Operator()) {
          if (!nodes_done.Contains(input->Operator())) {
            can_add = false;
            nodes_to_do.push_back(input->Operator());
          }
        } else {
          if (input->Kind() == MLOperand::KindEnum::kInput) {
            inputs.push_back(input.Get());
          } else {
            DCHECK(input->Kind() == MLOperand::KindEnum::kConstant);
            constants.push_back(input.Get());
          }
        }
      }
      if (can_add) {
        sorted_operators.push_back(node.Get());
        nodes_done.insert(node.Get());
        nodes_to_do.pop_back();
      }
    } else {
      nodes_to_do.pop_back();
    }
  }
  auto* graph = MakeGarbageCollected<MLGraphXnnpack>(ml_context_);
  if (!graph->BuildImpl(named_outputs, inputs, constants, sorted_operators,
                        exception_state)) {
    return nullptr;
  }
  return graph;
}

}  // namespace blink
