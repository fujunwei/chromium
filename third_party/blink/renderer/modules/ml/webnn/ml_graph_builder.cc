// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_builder.h"

#include "base/numerics/checked_math.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise_resolver.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_clamp_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_conv_2d_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_gemm_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_operand_descriptor.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_pool_2d_options.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"
#include "third_party/blink/renderer/modules/ml/ml_context.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operand.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operator.h"
#include "third_party/blink/renderer/modules/ml/webnn/mojo_context.h"
#include "third_party/blink/renderer/modules/ml/webnn/mojo_graph.h"
#include "third_party/blink/renderer/platform/bindings/exception_state.h"
#include "third_party/blink/renderer/platform/heap/collection_support/heap_deque.h"
#include "third_party/blink/renderer/platform/heap/collection_support/heap_hash_set.h"

namespace blink {

namespace {

bool BroadcastShape(const Vector<int32_t>& dims_input0,
                    const Vector<int32_t>& dims_input1,
                    Vector<int32_t>& dims_output,
                    wtf_size_t skip_axes = 0) {
  // The rank of the output tensor is the maximum rank of the input tensors.
  auto rank_a = dims_input0.size(), rank_b = dims_input1.size();
  auto rank_c = rank_a >= rank_b ? rank_a : rank_b;
  dims_output.resize(rank_c);
  if (rank_a < skip_axes || rank_b < skip_axes) {
    return false;
  }
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

bool CalculatePaddingForAutoPad(V8MLAutoPad::Enum autoPad,
                                base::CheckedNumeric<int32_t> input_size,
                                base::CheckedNumeric<int32_t> filter_size,
                                base::CheckedNumeric<int32_t> stride,
                                base::CheckedNumeric<int32_t> dilation,
                                int32_t& padding_begin,
                                int32_t& padding_end) {
  auto output_size = (input_size + stride - 1) / stride;
  auto dilated_filter_size = (filter_size - 1) * dilation + 1;
  auto needed_input_size = (output_size - 1) * stride + dilated_filter_size;
  if (!needed_input_size.IsValid() || !input_size.IsValid()) {
    return false;
  }
  auto total_padding = needed_input_size.ValueOrDie() > input_size.ValueOrDie()
                           ? needed_input_size - input_size
                           : base::MakeCheckedNum(0);
  base::CheckedNumeric<int32_t> checked_padding_begin(0),
      checked_padding_end(0);
  switch (autoPad) {
    case V8MLAutoPad::Enum::kSameUpper:
      checked_padding_begin = total_padding / 2;
      checked_padding_end = (total_padding + 1) / 2;
      break;
    case V8MLAutoPad::Enum::kSameLower:
      checked_padding_begin = (total_padding + 1) / 2;
      checked_padding_end = total_padding / 2;
      break;
    default:
      NOTREACHED();
  }
  if (!checked_padding_begin.AssignIfValid(&padding_begin) ||
      !checked_padding_end.AssignIfValid(&padding_end)) {
    return false;
  }
  return true;
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

MLContext* MLGraphBuilder::GetContext() const {
  return ml_context_.Get();
}

MLOperand* MLGraphBuilder::input(String name,
                                 const MLOperandDescriptor* desc,
                                 ExceptionState& exception_state) {
  auto* input =
      MakeGarbageCollected<MLOperand>(this, MLOperand::KindEnum::kInput);
  input->SetName(std::move(name));
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
  clamp->SetInputs({input});
  clamp->SetOptions(options);
  auto* output = MakeGarbageCollected<MLOperand>(this);
  output->SetType(input->Type());
  output->SetDimensions(input->Dimensions());
  output->SetOperator(clamp);
  clamp->SetOutputs({output});
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
  auto input_shape = input->Dimensions();
  if (input_shape.size() != 4) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "The input is not a 4-D tensor.");
    return nullptr;
  }
  auto filter_shape = filter->Dimensions();
  if (filter_shape.size() != 4) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "The filter is not a 4-D tensor.");
    return nullptr;
  }
  if (input->Type() != filter->Type()) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kDataError,
        "The filter type is not consistent with input.");
    return nullptr;
  }
  if (options->hasPadding() && options->padding().size() != 4) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "The length of padding is not 4.");
    return nullptr;
  }
  auto padding = options->getPaddingOr({0, 0, 0, 0});
  if (options->hasStrides() && options->strides().size() != 2) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "The length of strides is not 2.");
    return nullptr;
  }
  auto strides = options->getStridesOr({1, 1});
  if (options->hasDilations() && options->dilations().size() != 2) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "The length of dilations is not 2.");
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
  if (input_channels % options->groups() != 0 ||
      filter_depth_in != input_channels / options->groups()) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kDataError,
        "The groups is invalid, it must evenly divide the input channels to "
        "filter input depth.");
    return nullptr;
  }
  if (options->hasBias()) {
    auto shape_bias = options->bias()->Dimensions();
    if (shape_bias.size() != 1) {
      exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                        "The bias is not a 1-D tensor.");
      return nullptr;
    }
    if (shape_bias[0] != output_channels) {
      exception_state.ThrowDOMException(
          DOMExceptionCode::kDataError,
          "The bias shape is not [output_channels].");
      return nullptr;
    }
    if (options->bias()->Type() != input->Type()) {
      exception_state.ThrowDOMException(
          DOMExceptionCode::kDataError,
          "The bias type is not consistent with input.");
      return nullptr;
    }
  }

  // Calculate output shape
  int32_t padding_begin_height = padding[0], padding_end_height = padding[1],
          padding_begin_width = padding[2], padding_end_width = padding[3];
  int32_t stride_height = strides[0], stride_width = strides[1];
  int32_t dilation_height = dilations[0], dilation_width = dilations[1];
  if (options->autoPad().AsEnum() != V8MLAutoPad::Enum::kExplicit) {
    if (!CalculatePaddingForAutoPad(options->autoPad().AsEnum(), input_height,
                                    filter_height, stride_height,
                                    dilation_height, padding_begin_height,
                                    padding_end_height)) {
      exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                        "Overflow occurred when calculating "
                                        "padding along the height dimension.");
      return nullptr;
    }
    if (!CalculatePaddingForAutoPad(options->autoPad().AsEnum(), input_width,
                                    filter_width, stride_width, dilation_width,
                                    padding_begin_width, padding_end_width)) {
      exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                        "Overflow occurred when calculating "
                                        "padding along the width dimension.");
      return nullptr;
    }
  }

  base::CheckedNumeric<int32_t> checked_filter_height(filter_height),
      checked_filter_width(filter_width);
  auto dilated_filter_height =
      (checked_filter_height - 1) * dilation_height + 1;
  auto dilated_filter_width = (checked_filter_width - 1) * dilation_width + 1;
  if (!dilated_filter_height.IsValid()) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kDataError,
        "Overflow occurred when calculating dilated filter height.");
    return nullptr;
  }
  if (!dilated_filter_width.IsValid()) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kDataError,
        "Overflow occurred when calculating dilated filter width.");
    return nullptr;
  }
  base::CheckedNumeric<int32_t> checked_input_height(input_height),
      checked_input_width(input_width);
  auto checked_output_height = (checked_input_height - dilated_filter_height +
                                padding_begin_height + padding_end_height) /
                                   stride_height +
                               1;
  auto checked_output_width = (checked_input_width - dilated_filter_width +
                               padding_begin_width + padding_end_width) /
                                  stride_width +
                              1;
  int32_t output_height, output_width;
  if (!checked_output_height.AssignIfValid(&output_height)) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kDataError,
        "Overflow occurred when calculating output height.");
    return nullptr;
  }
  if (!checked_output_width.AssignIfValid(&output_width)) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kDataError,
        "Overflow occurred when calculating output width.");
    return nullptr;
  }
  Vector<int32_t> output_shape;
  if (nchw) {
    output_shape = {batch_size, output_channels, output_height, output_width};
  } else {
    output_shape = {batch_size, output_height, output_width, output_channels};
  }
  auto* conv2d =
      MakeGarbageCollected<MLOperator>(this, MLOperator::OpKind::kConv2d);
  HeapVector<Member<const MLOperand>> inputs;
  inputs.push_back(input);
  inputs.push_back(filter);
  if (options->hasBias()) {
    inputs.push_back(options->bias());
  }
  conv2d->SetInputs(std::move(inputs));
  conv2d->SetOptions(options);
  auto* output = MakeGarbageCollected<MLOperand>(this);
  output->SetType(input->Type());
  output->SetDimensions(std::move(output_shape));
  output->SetOperator(conv2d);
  conv2d->SetOutputs({output});
  return output;
}

MLOperand* MLGraphBuilder::gemm(const MLOperand* a,
                                const MLOperand* b,
                                const MLGemmOptions* options,
                                ExceptionState& exception_state) {
  if (a->Type() != b->Type()) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kDataError,
        "The types of input a and b are inconsistent.");
    return nullptr;
  }
  // The first input 2-D tensor with shape [M, K] if aTranspose is false, or [K,
  // M] if aTranspose is true. The second input 2-D tensor with shape [K, N] if
  // bTranspose is false, or [N, K] if bTranspose is true.
  auto shape_a = a->Dimensions();
  if (shape_a.size() != 2) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "The input a is not a 2-D tensor.");
    return nullptr;
  }
  auto shape_b = b->Dimensions();
  if (shape_b.size() != 2) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "The input b is not a 2-D tensor.");
    return nullptr;
  }
  bool is_valid_shape = (options->aTranspose() ? shape_a[0] : shape_a[1]) ==
                        (options->bTranspose() ? shape_b[1] : shape_b[0]);
  if (!is_valid_shape) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kDataError,
        "The shapes of input a and b are invalid for matrix multiplication.");
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
  HeapVector<Member<const MLOperand>> inputs;
  inputs.push_back(a);
  inputs.push_back(b);
  if (options->hasC()) {
    inputs.push_back(options->c());
  }
  gemm->SetInputs(std::move(inputs));
  gemm->SetOptions(options);
  auto* output = MakeGarbageCollected<MLOperand>(this);
  output->SetType(a->Type());
  output->SetDimensions(std::move(shape_output));
  output->SetOperator(gemm);
  gemm->SetOutputs({output});
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
                                        "The new shape is invalid.");
      return nullptr;
    } else if (i == -1) {
      if (has_minus1) {
        exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                          "The new shape is invalid.");
        return nullptr;
      }
      has_minus1 = true;
    }
  }
  auto input_shape = input->Dimensions();
  uint32_t input_size = 1;
  for (auto dim : input_shape) {
    input_size *= dim;
  }
  uint32_t capacity = 1;
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
    if (input_size % capacity != 0) {
      exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                        "The new shape is invalid.");
      return nullptr;
    }
    output_shape[minus1_dim_idx] = input_size / capacity;
  } else {
    // The number of elements implied by newShape must be the same as the number
    // of elements in the input tensor.
    if (input_size != capacity) {
      exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                        "The new shape is invalid.");
      return nullptr;
    }
  }
  auto* reshape =
      MakeGarbageCollected<MLOperator>(this, MLOperator::OpKind::kReshape);
  reshape->SetInputs({input});
  auto* output = MakeGarbageCollected<MLOperand>(this);
  output->SetType(input->Type());
  output->SetDimensions(std::move(output_shape));
  output->SetOperator(reshape);
  reshape->SetOutputs({output});
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
    exception_state.ThrowDOMException(
        DOMExceptionCode::kDataError,
        "The types of input a and b are inconsistent.");
    return nullptr;
  }
  Vector<int32_t> dims_output;
  if (!BroadcastShape(a->Dimensions(), b->Dimensions(), dims_output)) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kDataError,
        "The shapes of input a and b are not broadcast compatible.");
    return nullptr;
  }
  auto* binary = MakeGarbageCollected<MLOperator>(this, kind);
  binary->SetInputs({a, b});
  auto* c = MakeGarbageCollected<MLOperand>(this);
  c->SetType(a->Type());
  c->SetDimensions(std::move(dims_output));
  c->SetOperator(binary);
  binary->SetOutputs({c});
  return c;
}

MLOperand* MLGraphBuilder::BuildElementWiseUnary(
    MLOperator::OpKind kind,
    const MLOperand* input,
    ExceptionState& exception_state) {
  auto* unary = MakeGarbageCollected<MLOperator>(this, kind);
  unary->SetInputs({input});
  auto* output = MakeGarbageCollected<MLOperand>(this);
  output->SetType(input->Type());
  output->SetDimensions(input->Dimensions());
  output->SetOperator(unary);
  unary->SetOutputs({output});
  return output;
}

MLOperand* MLGraphBuilder::BuildPool2d(MLOperator::OpKind kind,
                                       const MLOperand* input,
                                       const MLPool2dOptions* options,
                                       ExceptionState& exception_state) {
  auto input_shape = input->Dimensions();
  if (input_shape.size() != 4) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "The input is not a 4-D tensor.");
    return nullptr;
  }
  if (options->hasWindowDimensions() &&
      options->windowDimensions().size() != 2) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kDataError,
        "The length of windowDimensions is not 2.");
    return nullptr;
  }
  if (options->hasOutputSizes() && options->outputSizes().size() != 2) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "The length of outputSizes is not 2.");
    return nullptr;
  }
  if (options->hasPadding() && options->padding().size() != 4) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "The length of padding is not 4.");
    return nullptr;
  }
  auto padding = options->getPaddingOr({0, 0, 0, 0});
  if (options->hasStrides() && options->strides().size() != 2) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "The length of strides is not 2.");
    return nullptr;
  }
  auto strides = options->getStridesOr({1, 1});
  if (options->hasDilations() && options->dilations().size() != 2) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "The length of dilations is not 2.");
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
    if (!CalculatePaddingForAutoPad(options->autoPad().AsEnum(), input_height,
                                    window_height, stride_height,
                                    dilation_height, padding_begin_height,
                                    padding_end_height)) {
      exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                        "Overflow occurred when calculating "
                                        "padding along the height dimension.");
      return nullptr;
    }
    if (!CalculatePaddingForAutoPad(options->autoPad().AsEnum(), input_width,
                                    window_width, stride_width, dilation_width,
                                    padding_begin_width, padding_end_width)) {
      exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                        "Overflow occurred when calculating "
                                        "padding along the width dimension.");
      return nullptr;
    }
  }

  base::CheckedNumeric<int32_t> checked_window_height(window_height),
      checked_window_width(window_width);
  auto dilated_window_height =
      (checked_window_height - 1) * dilation_height + 1;
  auto dilated_window_width = (checked_window_width - 1) * dilation_width + 1;
  if (!dilated_window_height.IsValid()) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kDataError,
        "Overflow occurred when calculating dilated window height.");
    return nullptr;
  }
  if (!dilated_window_width.IsValid()) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kDataError,
        "Overflow occurred when calculating dilated window width.");
    return nullptr;
  }
  base::CheckedNumeric<float> checked_input_height(input_height),
      checked_input_width(input_width);
  auto checked_output_height = (checked_input_height - dilated_window_height +
                                padding_begin_height + padding_end_height) /
                                   stride_height +
                               1.0;
  auto checked_output_width = (checked_input_width - dilated_window_width +
                               padding_begin_width + padding_end_width) /
                                  stride_width +
                              1.0;
  float float_output_height, float_output_width;
  if (!checked_output_height.AssignIfValid(&float_output_height)) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kDataError,
        "Overflow occurred when calculating output height.");
    return nullptr;
  }
  if (!checked_output_width.AssignIfValid(&float_output_width)) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kDataError,
        "Overflow occurred when calculating output width.");
    return nullptr;
  }
  int32_t output_height, output_width;
  if (!options->hasOutputSizes()) {
    output_height =
        options->roundingType().AsEnum() == V8MLRoundingType::Enum::kFloor
            ? base::ClampFloor<int32_t>(float_output_height)
            : base::ClampCeil<int32_t>(float_output_height);
    output_width =
        options->roundingType().AsEnum() == V8MLRoundingType::Enum::kFloor
            ? base::ClampFloor(float_output_width)
            : base::ClampCeil(float_output_width);
  } else {
    if ((options->outputSizes()[0] ==
             base::ClampFloor<int32_t>(float_output_height) &&
         options->outputSizes()[1] == base::ClampFloor(float_output_width)) ||
        (options->outputSizes()[0] ==
             base::ClampCeil<int32_t>(float_output_height) &&
         options->outputSizes()[1] == base::ClampCeil(float_output_width))) {
      output_height = options->outputSizes()[0];
      output_width = options->outputSizes()[1];
    } else {
      exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                        "The output sizes is invalid.");
      return nullptr;
    }
  }

  Vector<int32_t> output_shape;
  if (nchw) {
    output_shape = {batches, channels, output_height, output_width};
  } else {
    output_shape = {batches, output_height, output_width, channels};
  }

  auto* pool2d = MakeGarbageCollected<MLOperator>(this, kind);
  pool2d->SetInputs({input});
  pool2d->SetOptions(options);
  auto* output = MakeGarbageCollected<MLOperand>(this);
  output->SetType(input->Type());
  output->SetDimensions(std::move(output_shape));
  output->SetOperator(pool2d);
  pool2d->SetOutputs({output});
  return output;
}

// static
void MLGraphBuilder::SortOperators(
    const MLNamedOperands& named_outputs,
    HeapVector<Member<const MLOperand>>& inputs,
    HeapVector<Member<const MLOperand>>& constants,
    HeapVector<Member<const MLOperator>>& sorted_operators) {
  HeapDeque<Member<const MLOperator>> operators_to_do;
  HeapHashSet<Member<const MLOperator>> operators_done;
  for (const auto& output : named_outputs) {
    operators_to_do.push_back(output.second->Operator());
  }
  while (operators_to_do.size() > 0) {
    const auto& op = operators_to_do.back();
    if (!operators_done.Contains(op.Get())) {
      bool can_add = true;
      for (const auto& input : op->Inputs()) {
        const auto* dependent_op = input->Operator();
        if (dependent_op && !operators_done.Contains(dependent_op)) {
          // As the dependent operator is not done, skip processing of this
          // operator and push the dependent operator into the to-do stack.
          can_add = false;
          operators_to_do.push_back(dependent_op);
        }
      }
      if (can_add) {
        // All dependent operators are done, process and add it into the
        // done set.
        for (const auto& input : op->Inputs()) {
          if (input->Kind() == MLOperand::KindEnum::kInput) {
            inputs.push_back(input.Get());
          } else if (input->Kind() == MLOperand::KindEnum::kConstant) {
            constants.push_back(input.Get());
          }
        }
        sorted_operators.push_back(op.Get());
        operators_done.insert(op.Get());
        operators_to_do.pop_back();
      }
    } else {
      operators_to_do.pop_back();
    }
  }
}

ScriptPromise MLGraphBuilder::build(ScriptState* script_state,
                                    MLNamedOperands named_outputs,
                                    ExceptionState& exception_state) {
  if (!script_state->ContextIsValid()) {
    exception_state.ThrowDOMException(DOMExceptionCode::kInvalidStateError,
                                      "Invalid script state");
    return ScriptPromise();
  }

  // The Context is GPU device or low power preference, the graph is built by
  // MojoGraph object.
  if (GetContext()->GetDevicePreference() == V8MLDevicePreference::Enum::kGpu) {
    auto* graph = MakeGarbageCollected<MojoGraph>(script_state, ml_context_);
    return graph->BuildImpl(script_state, std::move(named_outputs),
                            exception_state);
  } else {
    NOTIMPLEMENTED();
    exception_state.ThrowDOMException(DOMExceptionCode::kNotSupportedError,
                                      "Not implemented");
    return ScriptPromise();
  }
}

MLGraph* MLGraphBuilder::buildSync(MLNamedOperands named_outputs,
                                   ExceptionState& exception_state) {
  NOTIMPLEMENTED();
  exception_state.ThrowDOMException(DOMExceptionCode::kNotSupportedError,
                                    "Not implemented");
  return nullptr;
}

}  // namespace blink
