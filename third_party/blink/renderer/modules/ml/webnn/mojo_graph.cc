// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/mojo_graph.h"

#include "mojo/public/cpp/bindings/pending_remote.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise_resolver.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_clamp_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_conv_2d_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_gemm_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_operand_descriptor.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_pool_2d_options.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_tensor.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"
#include "third_party/blink/renderer/core/execution_context/execution_context.h"
#include "third_party/blink/renderer/modules/ml/ml.h"
#include "third_party/blink/renderer/modules/ml/ml_context.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operand.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operator.h"
#include "third_party/blink/renderer/modules/ml/webnn/mojo_context.h"
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
using ml::webnn::mojom::blink::MemoryInfoPtr;
using ml::webnn::mojom::blink::ObjectHandle;
using ml::webnn::mojom::blink::ObjectHandlePtr;
using ml::webnn::mojom::blink::OperandType;
using ml::webnn::mojom::blink::Pool2dType;
using ml::webnn::mojom::blink::RoundingType;
using ml::webnn::mojom::blink::UnaryOperandType;

#if BUILDFLAG(IS_WIN)
static const uint32_t kBufferAlignment = 16;
#else
static const uint32_t kBufferAlignment = 1;
#endif
base::CheckedNumeric<size_t> Align(size_t value, uint32_t aligment) {
  size_t remainder = value % aligment;
  if (remainder != 0) {
    value += aligment - remainder;
  }

  return value;
}

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

UnaryOperandType BlinkUnaryOperandTypeToMojo(MLOperator::OperatorKind type) {
  switch (type) {
    case MLOperator::OperatorKind::kRelu:
      return UnaryOperandType::kRelu;
    case MLOperator::OperatorKind::kSoftmax:
      return UnaryOperandType::kSoftmax;
    default:
      NOTREACHED();
      return UnaryOperandType::kUnknown;
  }
}

BinaryOperandType BlinkBinaryOperandTypeToMojo(MLOperator::OperatorKind type) {
  switch (type) {
    case MLOperator::OperatorKind::kAdd:
      return BinaryOperandType::kAdd;
    default:
      NOTREACHED();
      return BinaryOperandType::kUnknown;
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

void AddClamp(const MLOperator* clamp,
              OperandDescriptorPtr output_desc,
              ml::webnn::mojom::blink::Graph* remote_graph) {
  const MLClampOptions* ml_options =
      static_cast<const MLClampOptions*>(clamp->Options());
  auto* input = clamp->Inputs()[0].Get();
  auto input_handle = ObjectHandle::New();
  input_handle->value = input->GetMojoHandle().value();
  remote_graph->AddClamp(std::move(input_handle),
                         BlinkClampOptioinToMojo(ml_options),
                         std::move(output_desc));
}

FusionOperatorPtr AddFusionOperator(
    const MLOperator* activation,
    ml::webnn::mojom::blink::Graph* remote_graph) {
  auto fusion_operation = ml::webnn::mojom::blink::FusionOperator::New();
  switch (activation->Kind()) {
    case MLOperator::OperatorKind::kClamp: {
      auto operator_handle = ObjectHandle::New();
      operator_handle->value = activation->GetMojoHandle().value();
      const MLClampOptions* ml_options =
          static_cast<const MLClampOptions*>(activation->Options());
      remote_graph->AddFusionClamp(BlinkClampOptioinToMojo(ml_options),
                                   std::move(operator_handle));

      fusion_operation->fusion_type = FusionType::kClamp;
      break;
    }
    case MLOperator::OperatorKind::kRelu: {
      fusion_operation->fusion_type = FusionType::kRelu;
      break;
    }
    default: {
      NOTREACHED();
      break;
    }
  }
  auto operator_handle = ObjectHandle::New();
  operator_handle->value = activation->GetMojoHandle().value();
  fusion_operation->handler = std::move(operator_handle);
  return fusion_operation;
}

void AddConv2d(const MLOperator* conv2d,
               OperandDescriptorPtr output_desc,
               ml::webnn::mojom::blink::Graph* remote_graph) {
  const MLConv2dOptions* ml_options =
      static_cast<const MLConv2dOptions*>(conv2d->Options());
  auto* input = conv2d->Inputs()[0].Get();
  auto options = ml::webnn::mojom::blink::Conv2dOptions::New();
  options->padding =
      ml_options->hasPadding() ? ml_options->padding() : Vector<int32_t>(4, 0);
  options->strides =
      ml_options->hasStrides() ? ml_options->strides() : Vector<int32_t>(2, 1);
  options->dilations = ml_options->hasDilations() ? ml_options->dilations()
                                                  : Vector<int32_t>(2, 1);
  options->auto_pad = BlinkAutoPadToMojo(ml_options->autoPad().AsEnum());
  options->groups = ml_options->groups();
  options->inputLayout =
      BlinkInputOperandLayoutToMojo(ml_options->inputLayout().AsEnum());
  options->filterLayout =
      BlinkConv2dFilterOperandLayoutToMojo(ml_options->filterLayout().AsEnum());
  if (ml_options->hasBias()) {
    auto bias_handle = ObjectHandle::New();
    bias_handle->value = ml_options->bias()->GetMojoHandle().value();
    options->bias_handle = std::move(bias_handle);
  }
  if (ml_options->hasActivation()) {
    options->activation =
        AddFusionOperator(ml_options->activation(), remote_graph);
  }

  auto input_handle = ObjectHandle::New();
  input_handle->value = input->GetMojoHandle().value();
  auto* filter = conv2d->Inputs()[1].Get();
  auto filter_handle = ObjectHandle::New();
  filter_handle->value = filter->GetMojoHandle().value();
  remote_graph->AddConv2d(std::move(input_handle), std::move(filter_handle),
                          std::move(options), std::move(output_desc));
}

void AddPool2d(const MLOperator* pool2d,
               OperandDescriptorPtr output_desc,
               ml::webnn::mojom::blink::Graph* remote_graph) {
  const MLPool2dOptions* ml_options =
      static_cast<const MLPool2dOptions*>(pool2d->Options());
  auto* input = pool2d->Inputs()[0].Get();

  auto options = ml::webnn::mojom::blink::Pool2dOptions::New();
  options->window_dimensions = ml_options->hasWindowDimensions()
                                   ? ml_options->windowDimensions()
                                   : Vector<int32_t>();
  options->padding =
      ml_options->hasPadding() ? ml_options->padding() : Vector<int32_t>(4, 0);
  options->strides =
      ml_options->hasStrides() ? ml_options->strides() : Vector<int32_t>(2, 1);
  options->dilations = ml_options->hasDilations() ? ml_options->dilations()
                                                  : Vector<int32_t>(2, 1);
  options->auto_pad = BlinkAutoPadToMojo(ml_options->autoPad().AsEnum());
  options->layout =
      BlinkInputOperandLayoutToMojo(ml_options->layout().AsEnum());
  options->rounding_type =
      BlinkRoundingTypeToMojo(ml_options->roundingType().AsEnum());
  options->output_sizes = ml_options->hasOutputSizes()
                              ? ml_options->outputSizes()
                              : Vector<int32_t>();
  auto input_handle = ObjectHandle::New();
  input_handle->value = input->GetMojoHandle().value();
  remote_graph->AddPool2d(std::move(input_handle), std::move(options),
                          BlinkPool2dTypeToMojo(pool2d->Kind()),
                          std::move(output_desc));
}

void AddGemm(const MLOperator* gemm,
             OperandDescriptorPtr output_desc,
             ml::webnn::mojom::blink::Graph* remote_graph) {
  const MLGemmOptions* ml_options =
      static_cast<const MLGemmOptions*>(gemm->Options());
  auto* input_a = gemm->Inputs()[0].Get();
  auto* input_b = gemm->Inputs()[1].Get();
  auto options = ml::webnn::mojom::blink::GemmOptions::New();
  if (ml_options->hasC()) {
    auto operand_handle = ObjectHandle::New();
    operand_handle->value = ml_options->c()->GetMojoHandle().value();
    options->c_handle = std::move(operand_handle);
  }
  options->alpha = ml_options->hasAlpha() ? ml_options->alpha() : 1.0;
  options->beta = ml_options->hasBeta() ? ml_options->beta() : 1.0;
  options->a_transpose =
      ml_options->hasATranspose() ? ml_options->aTranspose() : false;
  options->b_transpose =
      ml_options->hasBTranspose() ? ml_options->bTranspose() : false;
  auto a_input_handle = ObjectHandle::New();
  a_input_handle->value = input_a->GetMojoHandle().value();
  auto b_input_handle = ObjectHandle::New();
  b_input_handle->value = input_b->GetMojoHandle().value();
  remote_graph->AddGemm(std::move(a_input_handle), std::move(b_input_handle),
                        std::move(options), std::move(output_desc));
}

void AddBinary(const MLOperator* binary,
               OperandDescriptorPtr output_desc,
               ml::webnn::mojom::blink::Graph* remote_graph) {
  auto* input0 = binary->Inputs()[0].Get();
  auto* input1 = binary->Inputs()[1].Get();
  auto input0_handle = ObjectHandle::New();
  input0_handle->value = input0->GetMojoHandle().value();
  auto input1_handle = ObjectHandle::New();
  input1_handle->value = input1->GetMojoHandle().value();
  remote_graph->AddElementWiseBinary(
      std::move(input0_handle), std::move(input1_handle),
      BlinkBinaryOperandTypeToMojo(binary->Kind()), std::move(output_desc));
}

void AddUnary(const MLOperator* unary,
              OperandDescriptorPtr output_desc,
              ml::webnn::mojom::blink::Graph* remote_graph) {
  auto* input = unary->Inputs()[0].Get();
  auto input_handle = ObjectHandle::New();
  input_handle->value = input->GetMojoHandle().value();
  remote_graph->AddUnary(std::move(input_handle),
                         BlinkUnaryOperandTypeToMojo(unary->Kind()),
                         std::move(output_desc));
}

void AddReshape(const MLOperator* reshape,
                OperandDescriptorPtr output_desc,
                ml::webnn::mojom::blink::Graph* remote_graph) {
  auto* input = reshape->Inputs()[0].Get();
  auto input_handle = ObjectHandle::New();
  input_handle->value = input->GetMojoHandle().value();
  remote_graph->AddReshape(std::move(input_handle), std::move(output_desc));
}

typedef void (*OperatorFunc)(const MLOperator*,
                             OperandDescriptorPtr,
                             ml::webnn::mojom::blink::Graph*);

OperatorFunc GetOperatorFunc(MLOperator::OperatorKind type) {
  switch (type) {
    case MLOperator::OperatorKind::kClamp:
      return AddClamp;
    case MLOperator::OperatorKind::kConv2d:
      return AddConv2d;
    case MLOperator::OperatorKind::kAdd:
      return AddBinary;
    case MLOperator::OperatorKind::kGemm:
      return AddGemm;
    case MLOperator::OperatorKind::kAveragePool2d:
      return AddPool2d;
    case MLOperator::OperatorKind::kReshape:
      return AddReshape;
    case MLOperator::OperatorKind::kRelu:
    case MLOperator::OperatorKind::kSoftmax:
      return AddUnary;
    default:
      NOTREACHED();
      return nullptr;
  }
}

}  // namespace

MojoGraph::MojoGraph(ScriptState* script_state, MLContext* context)
    : MLGraph(context), remote_graph_(ExecutionContext::From(script_state)) {}

MojoGraph::~MojoGraph() = default;

ScriptPromise MojoGraph::BuildImpl(ScriptState* script_state,
                                   MLNamedOperands named_outputs,
                                   ExceptionState& exception_state) {
  auto* request = MakeGarbageCollected<BuildRequest>(std::move(named_outputs));
  auto* resolver = MakeGarbageCollected<ScriptPromiseResolver>(script_state);
  MojoContext* mojo_context = static_cast<MojoContext*>(context_.Get());
  mojo_context->CreateGraph(
      resolver,
      WTF::BindOnce(&MojoGraph::OnGraphCreated, WrapPersistent(this),
                    WrapPersistent(script_state), WrapPersistent(resolver),
                    WrapPersistent(request)));

  return resolver->Promise();
}

void MojoGraph::BuildSyncImpl(MLNamedOperands named_outputs,
                              ExceptionState& exception_state) {
  NOTIMPLEMENTED();
  exception_state.ThrowDOMException(DOMExceptionCode::kNotSupportedError,
                                    "Not implemented");
}

ScriptPromise MojoGraph::ComputeImpl(ScriptState* script_state,
                                     MLNamedArrayInputs inputs,
                                     MLNamedArrayOutputs outputs,
                                     ExceptionState& exception_state) {
  if (inputs.size() != inputs_byte_length_.size()) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "The number of inputs is invalid");
    return ScriptPromise();
  }
  auto named_inputs = ml::webnn::mojom::blink::NamedInputs::New();
  for (const auto& input : inputs) {
    String error_message;
    DOMArrayBufferView* array_buffer_view =
        ValidateInputBuffer(input, error_message);
    if (array_buffer_view == nullptr) {
      exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                        error_message);
      return ScriptPromise();
    }
    const String& input_name = input.first;
    auto memory_info = ml::webnn::mojom::blink::MemoryInfo::New();
    memory_info->byte_offset = inputs_byte_offset_.at(input_name);
    memory_info->byte_length = inputs_byte_length_.at(input_name);
    uint8_t* address = inputs_shm_region_.mapping.GetMemoryAs<uint8_t>() +
                       memory_info->byte_offset;
    memcpy(address, array_buffer_view->BaseAddressMaybeShared(),
           array_buffer_view->byteLength());
    named_inputs->inputs.insert(input_name, std::move(memory_info));
  }
  named_inputs->shared_memory = inputs_shm_region_.region.Duplicate();

  ScriptPromiseResolver* resolver =
      MakeGarbageCollected<ScriptPromiseResolver>(script_state);
  auto* request = MakeGarbageCollected<ComputeRequest>(std::move(inputs),
                                                       std::move(outputs));
  remote_graph_->Compute(
      std::move(named_inputs),
      WTF::BindOnce(&MojoGraph::OnGraphComputed, WrapPersistent(this),
                    WrapPersistent(resolver), WrapPersistent(request)));
  return resolver->Promise();
}

void MojoGraph::ComputeSyncImpl(MLNamedArrayInputs inputs,
                                MLNamedArrayOutputs outputs,
                                ExceptionState& exception_state) {
  NOTIMPLEMENTED();
  exception_state.ThrowDOMException(DOMExceptionCode::kNotSupportedError,
                                    "Not implemented");
}

void MojoGraph::Trace(Visitor* visitor) const {
  visitor->Trace(remote_graph_);
  MLGraph::Trace(visitor);
}

void MojoGraph::OnGraphCreated(
    ScriptState* script_state,
    ScriptPromiseResolver* resolver,
    BuildRequest* request,
    mojo::PendingRemote<ml::webnn::mojom::blink::Graph> pending_remote) {
  auto* execution_context = ExecutionContext::From(script_state);
  remote_graph_.Bind(
      std::move(pending_remote),
      execution_context->GetTaskRunner(TaskType::kInternalDefault));

  base::CheckedNumeric<size_t> aligned_offset(0);
  for (const auto& input : request->inputs_) {
    String error_message;
    absl::optional<size_t> input_byte_length = ValidateAndCalculateByteLength(
        input->Type(), input->Dimensions(), error_message);
    if (!input_byte_length) {
      resolver->Reject(MakeGarbageCollected<DOMException>(
          DOMExceptionCode::kUnknownError,
          "Overflow occurred when calcuating byte length of input: " +
              error_message));
      return;
    }
    auto desc = ml::webnn::mojom::blink::OperandDescriptor::New();
    desc->data_type = BlinkOperandTypeToMojo(input->Type());
    desc->dimensions = input->Dimensions();
    // The message pipe handle can't be made multiple ScopedHanlde to pass to
    // GPU process wit mojo::MakeScopedHandle(input->GetMojoHandle()), it will
    // crash in serializing the argument, so wrap the mojo handle value in a
    // struct named ObjectHandle.
    auto operand_handle = ObjectHandle::New();
    operand_handle->value = input->GetMojoHandle().value();
    desc->handler = std::move(operand_handle);
    remote_graph_->AddInput(input->Name(), std::move(desc));

    inputs_byte_length_.insert(input->Name(), input_byte_length.value());
    inputs_byte_offset_.insert(input->Name(), aligned_offset.ValueOrDie());
    aligned_offset +=
        Align(input_byte_length.value(), kBufferAlignment).ValueOrDie();
  }
  size_t inputs_buffer_length = aligned_offset.ValueOrDie();
  inputs_shm_region_ =
      base::ReadOnlySharedMemoryRegion::Create(inputs_buffer_length);

  base::CheckedNumeric<size_t> constants_buffer_length(0);
  for (const auto& constant : request->constants_) {
    wtf_size_t size = base::checked_cast<wtf_size_t>(
        constant->ArrayBufferView()->byteLength());
    constants_buffer_length += Align(size, kBufferAlignment);
  }
  base::MappedReadOnlyRegion constants_shm_region =
      base::ReadOnlySharedMemoryRegion::Create(
          constants_buffer_length.ValueOrDie());
  auto constants_info = ml::webnn::mojom::blink::ConstantsInfo::New();
  base::CheckedNumeric<size_t> aligned_constant_offset(0);
  for (const auto& constant : request->constants_) {
    auto desc = ml::webnn::mojom::blink::OperandDescriptor::New();
    desc->data_type = BlinkOperandTypeToMojo(constant->Type());
    desc->dimensions = constant->Dimensions();
    auto operand_handle = ObjectHandle::New();
    operand_handle->value = constant->GetMojoHandle().value();
    desc->handler = std::move(operand_handle);
    remote_graph_->AddConstant(std::move(desc));

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
    constants_info->constants.insert(constant->GetMojoHandle().value(),
                                     std::move(memory_info));
  }
  constants_info->shared_memory = constants_shm_region.region.Duplicate();

  for (const auto& op : request->sorted_operators_) {
    auto* output = op->Outputs()[0].Get();
    auto output_desc = ml::webnn::mojom::blink::OperandDescriptor::New();
    output_desc->dimensions = output->Dimensions();
    auto output_handle = ObjectHandle::New();
    output_handle->value = output->GetMojoHandle().value();
    output_desc->handler = std::move(output_handle);
    OperatorFunc add_operator_func = GetOperatorFunc(op->Kind());
    if (add_operator_func == nullptr) {
      resolver->Reject(MakeGarbageCollected<DOMException>(
          DOMExceptionCode::kUnknownError,
          "Failed to get function to add the operator"));
      return;
    }
    add_operator_func(op, std::move(output_desc), remote_graph_.get());
  }

  WTF::HashMap<WTF::String, uint64_t> named_operands;
  for (const auto& [name, output] : request->outputs_) {
    String error_message;
    absl::optional<size_t> output_byte_length = ValidateAndCalculateByteLength(
        output->Type(), output->Dimensions(), error_message);
    if (!output_byte_length) {
      resolver->Reject(MakeGarbageCollected<DOMException>(
          DOMExceptionCode::kUnknownError,
          "Overflow occurred when calcuating byte length of output: " +
              error_message));
      return;
    }

    outputs_byte_length_.insert(name, output_byte_length.value());
    named_operands.insert(name, output->GetMojoHandle().value());
  }
  remote_graph_->Build(
      std::move(named_operands),
      request->constants_.size() == 0 ? nullptr : std::move(constants_info),
      WTF::BindOnce(&MojoGraph::OnGraphBuilt, WrapPersistent(this),
                    WrapPersistent(resolver)));
  return;
}

void MojoGraph::OnGraphBuilt(ScriptPromiseResolver* resolver,
                             BuildResult result) {
  switch (result) {
    case BuildResult::kUnknownError: {
      resolver->Reject(MakeGarbageCollected<DOMException>(
          DOMExceptionCode::kUnknownError, "Internal error."));
      return;
    }
    case BuildResult::kOk: {
      resolver->Resolve(this);
      return;
    }
  }
}

void MojoGraph::OnGraphComputed(ScriptPromiseResolver* resolver,
                                ComputeRequest* request,
                                ComputeResult result,
                                NamedOutputsPtr named_outputs) {
  if (result != ComputeResult::kOk) {
    resolver->Reject(MakeGarbageCollected<DOMException>(
        DOMExceptionCode::kOperationError,
        "Failed to obtain the computation result."));
    return;
  }
  for (const auto& output : request->outputs_) {
    String error_message;
    void* output_buffer_address = ValidateOutputBuffer(output, error_message);
    if (output_buffer_address == nullptr) {
      resolver->Reject(MakeGarbageCollected<DOMException>(
          DOMExceptionCode::kOperationError, error_message));
      return;
    }
    auto iter = named_outputs->outputs.find(output.first);
    if (iter == named_outputs->outputs.end()) {
      resolver->Reject(MakeGarbageCollected<DOMException>(
          DOMExceptionCode::kOperationError,
          "Failed to get result for the output."));
      return;
    }
    MemoryInfoPtr memory_info = std::move(iter->value);
    base::ReadOnlySharedMemoryRegion& shared_memory_region =
        named_outputs->shared_memory;
    DCHECK(shared_memory_region.IsValid());
    size_t byte_length = base::checked_cast<size_t>(memory_info->byte_length);
    base::ReadOnlySharedMemoryMapping shared_memory_mapping =
        shared_memory_region.MapAt(memory_info->byte_offset, byte_length);
    memcpy(output_buffer_address, shared_memory_mapping.GetMemoryAs<uint8_t>(),
           byte_length);
  }
  resolver->Resolve();
  return;
}

}  // namespace blink
