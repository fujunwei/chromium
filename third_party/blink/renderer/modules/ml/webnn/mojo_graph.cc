// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/mojo_graph.h"

#include "mojo/public/cpp/bindings/pending_remote.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise_resolver.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_tensor.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"
#include "third_party/blink/renderer/core/execution_context/execution_context.h"
#include "third_party/blink/renderer/modules/ml/ml.h"
#include "third_party/blink/renderer/modules/ml/ml_context.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_builder.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operand.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operator.h"
#include "third_party/blink/renderer/modules/ml/webnn/mojo_model_info.h"
#include "third_party/blink/renderer/platform/bindings/exception_state.h"
#include "third_party/blink/renderer/platform/wtf/deque.h"
#include "third_party/blink/renderer/platform/wtf/functional.h"

#include <memory>

namespace blink {

namespace {

using ml::webnn::mojom::blink::BuildResult;
using ml::webnn::mojom::blink::ComputeResult;
using ml::webnn::mojom::blink::MemoryInfoPtr;

void AddOperation(MojoModelInfo* model_info, const MLOperator* op) {
  switch (op->Kind()) {
    case MLOperator::OperatorKind::kClamp:
      model_info->AddClamp(op);
      break;
    case MLOperator::OperatorKind::kConv2d:
      model_info->AddConv2d(op);
      break;
    case MLOperator::OperatorKind::kAdd:
    case MLOperator::OperatorKind::kSub:
    case MLOperator::OperatorKind::kMul:
    case MLOperator::OperatorKind::kDiv:
    case MLOperator::OperatorKind::kMin:
    case MLOperator::OperatorKind::kMax:
      model_info->AddElementWiseBinary(op);
      break;
    case MLOperator::OperatorKind::kGemm:
      model_info->AddGemm(op);
      break;
    case MLOperator::OperatorKind::kAveragePool2d:
    case MLOperator::OperatorKind::kMaxPool2d:
      model_info->AddPool2d(op);
      break;
    case MLOperator::OperatorKind::kRelu:
      model_info->AddRelu(op);
      break;
    case MLOperator::OperatorKind::kSoftmax:
      model_info->AddSoftmax(op);
      break;
    case MLOperator::OperatorKind::kReshape:
      model_info->AddReshape(op);
      break;
    case MLOperator::OperatorKind::kHardSwish:
      NOTIMPLEMENTED();
      break;
  }
}

}  // namespace

// static
void MojoGraph::ValidateAndBuildAsync(MLContext* context,
                                      const MLNamedOperands& named_outputs,
                                      ScriptPromiseResolver* resolver) {
  auto* graph =
      MakeGarbageCollected<MojoGraph>(resolver->GetScriptState(), context);
  graph->BuildAsync(named_outputs, resolver);
}

MojoGraph::MojoGraph(ScriptState* script_state, MLContext* context)
    : MLGraph(context), remote_graph_(ExecutionContext::From(script_state)) {}

MojoGraph::~MojoGraph() = default;

void MojoGraph::BuildAsyncImpl(const MLNamedOperands& outputs,
                               ScriptPromiseResolver* resolver) {
  auto* named_outputs = MakeGarbageCollected<MLNamedOperands>(outputs);
  ml_context_->CreateWebnnGraph(
      resolver,
      WTF::BindOnce(&MojoGraph::OnGraphCreated, WrapPersistent(this),
                    WrapPersistent(named_outputs), WrapPersistent(resolver)));
}

ScriptPromise MojoGraph::ComputeImpl(ScriptState* script_state,
                                     MLNamedArrayInputs inputs,
                                     MLNamedArrayOutputs outputs,
                                     ExceptionState& exception_state) {
  if (inputs.size() != input_resources_info_.size()) {
    exception_state.ThrowDOMException(DOMExceptionCode::kDataError,
                                      "The number of inputs is invalid");
    return ScriptPromise();
  }
  auto named_inputs = ml::webnn::mojom::blink::NamedResources::New();
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
    memory_info->byte_length = input_resources_info_.at(input_name).byte_length;
    uint8_t* address = inputs_shm_region_.mapping.GetMemoryAs<uint8_t>() +
                       memory_info->byte_offset;
    memcpy(address, array_buffer_view->BaseAddressMaybeShared(),
           array_buffer_view->byteLength());
    named_inputs->resources.insert(input_name, std::move(memory_info));
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
    MLNamedOperands* named_output,
    ScriptPromiseResolver* resolver,
    mojo::PendingRemote<ml::webnn::mojom::blink::Graph> pending_remote) {
  auto* execution_context = ExecutionContext::From(resolver->GetScriptState());
  remote_graph_.Bind(
      std::move(pending_remote),
      execution_context->GetTaskRunner(TaskType::kInternalDefault));

  HeapVector<Member<const MLOperand>> inputs;
  HeapVector<Member<const MLOperand>> constants;
  HeapVector<Member<const MLOperator>> sorted_operators;
  MLGraphBuilder::SortOperators(*named_output, inputs, constants,
                                sorted_operators);

  auto* model_info = MakeGarbageCollected<MojoModelInfo>();
  base::CheckedNumeric<size_t> aligned_offset(0);
  for (const auto& input : inputs) {
    model_info->AddInput(input.Get());
    // Create shared memory for inputs
    size_t input_byte_length =
        input_resources_info_.at(input->Name()).byte_length;
    inputs_byte_offset_.insert(input->Name(), aligned_offset.ValueOrDie());
    aligned_offset += Align(input_byte_length, kBufferAlignment).ValueOrDie();
  }
  size_t inputs_buffer_length = aligned_offset.ValueOrDie();
  inputs_shm_region_ =
      base::ReadOnlySharedMemoryRegion::Create(inputs_buffer_length);

  for (const auto& constant : constants) {
    model_info->AddConstant(constant.Get());
  }
  model_info->FillConstantsWithArrayBuffer();

  for (const auto& op : sorted_operators) {
    // Add the operation to model
    AddOperation(model_info, op.Get());
  }

  for (const auto& [name, operand] : *named_output) {
    // Add the output operand to model.
    model_info->AddOutput(std::move(name), operand);
  }

  remote_graph_->Build(
      model_info->GetModelInfo(),
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
                                NamedResourcesPtr named_outputs) {
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
    auto iter = named_outputs->resources.find(output.first);
    if (iter == named_outputs->resources.end()) {
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
