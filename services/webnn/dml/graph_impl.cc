// Copyright 2023 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "services/webnn/dml/graph_impl.h"

#include "base/check.h"
#include "base/containers/flat_map.h"
#include "base/memory/ptr_util.h"
#include "base/notreached.h"
#include "base/numerics/safe_conversions.h"
#include "base/strings/string_number_conversions.h"
#include "base/task/thread_pool.h"
#include "mojo/public/cpp/bindings/self_owned_receiver.h"
#include "services/webnn/dml/command_queue.h"
#include "services/webnn/dml/command_recorder.h"
#include "services/webnn/dml/graph_builder.h"
#include "services/webnn/dml/tensor_desc.h"
#include "ui/gl/gl_angle_util_win.h"

namespace webnn::dml {
namespace {

using Microsoft::WRL::ComPtr;
using mojom::Operand;
using mojom::OperandPtr;
using mojom::Operator;
using mojom::OperatorPtr;

// A map of all mojom operands in `mojom::GraphInfo` using the mojom operand id
// as key.
using IdToOperandMap = base::flat_map<uint64_t, OperandPtr>;
// A map of all NodeOutputInfos using the mojom operand id as key.
using IdToNodeOutputMap = std::map<uint64_t, NodeOutputInfo>;

DML_TENSOR_DATA_TYPE GetTensorDataType(Operand::DataType type) {
  switch (type) {
    case Operand::DataType::kFloat32:
      return DML_TENSOR_DATA_TYPE_FLOAT32;
    case Operand::DataType::kFloat16:
      return DML_TENSOR_DATA_TYPE_FLOAT16;
    case Operand::DataType::kInt8:
      return DML_TENSOR_DATA_TYPE_INT8;
    case Operand::DataType::kUint8:
      return DML_TENSOR_DATA_TYPE_UINT8;
    case Operand::DataType::kInt32:
      return DML_TENSOR_DATA_TYPE_INT32;
    case Operand::DataType::kUint32:
      return DML_TENSOR_DATA_TYPE_UINT32;
    default:
      DLOG(ERROR) << "This data type is not supported.";
      NOTREACHED_NORETURN();
  }
}

std::string OpKindToString(Operator::Kind kind) {
  switch (kind) {
    case Operator::Kind::kClamp:
      return "clamp";
    case Operator::Kind::kAdd:
      return "add";
    case Operator::Kind::kSub:
      return "sub";
    case Operator::Kind::kMul:
      return "mul";
    case Operator::Kind::kDiv:
      return "div";
    case Operator::Kind::kMax:
      return "max";
    case Operator::Kind::kMin:
      return "min";
    case Operator::Kind::kRelu:
      return "relu";
    case Operator::Kind::kReshape:
      return "reshape";
    case Operator::Kind::kSoftmax:
      return "softmax";
    default:
      return base::NumberToString(base::checked_cast<uint32_t>(kind));
  }
}

// Define some methods like CreateInputNode and CreateOperatorNodeForRelu here
// to focus on converting the mojo graph struct to corresponding DML graph node
// by using dml::GraphBuilder as a helper. dml::GraphBuilder should be decoupled
// from mojo graph structs and focus on manipulating DML graph structs.
void CreateInputNode(const IdToOperandMap& id_to_operand_map,
                     uint64_t input_id,
                     GraphBuilder& graph_builder,
                     IdToNodeOutputMap& id_to_node_output_map) {
  const OperandPtr& operand = id_to_operand_map.at(input_id);
  TensorDesc input_tensor_desc(GetTensorDataType(operand->data_type),
                               operand->dimensions);
  NodeInfo input_node = graph_builder.CreateInputNode();
  NodeOutputInfo input_node_output = graph_builder.CreateNodeOutput(
      std::move(input_node), std::move(input_tensor_desc));
  id_to_node_output_map[input_id] = std::move(input_node_output);
}

void CreateOperatorNodeForRelu(const IdToOperandMap& id_to_operand_map,
                               const OperatorPtr& operation,
                               GraphBuilder& graph_builder,
                               IdToNodeOutputMap& id_to_node_output_map) {
  uint64_t input_id = operation->input_operands[0];
  const auto input_iterator = id_to_node_output_map.find(input_id);
  CHECK(input_iterator != id_to_node_output_map.end());
  NodeOutputInfo input_node_output = input_iterator->second;
  TensorDesc input_tensor_desc =
      graph_builder.GetNodeOutput(input_node_output).tensor_desc;

  uint64_t output_id = operation->output_operands[0];
  const OperandPtr& output_operand = id_to_operand_map.at(output_id);
  TensorDesc output_tensor_desc(GetTensorDataType(output_operand->data_type),
                                output_operand->dimensions);

  DML_ACTIVATION_RELU_OPERATOR_DESC relu_operator_desc{
      .InputTensor = &input_tensor_desc.GetDMLTensorDesc(),
      .OutputTensor = &output_tensor_desc.GetDMLTensorDesc()};
  // TensorDesc input_tensor_desc(DML_TENSOR_DATA_TYPE_FLOAT32, {2, 2});
  // DML_ACTIVATION_RELU_OPERATOR_DESC relu_operator_desc{
  //     .InputTensor = &input_tensor_desc.GetDMLTensorDesc(),
  //     .OutputTensor = &input_tensor_desc.GetDMLTensorDesc()};
  NodeInfo relu_node = graph_builder.CreateOperatorNode(
      DML_OPERATOR_ACTIVATION_RELU, &relu_operator_desc, {input_node_output});
  NodeOutputInfo relu_output =
      graph_builder.CreateNodeOutput(relu_node, std::move(output_tensor_desc));
  id_to_node_output_map[output_id] = std::move(relu_output);
}

// DirectML API does not have a real Reshape operator. The WebNN Reshape is
// implemented by creating a new NodeOutput for the input Node. The new
// NodeOutput has the reshaped dimensions and is used as the output of the WebNN
// Reshape operator. And if the input and output of the Reshape are exactly the
// input and output of the DirectML graph, we need to add another DirectML
// Identity operator to ensure that the DirectML graph can be compiled and
// calculated correctly.
void CreateNodeOutputForReshape(const IdToOperandMap& id_to_operand_map,
                                const OperatorPtr& operation,
                                GraphBuilder& graph_builder,
                                IdToNodeOutputMap& id_to_node_output_map) {
  uint64_t input_id = operation->input_operands[0];
  const auto input_iterator = id_to_node_output_map.find(input_id);
  CHECK(input_iterator != id_to_node_output_map.end());
  NodeOutputInfo input_node_output_info = input_iterator->second;
  NodeOutput input_node_output =
      graph_builder.GetNodeOutput(input_node_output_info);
  TensorDesc input_tensor_desc = input_node_output.tensor_desc;
  NodeInfo input_node = input_node_output.node_info;
  uint64_t output_id = operation->output_operands[0];
  const OperandPtr& output_operand = id_to_operand_map.at(output_id);
  TensorDesc output_tensor_desc(input_tensor_desc.GetDataType(),
                                DML_TENSOR_FLAG_NONE,
                                output_operand->dimensions);
  NodeOutputInfo reshaped_input_node_output =
      graph_builder.CreateNodeOutput(input_node, std::move(output_tensor_desc));
  id_to_node_output_map[output_id] = std::move(reshaped_input_node_output);
}

// Creates a DirectML operator for the WebNN general matrix multiplication
// (GEMM) of the expression alpha * A * B + beta * C.
bool CreateOperatorNodeForGemm(const IdToOperandMap& id_to_operand_map,
                               const OperatorPtr& operation,
                               GraphBuilder& graph_builder,
                               IdToNodeOutputMap& id_to_node_output_map) {
  uint64_t input_a_id = operation->input_operands[0];
  uint64_t input_b_id = operation->input_operands[1];

  const auto input_a_node_output_iterator =
      id_to_node_output_map.find(input_a_id);
  CHECK(input_a_node_output_iterator != id_to_node_output_map.end());

  const auto input_b_node_output_iterator =
      id_to_node_output_map.find(input_b_id);
  CHECK(input_b_node_output_iterator != id_to_node_output_map.end());

  NodeOutputInfo input_a_node_output = input_a_node_output_iterator->second;
  TensorDesc input_a_tensor_desc =
      graph_builder.GetNodeOutput(input_a_node_output).tensor_desc;

  NodeOutputInfo input_b_node_output = input_b_node_output_iterator->second;
  TensorDesc input_b_tensor_desc =
      graph_builder.GetNodeOutput(input_b_node_output).tensor_desc;

  uint64_t output_id = operation->output_operands[0];
  const OperandPtr& output_operand = id_to_operand_map.at(output_id);
  TensorDesc output_tensor_desc(GetTensorDataType(output_operand->data_type),
                                output_operand->dimensions);

  absl::optional<TensorDesc> input_c_tensor_desc = absl::nullopt;
  auto& gemm_attributes = operation->attributes->get_gemm();

  auto& c_operand_id = gemm_attributes->c_operand_id;
  if (c_operand_id) {
    uint64_t input_c_id = c_operand_id.value();

    const auto input_c_node_output_iterator =
        id_to_node_output_map.find(input_c_id);
    CHECK(input_c_node_output_iterator != id_to_node_output_map.end());

    NodeOutputInfo input_c_node_output_info =
        input_c_node_output_iterator->second;
    input_c_tensor_desc =
        graph_builder.GetNodeOutput(input_c_node_output_info).tensor_desc;

    // TODO(crbug.com/1471201): Support broadcasting for C.
    auto input_c_shape = input_c_tensor_desc->GetDimensions();
    if (input_c_shape.size() < 2) {
      return false;
    }

    auto output_shape = output_tensor_desc.GetDimensions();
    CHECK_EQ(output_shape.size(), input_c_shape.size());

    if (output_shape[0] != input_c_shape[0] ||
        output_shape[1] != input_c_shape[1]) {
      return false;
    }
  }

  DML_GEMM_OPERATOR_DESC gemm_operator_desc{
      .ATensor = &input_a_tensor_desc.GetDMLTensorDesc(),
      .BTensor = &input_b_tensor_desc.GetDMLTensorDesc(),
      .CTensor = (input_c_tensor_desc.has_value())
                     ? &input_c_tensor_desc->GetDMLTensorDesc()
                     : nullptr,
      .OutputTensor = &output_tensor_desc.GetDMLTensorDesc(),
      .TransA = (gemm_attributes->a_transpose) ? DML_MATRIX_TRANSFORM_TRANSPOSE
                                               : DML_MATRIX_TRANSFORM_NONE,
      .TransB = (gemm_attributes->b_transpose) ? DML_MATRIX_TRANSFORM_TRANSPOSE
                                               : DML_MATRIX_TRANSFORM_NONE,
      .Alpha = gemm_attributes->alpha,
      .Beta = gemm_attributes->beta,
      .FusedActivation = nullptr,  // Not supported
  };

  NodeInfo gemm_node_info = graph_builder.CreateOperatorNode(
      DML_OPERATOR_GEMM, &gemm_operator_desc,
      {input_a_node_output, input_b_node_output});
  if (gemm_node_info.type == NodeInfo::Type::kInvalid) {
    return false;
  }

  NodeOutputInfo gemm_output = graph_builder.CreateNodeOutput(
      gemm_node_info, std::move(output_tensor_desc));
  id_to_node_output_map[output_id] = std::move(gemm_output);

  return true;
}

}  // namespace

D3D12_RESOURCE_BARRIER CreateTransitionBarrier(ID3D12Resource* resource,
                                               D3D12_RESOURCE_STATES before,
                                               D3D12_RESOURCE_STATES after) {
  return {.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
          .Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE,
          .Transition = {.pResource = resource,
                         .Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                         .StateBefore = before,
                         .StateAfter = after}};
}

void Upload(CommandRecorder* command_recorder,
            void* src_buffer,
            size_t buffer_size,
            ComPtr<ID3D12Resource> dst_resource) {
  // Copy the contents from source buffer to upload buffer.
  ComPtr<ID3D12Resource> upload_buffer;
  HRESULT hr = command_recorder->CreateUploadBuffer(buffer_size, upload_buffer);
  if (FAILED(hr)) {
    DLOG(ERROR) << "Failed to CreateUploadBuffer: "
                << logging::SystemErrorCodeToString(hr);
    return;
  }
  void* upload_buffer_data = nullptr;
  hr = upload_buffer->Map(0, nullptr, &upload_buffer_data);
  if (FAILED(hr)) {
    DLOG(ERROR) << "Failed to Map: " << logging::SystemErrorCodeToString(hr);
    return;
  }
  memcpy(upload_buffer_data, src_buffer, buffer_size);
  upload_buffer->Unmap(0, nullptr);

  // Copy the input data from upload buffer to input buffer.
  D3D12_RESOURCE_BARRIER barriers[1];
  barriers[0] = CreateTransitionBarrier(dst_resource.Get(),
                                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                        D3D12_RESOURCE_STATE_COPY_DEST);
  command_recorder->ResourceBarrier(barriers);
  command_recorder->CopyBufferRegion(dst_resource.Get(), 0, upload_buffer.Get(), 0,
                                     buffer_size);
  // The bound resources should be in D3D12_RESOURCE_STATE_UNORDERED_ACCESS
  // state before the execution of RecordDispatch on the GPU.
  barriers[0] =
      CreateTransitionBarrier(dst_resource.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                              D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  command_recorder->ResourceBarrier(barriers);

  // Keep the upload_buffer alive until the GPU work is done.
  command_recorder->GetCommandQueue()->ReferenceUntilCompleted(
      std::move(upload_buffer));
  command_recorder->GetCommandQueue()->ReferenceUntilCompleted(
      std::move(dst_resource));
}

void OnExecutionComplete(
    ComPtr<ID3D12Resource> readback_buffer) {
  // // Release the resources referred by GPU execution.
  // command_recorder->GetCommandQueue()->ReleaseCompletedResources();

  // Copy the contents from readback buffer to destination buffer.
  LOG(ERROR) << "==============OnExecutionComplete";
  void* readback_buffer_data = nullptr;
  readback_buffer->Map(0, nullptr, &readback_buffer_data);
  for (size_t i = 0; i < 4; ++i) {
    LOG(ERROR) << "==========result: " << ((float*)readback_buffer_data)[i];
  }
  readback_buffer->Unmap(0, nullptr);
}

void Download(CommandRecorder* command_recorder,
                                        size_t buffer_size,
                                        ComPtr<ID3D12Resource> src_resource) {
  ComPtr<ID3D12Resource> readback_buffer;
  HRESULT hr = command_recorder->CreateReadbackBuffer(buffer_size, readback_buffer);
  if (FAILED(hr)) {
    DLOG(ERROR) << "Failed to CreateReadbackBuffer: "
                << logging::SystemErrorCodeToString(hr);
    return;
  }
  // Copy the result from output buffer to readback buffer.
  D3D12_RESOURCE_BARRIER barriers[1];
  barriers[0] = CreateTransitionBarrier(src_resource.Get(),
                                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                        D3D12_RESOURCE_STATE_COPY_SOURCE);
  command_recorder->ResourceBarrier(barriers);
  command_recorder->CopyBufferRegion(readback_buffer.Get(), 0, src_resource.Get(), 0,
                                     buffer_size);
  barriers[0] =
      CreateTransitionBarrier(src_resource.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                              D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  command_recorder->ResourceBarrier(barriers);

  command_recorder->GetCommandQueue()->ReferenceUntilCompleted(readback_buffer);
  command_recorder->GetCommandQueue()->ReferenceUntilCompleted(src_resource);
  // Close, execute and wait for completion.
  hr = command_recorder->CloseAndExecute();
  if (FAILED(hr)) {
    DLOG(ERROR) << "Failed to CloseAndExecute: "
                << logging::SystemErrorCodeToString(hr);
    return;
  }
  // hr = command_recorder->GetCommandQueue()->WaitSyncForTesting();
  // if (FAILED(hr)) {
  //   DLOG(ERROR) << "Failed to WaitSyncForTesting: "
  //               << logging::SystemErrorCodeToString(hr);
  //   return;
  // }
  hr = command_recorder->GetCommandQueue()->WaitAsync(base::BindOnce(
      &OnExecutionComplete, std::move(readback_buffer)));
  if (FAILED(hr)) {
    DLOG(ERROR) << "Failed to wait the initialization completed: "
                << logging::SystemErrorCodeToString(hr);
    return;
  }

}

ComPtr<IDMLCompiledOperator> BuildRelu(CommandRecorder* command_recorder) {
  // Test dispatching a DirectML Relu operator twice for different input and
  // output bindings before waiting for GPU work to complete.
  //
  // Create a Relu operator.
  TensorDesc input_tensor_desc(DML_TENSOR_DATA_TYPE_FLOAT32, {2, 2});
  DML_ACTIVATION_RELU_OPERATOR_DESC relu_operator_desc{
      .InputTensor = &input_tensor_desc.GetDMLTensorDesc(),
      .OutputTensor = &input_tensor_desc.GetDMLTensorDesc()};
  DML_OPERATOR_DESC operator_desc{.Type = DML_OPERATOR_ACTIVATION_RELU,
                                  .Desc = &relu_operator_desc};
  ComPtr<IDMLOperator> dml_operator;
  command_recorder->GetDMLDevice()->CreateOperator(
      &operator_desc, IID_PPV_ARGS(&dml_operator));

  // Compile the operator.
  ComPtr<IDMLCompiledOperator> compiled_operator;
  command_recorder->GetDMLDevice()->CompileOperator(
      dml_operator.Get(), DML_EXECUTION_FLAG_NONE,
      IID_PPV_ARGS(&compiled_operator));

  // Initialize the operator.
  // auto command_recorder = CommandRecorder::Create(adapter_->command_queue(),
  //                                                 adapter_->dml_device());
  // ASSERT_NE(command_recorder.get(), nullptr);
  command_recorder->Open();
  // Relu operator initializer deson't need to bind any input and persistent
  // resources.
  command_recorder->InitializeOperator(
      compiled_operator.Get(), absl::nullopt, absl::nullopt);
  command_recorder->CloseAndExecute();
  
      command_recorder->GetCommandQueue()->WaitSyncForTesting();
  command_recorder->GetCommandQueue()->ReleaseCompletedResources();
  // adapter_->dml_device()->GetDeviceRemovedReason();
  // adapter_->d3d12_device()->GetDeviceRemovedReason();
  return compiled_operator;
}

GraphImpl::GraphImpl(
    std::unique_ptr<CommandRecorder> command_recorder,
    ComPtr<ID3D12Resource> persistent_buffer,
    ComPtr<IDMLCompiledOperator> compiled_operator,
    std::unique_ptr<ComputeBufferValidator> compute_buffer_validator)
    : WebNNGraphImpl(std::move(compute_buffer_validator)),
      persistent_buffer_(std::move(persistent_buffer)),
      command_recorder_(std::move(command_recorder)),
      compiled_operator_(std::move(compiled_operator)) {
  // ComPtr<IDMLCompiledOperator> compiled_operator = BuildRelu(command_recorder_);
  // Create input and output resources that will be bound for operator for
  // execution.
  const uint64_t input_buffer_size = 2 * 2 * 4;
  ComPtr<ID3D12Resource> inputA_buffer;
  HRESULT hr =
      command_recorder_->CreateDefaultBuffer(input_buffer_size, inputA_buffer);
  if (FAILED(hr)) {
    DLOG(ERROR) << "Failed to CreateDefaultBuffer: "
                << logging::SystemErrorCodeToString(hr);
    return;
  }
  ComPtr<ID3D12Resource> inputB_buffer;
  hr = command_recorder_->CreateDefaultBuffer(input_buffer_size, inputB_buffer);
  if (FAILED(hr)) {
    DLOG(ERROR) << "Failed to CreateDefaultBuffer: "
                << logging::SystemErrorCodeToString(hr);
    return;
  }
  const uint64_t output_buffer_size = input_buffer_size;
  ComPtr<ID3D12Resource> output_buffer;
  hr =
      command_recorder_->CreateDefaultBuffer(output_buffer_size, output_buffer);
  if (FAILED(hr)) {
    DLOG(ERROR) << "Failed to CreateDefaultBuffer: "
                << logging::SystemErrorCodeToString(hr);
    return;
  }

  // Re-open the command recorder for recording operator execution commands.
  hr = command_recorder_->Open();
  if (FAILED(hr)) {
    DLOG(ERROR) << "Failed to CreateDefaultBuffer: "
                << logging::SystemErrorCodeToString(hr);
    return;
  }

  // Upload input data to input resource.
  std::vector<float> inputA_data({1.0, -2.0, 3.0, -4.0});
  std::vector<float> inputB_data({5.0, 6.0, 7.0, 8.0});
  Upload(command_recorder_.get(), inputA_data.data(), input_buffer_size,
         inputA_buffer);
  Upload(command_recorder_.get(), inputB_data.data(), input_buffer_size,
         inputB_buffer);

  // Create the input and output resources binding for operator execution.
  DML_BUFFER_BINDING inputA_buffer_binding{.Buffer = inputA_buffer.Get(),
                                          .Offset = 0,
                                          .SizeInBytes = input_buffer_size};
  // DML_BUFFER_BINDING inputB_buffer_binding{.Buffer = inputB_buffer.Get(),
  //                                         .Offset = 0,
  //                                         .SizeInBytes = input_buffer_size};
  std::vector<DML_BINDING_DESC> input_bindings(
      {// InputA.
       {.Type = DML_BINDING_TYPE_BUFFER, .Desc = &inputA_buffer_binding}});
  DML_BUFFER_BINDING output_buffer_binding{.Buffer = output_buffer.Get(),
                                           .Offset = 0,
                                           .SizeInBytes =
                                           output_buffer_size};
  std::vector<DML_BINDING_DESC> output_bindings(
      {{.Type = DML_BINDING_TYPE_BUFFER, .Desc = &output_buffer_binding}});

  // Execute the operator with persistent, input and output bindings.
  hr = command_recorder_->ExecuteOperator(
      compiled_operator_.Get(), input_bindings, output_bindings,
      absl::nullopt);
  if (FAILED(hr)) {
    DLOG(ERROR) << "Failed to ExecuteOperator: "
                << logging::SystemErrorCodeToString(hr);
    return;
  }
  //persistent_buffer_binding_desc

  // Download the result from output resource.
  Download(command_recorder_.get(), output_buffer_size,
           output_buffer);
}

//  Notice that it's the CommandQueue's responsibility to wait for all of the
//  queued work to complete before destructing itself.
GraphImpl::~GraphImpl() = default;

ComPtr<IDMLCompiledOperator> GraphImpl::CompileOnBackgroundThread(
    std::vector<NodeOutputInfo> graph_outputs,
    GraphBuilder graph_builder) {
  return graph_builder.Compile(graph_outputs, DML_EXECUTION_FLAG_NONE);
}

// static
void GraphImpl::OnCompilationComplete(
    mojom::WebNNContext::CreateGraphCallback callback,
    std::unique_ptr<CommandRecorder> command_recorder,
    std::unique_ptr<ComputeBufferValidator> compute_buffer_validator,
    ComPtr<IDMLCompiledOperator> compiled_operator) {
  if (!compiled_operator) {
    DLOG(ERROR) << "Failed to compile the graph.";
    std::move(callback).Run(mojo::NullRemote());
    return;
  }

  HRESULT hr = command_recorder->Open();
  if (FAILED(hr)) {
    DLOG(ERROR) << "Failed to open the command recorder: "
                << logging::SystemErrorCodeToString(hr);
    std::move(callback).Run(mojo::NullRemote());
    return;
  }

  // TODO(crbug.com/1273291): Create the input resource binding for
  // operator initialization. Only the constant resource needs to be bound.

  // Create the persistent resource which is bound as output of operator
  // initializer.
  absl::optional<DML_BINDING_DESC> persistent_buffer_binding_desc =
      absl::nullopt;
  DML_BINDING_PROPERTIES execution_binding_properties =
      compiled_operator->GetBindingProperties();
  uint64_t persistent_buffer_size =
      execution_binding_properties.PersistentResourceSize;
  ComPtr<ID3D12Resource> persistent_buffer;
  LOG(ERROR) << "=======persistent_buffer_size " << persistent_buffer_size;
  if (persistent_buffer_size) {
    hr = command_recorder->CreateDefaultBuffer(persistent_buffer_size,
                                               persistent_buffer);
    if (FAILED(hr)) {
      DLOG(ERROR) << "Failed to create the default buffer: "
                  << logging::SystemErrorCodeToString(hr);
      std::move(callback).Run(mojo::NullRemote());
      return;
    }

    DML_BUFFER_BINDING persistent_buffer_binding{
        .Buffer = persistent_buffer.Get(),
        .Offset = 0,
        .SizeInBytes = persistent_buffer_size};

    persistent_buffer_binding_desc = DML_BINDING_DESC{
        .Type = DML_BINDING_TYPE_BUFFER, .Desc = &persistent_buffer_binding};
  }

  hr = command_recorder->InitializeOperator(
      compiled_operator.Get(), absl::nullopt, persistent_buffer_binding_desc);
  if (FAILED(hr)) {
    DLOG(ERROR) << "Failed to initialize the operator: "
                << logging::SystemErrorCodeToString(hr);
    std::move(callback).Run(mojo::NullRemote());
    return;
  }

  hr = command_recorder->CloseAndExecute();
  if (FAILED(hr)) {
    DLOG(ERROR) << "Failed to close and execute the command list: "
                << logging::SystemErrorCodeToString(hr);
    std::move(callback).Run(mojo::NullRemote());
    return;
  }

  scoped_refptr<CommandQueue> command_queue(
      command_recorder->GetCommandQueue());

  // Ensure the GPU resources needed by the initialization work on the
  // CommandQueue not to be released before the work completes.
  if (persistent_buffer) {
    command_queue->ReferenceUntilCompleted(persistent_buffer);
  }
  //  The IDMLCompiledOperator should also be referenced before the work
  //  completes.
  command_queue->ReferenceUntilCompleted(compiled_operator);

  hr = command_queue->WaitAsync(base::BindOnce(
      &GraphImpl::OnInitializationComplete, std::move(command_recorder),
      std::move(persistent_buffer), std::move(compiled_operator),
      std::move(compute_buffer_validator), std::move(callback)));
  if (FAILED(hr)) {
    DLOG(ERROR) << "Failed to wait the initialization completed: "
                << logging::SystemErrorCodeToString(hr);
    std::move(callback).Run(mojo::NullRemote());
  }
}

// static
void GraphImpl::OnInitializationComplete(
    std::unique_ptr<CommandRecorder> command_recorder,
    ComPtr<ID3D12Resource> persistent_buffer,
    ComPtr<IDMLCompiledOperator> compiled_operator,
    std::unique_ptr<ComputeBufferValidator> compute_buffer_validator,
    mojom::WebNNContext::CreateGraphCallback callback) {
  scoped_refptr<CommandQueue> command_queue(
      command_recorder->GetCommandQueue());
  // The remote sent to the renderer.
  mojo::PendingRemote<mojom::WebNNGraph> blink_remote;
  // The receiver bound to GraphImpl.
  mojo::MakeSelfOwnedReceiver<mojom::WebNNGraph>(
      base::WrapUnique(new GraphImpl(
          std::move(command_recorder), std::move(persistent_buffer),
          compiled_operator, std::move(compute_buffer_validator))),
      blink_remote.InitWithNewPipeAndPassReceiver());
  command_queue->ReleaseCompletedResources();
  std::move(callback).Run(std::move(blink_remote));
}

// static
void GraphImpl::CreateAndBuild(
    scoped_refptr<CommandQueue> command_queue,
    ComPtr<IDMLDevice> dml_device,
    const mojom::GraphInfoPtr& graph_info,
    mojom::WebNNContext::CreateGraphCallback callback) {
  // `CommandRecorder` would keep reference of command queue and DML device.
  std::unique_ptr<CommandRecorder> command_recorder =
      CommandRecorder::Create(command_queue, dml_device);
  if (!command_recorder) {
    DLOG(ERROR) << "Failed to open the command recorder.";
    std::move(callback).Run(mojo::NullRemote());
    return;
  }

  GraphBuilder graph_builder(dml_device);
  IdToNodeOutputMap id_to_node_output_map;
  const IdToOperandMap& id_to_operand_map = graph_info->id_to_operand_map;
  // Add inputs.
  for (auto& input_id : graph_info->input_operands) {
    CreateInputNode(id_to_operand_map, input_id, graph_builder,
                    id_to_node_output_map);
  }

  // TODO(crbug.com/1455278): Add constants. It depends on the mojo constant
  // definition is ready.

  // Add operations.
  for (auto& operation : graph_info->operators) {
    // For operators that deal with DML API, there is a chance that operator
    // creation will fail.
    bool is_create_successful = true;
    switch (operation->kind) {
      case Operator::Kind::kRelu: {
        CreateOperatorNodeForRelu(id_to_operand_map, operation, graph_builder,
                                  id_to_node_output_map);
        break;
      }
      case Operator::Kind::kReshape: {
        CreateNodeOutputForReshape(id_to_operand_map, operation, graph_builder,
                                   id_to_node_output_map);
        break;
      }
      case Operator::Kind::kGemm: {
        is_create_successful = CreateOperatorNodeForGemm(
            id_to_operand_map, operation, graph_builder, id_to_node_output_map);
        break;
      }
      default:
        DLOG(ERROR) << "This operator kind (" +
                           OpKindToString(operation->kind) +
                           ") is not supported.";
        is_create_successful = false;
    }
    if (!is_create_successful) {
      std::move(callback).Run(mojo::NullRemote());
      return;
    }
  }

  std::vector<NodeOutputInfo> graph_outputs;
  graph_outputs.reserve(graph_info->output_operands.size());
  for (auto& output_id : graph_info->output_operands) {
    const auto output_iterator = id_to_node_output_map.find(output_id);
    CHECK(output_iterator != id_to_node_output_map.end());
    NodeOutputInfo node_output = output_iterator->second;

    // TODO: A DML graph's output tensor may have adjusted strides rather than
    // default strides which are calculated by its' dimensions. For example,
    // dimensions [1,2,3,4] should have default strides [24,12,4,1] according to
    // https://docs.microsoft.com/en-us/windows/win32/direct3d12/dml-helper-functions#calculatestrides,
    // but the strides may be adjusted for supporting some ops such as
    // transpose. Append an identity operator to consume the adjusted strides to
    // ensure a correct output result.

    // Appending an identity operator DML_OPERATOR_ELEMENT_WISE_IDENTITY which
    // effectively copies input tensor to the output tensor to avoid directly
    // using graph input as output.
    NodeOutput output_node = graph_builder.GetNodeOutput(node_output);
    TensorDesc output_tensor_desc = output_node.tensor_desc;
    auto output_type = output_node.node_info.type;
    if (output_type == NodeInfo::Type::kInput) {
      TensorDesc identity_tensor_desc(output_tensor_desc.GetDataType(),
                                      DML_TENSOR_FLAG_NONE,
                                      output_tensor_desc.GetDimensions());
      DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC identity_operator_desc{
          .InputTensor = &output_tensor_desc.GetDMLTensorDesc(),
          .OutputTensor = &identity_tensor_desc.GetDMLTensorDesc()};
      NodeInfo identity_node = graph_builder.CreateOperatorNode(
          DML_OPERATOR_ELEMENT_WISE_IDENTITY, &identity_operator_desc,
          {node_output});
      NodeOutputInfo identity_node_output = graph_builder.CreateNodeOutput(
          identity_node, std::move(identity_tensor_desc));
      graph_outputs.push_back(std::move(identity_node_output));
    } else {
      graph_outputs.push_back(std::move(node_output));
    }
  }
  

  base::ThreadPool::PostTaskAndReplyWithResult(
      FROM_HERE,
      base::BindOnce(&GraphImpl::CompileOnBackgroundThread,
                     std::move(graph_outputs), std::move(graph_builder)),
      base::BindOnce(&GraphImpl::OnCompilationComplete, std::move(callback),
                     std::move(command_recorder),
                     std::make_unique<ComputeBufferValidator>(graph_info)));

  // ComPtr<IDMLCompiledOperator> compiled_operator = 
  // graph_builder.Compile(graph_outputs, DML_EXECUTION_FLAG_NONE);

  // command_recorder->Open();
  // Relu operator initializer deson't need to bind any input and persistent
  // resources.
  // command_recorder->InitializeOperator(
  //     compiled_operator.Get(), absl::nullopt, absl::nullopt);
  // command_recorder->CloseAndExecute();
  // command_recorder->GetCommandQueue()->WaitSyncForTesting();

  
}

void GraphImpl::ComputeImpl(
    base::flat_map<std::string, mojo_base::BigBuffer> named_inputs,
    mojom::WebNNGraph::ComputeCallback callback) {
  // Copy all array buffers of inputs to an _UPLOAD heap and create a committed
  // resource which is mapped to the heap.
  //
  // Calculate the total byte length of inputs array buffer to create an upload
  // buffer which can be read by GPU.
  base::CheckedNumeric<size_t> total_byte_length(0);
  base::flat_map<std::string, size_t> input_to_byte_offset_map;
  for (auto& [input_name, input_buffer] : named_inputs) {
    // There is only one upload heap for all inputs, the byte offset is used to
    // get the copied address for each input tensor.
    input_to_byte_offset_map[input_name] = total_byte_length.ValueOrDie();

    // The buffer has a minimum base address alignment requirement of 16 bytes
    // in the macro `DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT`:
    // https://learn.microsoft.com/en-us/windows/win32/direct3d12/direct3d-directml-constants
    total_byte_length +=
        (base::MakeCheckedNum<size_t>(input_buffer.size()) + 15) & ~15ull;
    if (!total_byte_length.IsValid()) {
      DLOG(ERROR) << "Failed to calculate the total byte length of inputs.";
      std::move(callback).Run(mojom::ComputeResult::kUnknownError,
                              absl::nullopt);
      return;
    }
  }
  // Create the upload heap and a resource that is mapped to the heap.
  ComPtr<ID3D12Resource> input_upload_buffer;
  HRESULT hr = command_recorder_->CreateUploadBuffer(
      total_byte_length.ValueOrDie(), input_upload_buffer);
  if (FAILED(hr)) {
    DLOG(ERROR) << "Failed to create upload buffer for inputs: "
                << logging::SystemErrorCodeToString(hr);
    std::move(callback).Run(mojom::ComputeResult::kUnknownError, absl::nullopt);
    return;
  }
  // Map entire resource to copy the array buffer of input one by one with byte
  // offset.
  void* mapped_upload_buffer = nullptr;
  hr = input_upload_buffer->Map(0, nullptr, &mapped_upload_buffer);
  if (FAILED(hr)) {
    DLOG(ERROR) << "Failed to map upload buffer for inputs: "
                << logging::SystemErrorCodeToString(hr);
    std::move(callback).Run(mojom::ComputeResult::kUnknownError, absl::nullopt);
    return;
  }
  for (auto& [input_name, input_buffer] : named_inputs) {
    memcpy(static_cast<uint8_t*>(mapped_upload_buffer) +
               input_to_byte_offset_map.at(input_name),
           input_buffer.data(), input_buffer.size());
  }
  input_upload_buffer->Unmap(0, nullptr);

  // TODO(crbug.com/1273291): Execute the compiled operator with inputs/outputs.
  std::move(callback).Run(mojom::ComputeResult::kUnknownError, absl::nullopt);
}

}  // namespace webnn::dml
