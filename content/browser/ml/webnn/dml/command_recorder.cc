// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/browser/ml/webnn/dml/command_recorder.h"

#include "content/browser/ml/webnn/dml/adapter_dml.h"
#include "content/browser/ml/webnn/dml/execution_resources.h"

namespace content::webnn {

CommandRecorder::~CommandRecorder() = default;

CommandRecorder::CommandRecorder(scoped_refptr<AdapterDML> adapter,
                                 ComPtr<IDMLDevice> dml_device)
    : adapter_(std::move(adapter)), dml_device_(std::move(dml_device)) {}

void CommandRecorder::ResourceBarrier(
    std::vector<const D3D12_RESOURCE_BARRIER> barriers) {
  command_list_->ResourceBarrier(barriers.size(), barriers.data());
}

void CommandRecorder::CopyBufferRegion(ID3D12Resource* dst_buffer,
                                       uint64_t dst_offset,
                                       ID3D12Resource* src_buffer,
                                       uint64_t src_offset,
                                       uint64_t byte_length) {
  command_list_->CopyBufferRegion(dst_buffer, dst_offset, src_buffer,
                                  src_offset, byte_length);
}

HRESULT CommandRecorder::Initialize() {
  HRESULT hr = dml_device_->GetParentDevice(IID_PPV_ARGS(&d3d12_device_));
  if (FAILED(hr)) {
    return hr;
  }

  hr = d3d12_device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                             IID_PPV_ARGS(&command_allocator_));
  if (FAILED(hr)) {
    return hr;
  }

  hr = d3d12_device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                        command_allocator_.Get(), nullptr,
                                        IID_PPV_ARGS(&command_list_));
  if (FAILED(hr)) {
    return hr;
  }

  hr = dml_device_->CreateOperatorInitializer(
      0, nullptr, IID_PPV_ARGS(&operator_initializer_));
  if (FAILED(hr)) {
    return hr;
  }
  hr = dml_device_->CreateCommandRecorder(IID_PPV_ARGS(&command_recorder_));
  if (FAILED(hr)) {
    return hr;
  }

  return S_OK;
}

HRESULT CommandRecorder::InitializeGraph(
    GraphDMLImpl* graph,
    IDMLCompiledOperator* compiled_operator,
    const DML_BINDING_DESC& input_array_binding) {
  // Reset the initializer to reference the compiled operator.
  IDMLCompiledOperator* ops[] = {compiled_operator};
  HRESULT hr = operator_initializer_->Reset(ARRAYSIZE(ops), ops);
  if (FAILED(hr)) {
    return hr;
  }

  // TODO
  DML_BINDING_PROPERTIES initializeBindingProperties =
      operator_initializer_->GetBindingProperties();
  DML_BINDING_PROPERTIES executeBindingProperties =
      compiled_operator->GetBindingProperties();
  UINT descriptorCount =
      std::max(initializeBindingProperties.RequiredDescriptorCount,
               executeBindingProperties.RequiredDescriptorCount);

  // Describe and create a constant buffer view (CBV), Shader resource view
  // (SRV), and unordered access view (UAV) descriptor heap.
  D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc{};
  descriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
  descriptorHeapDesc.NumDescriptors = descriptorCount;
  descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
  hr = d3d12_device_->CreateDescriptorHeap(&descriptorHeapDesc,
                                           IID_PPV_ARGS(&mDescriptorHeap));
  if (FAILED(hr)) {
    return hr;
  }
  // Set the descriptor heap(s).
  ID3D12DescriptorHeap* descriptorHeaps[] = {mDescriptorHeap.Get()};
  command_list_->SetDescriptorHeaps(ARRAYSIZE(descriptorHeaps),
                                    descriptorHeaps);

  // Create a binding table over the descriptor heap we just created.
  mBindingTableDesc.Dispatchable = operator_initializer_.Get();
  mBindingTableDesc.CPUDescriptorHandle =
      mDescriptorHeap->GetCPUDescriptorHandleForHeapStart();
  mBindingTableDesc.GPUDescriptorHandle =
      mDescriptorHeap->GetGPUDescriptorHandleForHeapStart();
  // The size of the binding table, in descriptors. This is the maximum number
  // of descriptors that DirectML is permitted to write, from the start of both
  // the supplied CPU and GPU descriptor handles.
  mBindingTableDesc.SizeInDescriptors = descriptorCount;

  hr = dml_device_->CreateBindingTable(&mBindingTableDesc,
                                       IID_PPV_ARGS(&mBindingTable));
  if (FAILED(hr)) {
    return hr;
  }

  UINT64 temporary_resource_size =
      std::max(initializeBindingProperties.TemporaryResourceSize,
               executeBindingProperties.TemporaryResourceSize);
  // Bind and initialize the operator on the GPU.
  if (temporary_resource_size != 0) {
    ID3D12Resource* temporary_resource = execution_resources_->Allocate(
        ResourceType::kTemporary, temporary_resource_size, graph);
    if (initializeBindingProperties.TemporaryResourceSize != 0) {
      DML_BUFFER_BINDING bufferBinding{temporary_resource, 0,
                                       temporary_resource_size};
      DML_BINDING_DESC bindingDesc{DML_BINDING_TYPE_BUFFER, &bufferBinding};
      mBindingTable->BindTemporaryResource(&bindingDesc);
    }
  }

  // Persistent resources must be supplied during initialization of a compiled
  // operator (where it is bound as an output of the operator initializer) as
  // well as during execution.
  UINT64 persistent_resource_size =
      executeBindingProperties.PersistentResourceSize;
  if (persistent_resource_size != 0) {
    ID3D12Resource* persistent_resource = execution_resources_->Allocate(
        ResourceType::kPersistent, persistent_resource_size, graph);
    DML_BUFFER_BINDING bufferBinding{persistent_resource, 0,
                                     persistent_resource_size};
    DML_BINDING_DESC bindingDesc{DML_BINDING_TYPE_BUFFER, &bufferBinding};
    mBindingTable->BindOutputs(1, &bindingDesc);
  }

  // Bind inputs if there are constant data.
  if (input_array_binding.Type != DML_BINDING_TYPE_NONE) {
    // An operator with inputs to bind be a BUFFER_ARRAY type.
    DCHECK(input_array_binding.Type == DML_BINDING_TYPE_BUFFER_ARRAY);
    mBindingTable->BindInputs(1, &input_array_binding);
  }

  // Record execution of the operator initializer.
  // The command recorder is a stateless object that records Dispatches into an
  // existing Direct3D 12 command list.
  command_recorder_->RecordDispatch(
      command_list_.Get(), operator_initializer_.Get(), mBindingTable.Get());

  return S_OK;
}

HRESULT CommandRecorder::ExecuteGraph(
    GraphDMLImpl* graph,
    IDMLCompiledOperator* compiled_operator,
    const std::vector<DML_BINDING_DESC>& input_bindings,
    const std::vector<DML_BINDING_DESC>& output_bindings) {
  // Bind and execute the operator on the GPU.
  // Reset the binding table to bind for the operator we want to execute (it
  // was previously used to bind for the initializer).
  mBindingTableDesc.Dispatchable = compiled_operator;
  mBindingTable->Reset(&mBindingTableDesc);

  DML_BINDING_PROPERTIES binding_properties =
      compiled_operator->GetBindingProperties();
  UINT64 temporary_resource_size = binding_properties.TemporaryResourceSize;
  if (temporary_resource_size != 0) {
    ID3D12Resource* temporary_resource =
        execution_resources_->GetResource(graph, ResourceType::kTemporary);
    DML_BUFFER_BINDING bufferBinding{temporary_resource, 0,
                                     temporary_resource_size};
    DML_BINDING_DESC bindingDesc{DML_BINDING_TYPE_BUFFER, &bufferBinding};
    mBindingTable->BindTemporaryResource(&bindingDesc);
  }

  UINT64 persistent_resource_size = binding_properties.PersistentResourceSize;
  if (persistent_resource_size != 0) {
    ID3D12Resource* persistent_resource =
        execution_resources_->GetResource(graph, ResourceType::kPersistent);
    DML_BUFFER_BINDING bufferBinding{persistent_resource, 0,
                                     persistent_resource_size};
    DML_BINDING_DESC bindingDesc{DML_BINDING_TYPE_BUFFER, &bufferBinding};
    mBindingTable->BindPersistentResource(&bindingDesc);
  }

  mBindingTable->BindInputs(input_bindings.size(), input_bindings.data());
  mBindingTable->BindOutputs(output_bindings.size(), output_bindings.data());

  // Record execution of the compiled operator.
  ID3D12DescriptorHeap* descriptorHeaps[] = {mDescriptorHeap.Get()};
  command_list_->SetDescriptorHeaps(ARRAYSIZE(descriptorHeaps),
                                    descriptorHeaps);
  command_recorder_->RecordDispatch(command_list_.Get(), compiled_operator,
                                    mBindingTable.Get());
  return S_OK;
}

void CommandRecorder::CloseAndExecute() const {
  const auto& command_queue = adapter_->GetCommandQueue();
  HRESULT hr = command_list_->Close();
  if (FAILED(hr)) {
    return;
  }
  command_queue->ExecuteCommandLists({command_list_.Get()});
}

HRESULT CommandRecorder::ResetCommandList() const {
  HRESULT hr = command_allocator_->Reset();
  if (FAILED(hr)) {
    return hr;
  }
  return command_list_->Reset(command_allocator_.Get(), nullptr);
}

void CommandRecorder::SetExecutionResources(ExecutionResources* resources) {
  execution_resources_ = resources;
}

ComPtr<IDMLDevice> CommandRecorder::GetDMLDevice() {
  return dml_device_;
}

}  // namespace content::webnn
