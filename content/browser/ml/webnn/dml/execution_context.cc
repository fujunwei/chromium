// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/browser/ml/webnn/dml/execution_context.h"

#include "content/browser/ml/webnn/dml/adapter_dml.h"
#include "content/browser/ml/webnn/dml/upload_resource.h"

namespace content::webnn {

ExecutionContext::ExecutionContext(scoped_refptr<AdapterDML> adapter)
    : d3d12_device_(adapter->GetD3D12Device()),
      command_queue_(adapter->GetCommandQueue()),
      command_recorder_(adapter->GetDMLDevice(), command_queue_) {}

ExecutionContext::~ExecutionContext() = default;

// Record a CopyBufferRegion for execution. Transition barriers are
// automatically inserted to transition the source and destination resources to
// COPY_SOURCE and COPY_DEST if necessary.
void ExecutionContext::CopyBufferRegion(ID3D12Resource* dest_resource,
                                        ID3D12Resource* src_resource,
                                        UINT64 resource_size,
                                        D3D12_RESOURCE_STATES state,
                                        bool needBarrierEnd) {
  DCHECK(state == D3D12_RESOURCE_STATE_COPY_DEST ||
         state == D3D12_RESOURCE_STATE_COPY_SOURCE);

  D3D12_RESOURCE_BARRIER resourceBarrier;
  // D3D12_RESOURCE_STATE_COPY_DEST is used to upload CPU data to GPU resource,
  // the resource state of source is GENERIC_READ that doesn't need to be
  // COPY_SOURCE. The destination resource state is UNORDERED_ACCESS that need
  // to transform to COPY_DEST.
  if (state == D3D12_RESOURCE_STATE_COPY_DEST) {
    resourceBarrier.Transition.pResource = dest_resource;
  } else if (state == D3D12_RESOURCE_STATE_COPY_SOURCE) {
    // D3D12_RESOURCE_STATE_COPY_SOURCE is used to read back resource from GPU
    // to CPU buffer, the source resource state is UNORDERED_ACCESS that need to
    // transform to COPY_SOURCE, the destination resource state is COPY_DEST
    // when creating the committed resource, so the barrier is unnecessary for
    // it.
    resourceBarrier.Transition.pResource = src_resource;
  }
  resourceBarrier.Transition.StateBefore =
      D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  resourceBarrier.Transition.StateAfter = state;
  resourceBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  resourceBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
  resourceBarrier.Transition.Subresource =
      D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
  command_recorder_.ResourceBarrier({resourceBarrier});
  command_recorder_.CopyBufferRegion(dest_resource, 0, src_resource, 0,
                                     resource_size);
  // if (needBarrierEnd) {
  // Reset the destination state of COPY_DEST to UNORDERED_ACCESS when uploading
  // data from CPU to GPU, the source state of COPY_SOURCE to UNORDERED_ACCESS
  // when read back GPU resource to CPU buffer.
  resourceBarrier.Transition.StateBefore = state;
  resourceBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  command_recorder_.ResourceBarrier({resourceBarrier});
  // }
}

HRESULT ExecutionContext::Initialize() {
  HRESULT hr = command_recorder_.Initialize();
  if (FAILED(hr)) {
    return hr;
  }

  execution_resources_ = std::make_unique<ExecutionResources>(this);
  command_recorder_.SetExecutionResources(execution_resources_.get());

  return S_OK;
}

HRESULT ExecutionContext::InitializeGraph(
    GraphDMLImpl* graph,
    IDMLCompiledOperator* compiled_operator,
    const DML_BINDING_DESC& input_array_binding) {
  return command_recorder_.InitializeGraph(graph, compiled_operator,
                                           input_array_binding);
}

HRESULT ExecutionContext::ExecuteGraph(
    GraphDMLImpl* graph,
    IDMLCompiledOperator* compiled_operator,
    const std::vector<DML_BINDING_DESC>& input_bindings,
    const std::vector<DML_BINDING_DESC>& output_bindings) {
  return command_recorder_.ExecuteGraph(graph, compiled_operator,
                                        input_bindings, output_bindings);
}

void ExecutionContext::Flush() {
  command_recorder_.CloseAndExecute();
}

ComPtr<ID3D12Device> ExecutionContext::GetD3D12Device() const {
  return d3d12_device_;
}

ComPtr<ID3D12CommandQueue> ExecutionContext::GetCommandQueue() const {
  return command_queue_;
}

ExecutionResources* ExecutionContext::GetExecutionResources() {
  return execution_resources_.get();
}

ComPtr<ID3D12CommandAllocator> ExecutionContext::GetCommandAllocator() {
  return command_recorder_.GetCommandAllocator();
}

ComPtr<ID3D12GraphicsCommandList> ExecutionContext::GetCommandList() {
  return command_recorder_.GetCommandList();
}

ComPtr<IDMLDevice> ExecutionContext::GetDMLDevice() {
  return command_recorder_.GetDMLDevice();
}

}  // namespace content::webnn
