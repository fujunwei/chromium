// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/browser/ml/webnn/dml/execution_context.h"

#include "content/browser/ml/webnn/dml/adapter_dml.h"
#include "content/browser/ml/webnn/dml/upload_resource.h"

namespace content::webnn {

ExecutionContext::ExecutionContext(scoped_refptr<AdapterDML> adapter)
    : d3d12_device_(adapter->GetD3D12Device()),
      command_recorder_(adapter, adapter->GetDMLDevice()),
      command_queue_(adapter->GetCommandQueue()),
      resource_allocator_(adapter->GetResourceAllocator()) {}

ExecutionContext::~ExecutionContext() = default;

// Record a CopyBufferRegion for execution. Transition barriers are
// automatically inserted to transition the source and destination resources to
// COPY_SOURCE and COPY_DEST if necessary.
void ExecutionContext::CopyBufferRegion(ID3D12Resource* dest_resource,
                                        ID3D12Resource* src_resource,
                                        UINT64 resource_size,
                                        D3D12_RESOURCE_STATES state) {
  DCHECK(state == D3D12_RESOURCE_STATE_COPY_DEST ||
         state == D3D12_RESOURCE_STATE_COPY_SOURCE);

  D3D12_RESOURCE_BARRIER transition_barrier;
  // D3D12_RESOURCE_STATE_COPY_DEST is used to upload CPU data to GPU resource,
  // the resource state of source is GENERIC_READ that doesn't need to be
  // COPY_SOURCE. The destination resource state is UNORDERED_ACCESS that need
  // to transform to COPY_DEST.
  if (state == D3D12_RESOURCE_STATE_COPY_DEST) {
    transition_barrier.Transition.pResource = dest_resource;
  } else if (state == D3D12_RESOURCE_STATE_COPY_SOURCE) {
    // D3D12_RESOURCE_STATE_COPY_SOURCE is used to read back resource from GPU
    // to CPU buffer, the source resource state is UNORDERED_ACCESS that need to
    // transform to COPY_SOURCE, the destination resource state is COPY_DEST
    // when creating the committed resource, so the barrier is unnecessary for
    // it.
    transition_barrier.Transition.pResource = src_resource;
  }
  transition_barrier.Transition.StateBefore =
      D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  transition_barrier.Transition.StateAfter = state;
  transition_barrier.Transition.Subresource =
      D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
  transition_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  transition_barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
  command_recorder_.CopyBufferRegion(dest_resource, 0, src_resource, 0,
                                     resource_size);
  D3D12_RESOURCE_BARRIER reset_barrier = transition_barrier;
  reset_barrier.Transition.StateBefore = state;
  reset_barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  // Both transitions to/from UAV can be combined into a single ResourceBarrier
  // call, and the transition barrier command can be enqueued after
  // CopyBufferRegion.
  command_recorder_.ResourceBarrier({transition_barrier, reset_barrier});
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

void ExecutionContext::Flush() const {
  command_recorder_.CloseAndExecute();
}

void ExecutionContext::WaitForSignal() const {
  command_queue_->Wait();
  // Unlike ID3D12GraphicsCommandList::Reset, it is not recommended to call
  // Reset on the command allocator while a command list is still being
  // executed.
  command_recorder_.ResetCommandList();
}

void ExecutionContext::ReferenceUntilCompleted(ComPtr<IUnknown> object) {
  command_queue_->ReferenceUntilCompleted(std::move(object));
}

void ExecutionContext::ReleaseCompletedResources() const {
  command_queue_->ReleaseCompletedResources();
}

ComPtr<ID3D12Device> ExecutionContext::GetD3D12Device() const {
  return d3d12_device_;
}

ExecutionResources* ExecutionContext::GetExecutionResources() {
  return execution_resources_.get();
}

ComPtr<IDMLDevice> ExecutionContext::GetDMLDevice() {
  return command_recorder_.GetDMLDevice();
}

ComPtr<gpgmm::d3d12::ResourceAllocator>
ExecutionContext::GetResourceAllocator() {
  DCHECK(resource_allocator_.Get() != nullptr);
  return resource_allocator_;
}

}  // namespace content::webnn
