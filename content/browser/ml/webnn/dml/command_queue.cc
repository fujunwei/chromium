// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/browser/ml/webnn/dml/command_queue.h"

namespace content::webnn {

CommandQueue::~CommandQueue() = default;

CommandQueue::CommandQueue() {}

HRESULT CommandQueue::Initialize(ID3D12Device* d3d12_device) {
  D3D12_COMMAND_QUEUE_DESC command_queue_desc = {};
  command_queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
  command_queue_desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
  HRESULT hr = d3d12_device->CreateCommandQueue(&command_queue_desc,
                                                IID_PPV_ARGS(&command_queue_));
  if (FAILED(hr)) {
    return hr;
  }

  hr = d3d12_device->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                 IID_PPV_ARGS(&fence_));
  if (FAILED(hr)) {
    return hr;
  }
  fence_event_ = CreateEvent(nullptr, FALSE, FALSE, nullptr);
  DCHECK(fence_event_ != nullptr);

  return S_OK;
}

void CommandQueue::ReferenceUntilCompleted(ComPtr<IUnknown> object) {
  QueuedObject object_ref = {last_fence_value_, std::move(object)};
  queued_object_refs_.push_back(object_ref);
}

HRESULT CommandQueue::ExecuteCommandLists(
    std::vector<ID3D12CommandList*> command_lists) {
  command_queue_->ExecuteCommandLists(command_lists.size(),
                                      command_lists.data());
  ++last_fence_value_;
  return command_queue_->Signal(fence_.Get(), last_fence_value_);
}

void CommandQueue::Wait() {
  if (fence_->GetCompletedValue() >= last_fence_value_) {
    return;
  }
  HRESULT hr = fence_->SetEventOnCompletion(last_fence_value_, fence_event_);
  if (FAILED(hr)) {
    return;
  }
  WaitForSingleObject(fence_event_, INFINITE);
}

void CommandQueue::ReleaseCompletedResources() {
  uint64_t completed_value = fence_->GetCompletedValue();
  while (!queued_object_refs_.empty() &&
         queued_object_refs_.front().fence_value <= completed_value) {
    queued_object_refs_.pop_front();
  }
}

CommandQueue::QueuedObject::QueuedObject(uint64_t fence_value,
                                         ComPtr<IUnknown> object) {
  this->fence_value = fence_value;
  this->object = std::move(object);
}

CommandQueue::QueuedObject::QueuedObject(const QueuedObject& other) {
  this->fence_value = other.fence_value;
  this->object = std::move(other.object);
}

CommandQueue::QueuedObject::QueuedObject() = default;
CommandQueue::QueuedObject::~QueuedObject() = default;

}  // namespace content::webnn
