// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CONTENT_BROWSER_ML_WEBNN_DML_COMMAND_QUEUE_H_
#define CONTENT_BROWSER_ML_WEBNN_DML_COMMAND_QUEUE_H_

#include <wrl.h>
#include <deque>

#include "DirectML.h"
#include "base/memory/ref_counted.h"

namespace content::webnn {

using Microsoft::WRL::ComPtr;

// There is only one D3D12 command queue wrapped an existing queue that will
// be shared with WebNN context, and provides a waitable fence which is signaled
// with a increasing value once the execution complete on the GPU.
class CommandQueue final : public base::RefCounted<CommandQueue> {
 public:
  CommandQueue();

  CommandQueue(const CommandQueue&) = delete;
  CommandQueue& operator=(const CommandQueue&) = delete;

  HRESULT Initialize(ID3D12Device* d3d12_device);
  void ReferenceUntilCompleted(ComPtr<IUnknown> object);
  HRESULT ExecuteCommandLists(std::vector<ID3D12CommandList*>);
  // Queues a wait to block the GPU until the fence is signaled with the last
  // value.
  void Wait();
  void ReleaseCompletedResources();

 private:
  friend class base::RefCounted<CommandQueue>;
  ~CommandQueue();

  struct QueuedObject {
    QueuedObject();
    QueuedObject(const QueuedObject& other);
    QueuedObject(uint64_t fence_value, ComPtr<IUnknown> object);
    ~QueuedObject();

    uint64_t fence_value;
    ComPtr<IUnknown> object;
  };
  std::deque<QueuedObject> queued_object_refs_;

  ComPtr<ID3D12CommandQueue> command_queue_;
  // the fence value used to watch the progression of GPU execution on a queue
  // that is incremented by one time. This way to know if something is done
  // executing, we just need to compare its value with the currently completed
  // value.
  uint64_t last_fence_value_ = 0;
  ComPtr<ID3D12Fence> fence_;
  HANDLE fence_event_ = nullptr;
};

}  // namespace content::webnn

#endif  // CONTENT_BROWSER_ML_WEBNN_DML_COMMAND_QUEUE_H_
