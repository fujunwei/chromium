// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CONTENT_BROWSER_ML_WEBNN_DML_EXECUTION_CONTEXT_H_
#define CONTENT_BROWSER_ML_WEBNN_DML_EXECUTION_CONTEXT_H_

#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl.h>

#include "base/memory/ref_counted.h"
#include "content/browser/ml/webnn/dml/command_queue.h"
#include "content/browser/ml/webnn/dml/command_recorder.h"
#include "content/browser/ml/webnn/dml/execution_resources.h"
#include "content/browser/ml/webnn/dml/gpgmm_d3d12.h"
#include "content/browser/ml/webnn/dml/graph_dml_impl.h"

namespace content::webnn {

using Microsoft::WRL::ComPtr;
class AdapterDML;

class ExecutionContext final : public base::RefCounted<ExecutionContext> {
 public:
  explicit ExecutionContext(scoped_refptr<AdapterDML> adapter);

  ExecutionContext(const ExecutionContext&) = delete;
  ExecutionContext& operator=(const ExecutionContext&) = delete;

  HRESULT Initialize();

  void CopyBufferRegion(ID3D12Resource* dest_resource,
                        ID3D12Resource* src_resource,
                        UINT64 resource_size,
                        D3D12_RESOURCE_STATES state);

  HRESULT InitializeGraph(GraphDMLImpl* graph,
                          IDMLCompiledOperator* compiled_operator,
                          const DML_BINDING_DESC& input_array_binding);

  HRESULT ExecuteGraph(GraphDMLImpl* graph,
                       IDMLCompiledOperator* compiled_operator,
                       const std::vector<DML_BINDING_DESC>& input_bindings,
                       const std::vector<DML_BINDING_DESC>& output_bindings);

  // Forces all queued work to begin executing on the GPU.
  void Flush() const;
  // Blocks until the current fence is signaled.
  void WaitForSignal() const;
  void ReferenceUntilCompleted(ComPtr<IUnknown> object);
  void ReleaseCompletedResources() const;

  ComPtr<ID3D12Device> GetD3D12Device() const;
  ComPtr<IDMLDevice> GetDMLDevice();
  ExecutionResources* GetExecutionResources();
  ComPtr<gpgmm::d3d12::ResourceAllocator> GetResourceAllocator();

 private:
  friend class base::RefCounted<ExecutionContext>;
  ~ExecutionContext();

  // Device is owned by adapter.
  ComPtr<ID3D12Device> d3d12_device_;
  // There is one active command recorder at a time.
  CommandRecorder command_recorder_;
  scoped_refptr<CommandQueue> command_queue_;

  // ResourceAllocator is owned by adapter
  ComPtr<gpgmm::d3d12::ResourceAllocator> resource_allocator_;
  std::unique_ptr<ExecutionResources> execution_resources_;
};

}  // namespace content::webnn

#endif  // CONTENT_BROWSER_ML_WEBNN_DML_EXECUTION_CONTEXT_H_
