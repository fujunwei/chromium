// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CONTENT_BROWSER_ML_WEBNN_DML_EXECUTION_RESOURCES_H_
#define CONTENT_BROWSER_ML_WEBNN_DML_EXECUTION_RESOURCES_H_

#include <wrl.h>
#include <map>
#include <vector>

#include "DirectML.h"
#include "content/browser/ml/webnn/dml/graph_dml_impl.h"

namespace content::webnn {

using Microsoft::WRL::ComPtr;
class ExecutionContext;

enum class ResourceType {
  kInput = 0,
  kOutput = 1,
  kTemporary = 2,
  kPersistent = 3,
  kUnknow = 4,
};

// A unordered resources represent input, output, temporary and persistent
// resource.
class ExecutionResources final {
 public:
  explicit ExecutionResources(ExecutionContext* execution_context);
  ~ExecutionResources();

  // Allocate a resource that is released by manually.
  ComPtr<ID3D12Resource> Allocate(UINT64 resource_size);
  // Allocate a resource that is owned by a graph and reused for execution.
  ID3D12Resource* Allocate(ResourceType type,
                           UINT64 resource_size,
                           GraphDMLImpl* graph);
  ID3D12Resource* GetResource(GraphDMLImpl* graph, ResourceType type);
  // Free a resource that is owned by graph such as temporary, persistent,
  // input and output unordered resouce for execution.
  void Free(GraphDMLImpl* graph);

 private:
  struct Resources {
    Resources();
    Resources(ResourceType type, ComPtr<ID3D12Resource> resource);
    ~Resources();
    std::map<ResourceType, ComPtr<ID3D12Resource>> resources;
  };

  ExecutionContext* execution_context_;
  std::map<GraphDMLImpl*, Resources> pool_;
};

}  // namespace content::webnn

#endif  // CONTENT_BROWSER_ML_WEBNN_DML_EXECUTION_RESOURCES_H_
