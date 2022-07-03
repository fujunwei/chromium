// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CONTENT_BROWSER_ML_WEBNN_DML_UPLOAD_RESOURCE_H_
#define CONTENT_BROWSER_ML_WEBNN_DML_UPLOAD_RESOURCE_H_

#include <wrl.h>

#include "DirectML.h"
#include "components/ml/mojom/webnn_graph.mojom.h"
#include "content/browser/ml/webnn/dml/gpgmm_d3d12.h"
#include "content/browser/ml/webnn/dml/utils_dml.h"

namespace content::webnn {

using Microsoft::WRL::ComPtr;
using ml::webnn::mojom::ConstantsInfoPtr;
using ml::webnn::mojom::NamedResourcesPtr;
class ExecutionContext;

// A ring-buffer style upload heap for copying CPU data to GPU resources.
class UploadResource final {
 public:
  explicit UploadResource(ExecutionContext* execution_context);
  ~UploadResource();

  HRESULT UploadConstants(ID3D12Resource* dst_resource,
                          ConstantsInfoPtr& constants_info);
  HRESULT UploadInputs(ID3D12Resource* dst_resource,
                       NamedResourcesPtr& named_inputs);

 private:
  HRESULT CreateUploadResource(size_t byte_length);

  ExecutionContext* execution_context_;
  ComPtr<gpgmm::d3d12::ResourceAllocation> upload_resource_;
};

}  // namespace content::webnn

#endif  // CONTENT_BROWSER_ML_WEBNN_DML_UPLOAD_RESOURCE_H_
