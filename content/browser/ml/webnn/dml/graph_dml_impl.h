// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CONTENT_BROWSER_ML_WEBNN_DML_GRAPH_DML_IMPL_H_
#define CONTENT_BROWSER_ML_WEBNN_DML_GRAPH_DML_IMPL_H_

#define DML_TARGET_VERSION_USE_LATEST 1

#include <dxgi1_4.h>
#include <dxgi1_6.h>
#include <wrl\client.h>
#include <unordered_map>
#include <unordered_set>

#include <algorithm>
#include "DirectML.h"
#include "base/memory/read_only_shared_memory_region.h"
#include "base/memory/shared_memory_mapping.h"
#include "components/ml/mojom/webnn_graph.mojom.h"
#include "content/browser/ml/webnn/dml/gpgmm_d3d12.h"
#include "content/browser/ml/webnn/dml/graph_desc_builder.h"
#include "content/browser/ml/webnn/dml/graph_node_output.h"
#include "content/browser/ml/webnn/dml/readback_resource.h"
#include "content/browser/ml/webnn/dml/upload_resource.h"
#include "mojo/public/cpp/bindings/pending_receiver.h"
#include "utils_dml.h"

namespace content::webnn {

namespace {

using Microsoft::WRL::ComPtr;
using ml::webnn::mojom::BuildResult;
using ml::webnn::mojom::ClampOptions;
using ml::webnn::mojom::ClampOptionsPtr;
using ml::webnn::mojom::ComputeResult;
using ml::webnn::mojom::ConstantsInfoPtr;
using ml::webnn::mojom::Conv2dOptionsPtr;
using ml::webnn::mojom::ElementWiseBinaryType;
using ml::webnn::mojom::GemmOptionsPtr;
using ml::webnn::mojom::ModelInfoPtr;
using ml::webnn::mojom::NamedResourcesPtr;
using ml::webnn::mojom::OperandDescriptorPtr;
using ml::webnn::mojom::OperationInfo;
using ml::webnn::mojom::OperationInfoPtr;
using ml::webnn::mojom::Pool2dOptions;
using ml::webnn::mojom::Pool2dOptionsPtr;
using ml::webnn::mojom::Pool2dType;

}  // namespace

class ExecutionContext;

class GraphDMLImpl : public ml::webnn::mojom::Graph {
 public:
  ~GraphDMLImpl() override;
  static void Create(mojo::PendingReceiver<ml::webnn::mojom::Graph> receiver,
                     scoped_refptr<ExecutionContext> execution_context);

  GraphDMLImpl(const GraphDMLImpl&) = delete;
  GraphDMLImpl& operator=(const GraphDMLImpl&) = delete;

 protected:
  GraphDMLImpl(scoped_refptr<ExecutionContext> execution_context);

 private:
  // ml::webnn::mojom::Graph
  void AddInput(const std::string&, OperandDescriptorPtr, UINT64 index);
  void AddConstant(OperandDescriptorPtr, UINT64 index);
  void AddClamp(UINT64 input_index,
                ClampOptionsPtr options,
                UINT64 output_index);
  void AddConv2d(UINT64 input_index,
                 UINT64 filter_index,
                 Conv2dOptionsPtr options,
                 OperandDescriptorPtr desc,
                 UINT64 output_index);
  void AddElementWiseBinary(UINT64,
                            UINT64,
                            ElementWiseBinaryType,
                            OperandDescriptorPtr,
                            UINT64 output_index);
  void AddGemm(UINT64,
               UINT64,
               GemmOptionsPtr,
               OperandDescriptorPtr,
               UINT64 output_index);
  void AddPool2d(UINT64 input_index,
                 Pool2dOptionsPtr options,
                 Pool2dType type,
                 OperandDescriptorPtr desc,
                 UINT64 output_index);
  void AddRelu(UINT64 input_index,
               OperandDescriptorPtr desc,
               UINT64 output_index);
  void AddReshape(UINT64 input_index,
                  OperandDescriptorPtr desc,
                  UINT64 output_index);
  void AddSoftmax(UINT64 input_index,
                  OperandDescriptorPtr desc,
                  UINT64 output_index);

  void Build(ModelInfoPtr model_info, BuildCallback callback) override;
  void Compute(NamedResourcesPtr named_inputs,
               ComputeCallback callback) override;

  std::unique_ptr<NodeOutput> Clamp(NodeOutput* input_node,
                                    const ClampOptions* options);
  void EmulateFusedOperator(const OperationInfo* activation,
                            std::unique_ptr<NodeOutput>& input_node,
                            const std::vector<UINT>& inputDims);
  void TransposeOutputToNhwc(std::unique_ptr<NodeOutput>& input_node,
                             const std::vector<UINT>& nchwOutputDims);

  void AddOutput(const std::string&, UINT64);

  scoped_refptr<ExecutionContext> execution_context_;
  std::unique_ptr<UploadResource> input_resource_uploader_;
  std::unique_ptr<ReadbackResource> output_resource_readback_;
  std::unique_ptr<GraphDescBuilder> graph_desc_builder_;

  // IDMLCompiledOperator represents the DirectML graph's output which need to
  // be initialized by IDMLOperatorInitializer.
  ComPtr<IDMLCompiledOperator> mCompiledOperator;

  std::map<UINT64, std::unique_ptr<NodeOutput>> node_output_map_;

  std::string error_messages_;
  BuildResult build_result_;
};

}  // namespace content::webnn

#endif  // CONTENT_BROWSER_ML_WEBNN_DML_GRAPH_DML_IMPL_H_
