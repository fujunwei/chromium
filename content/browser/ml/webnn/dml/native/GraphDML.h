// Copyright 2021 The WebNN-native Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef WEBNN_NATIVE_DML_MODEL_DML_H_
#define WEBNN_NATIVE_DML_MODEL_DML_H_

#define DML_TARGET_VERSION_USE_LATEST 1

#include <dxgi1_4.h>
#include <dxgi1_6.h>
#include <wrl\client.h>
#include <unordered_map>
#include <unordered_set>

#include "DirectML.h"
#include "components/ml/webnn/mojom/graph.mojom.h"

namespace content {

namespace webnn {

using namespace Microsoft::WRL;
using ml::webnn::mojom::BinaryOperandType;
using ml::webnn::mojom::BuildResult;
using ml::webnn::mojom::ClampOptions;
using ml::webnn::mojom::ClampOptionsPtr;
using ml::webnn::mojom::ComputeResult;
using ml::webnn::mojom::Conv2dOptionsPtr;
using ml::webnn::mojom::FusionOperator;
using ml::webnn::mojom::GemmOptionsPtr;
using ml::webnn::mojom::NamedInputsPtr;
using ml::webnn::mojom::NamedOutputsPtr;
using ml::webnn::mojom::OperandDescriptorPtr;
using ml::webnn::mojom::Pool2dOptions;
using ml::webnn::mojom::Pool2dOptionsPtr;
using ml::webnn::mojom::Pool2dType;
using ml::webnn::mojom::UnaryOperandType;

class FusionOperators;

struct MemoryInfo {
  MemoryInfo();
  ~MemoryInfo();
  size_t byte_offset;
  size_t byte_length;
};

// Represent the DirectML tensor description.
struct DmlTensorDesc {
  DmlTensorDesc();
  ~DmlTensorDesc();
  std::vector<UINT> dimensions = {};
  std::vector<UINT> strides = {};
  // Describes a tensor that will be stored in a Direct3D 12 buffer resource.
  DML_BUFFER_TENSOR_DESC bufferDesc = {};
};

// Represent the information of the graph's edges.
struct EdgeInfoBase {
  virtual ~EdgeInfoBase() = default;
  DML_TENSOR_DESC outputTensorDESC = {};
  std::string name = "";
  bool isInputEdge = false;
};

// Only represent the information of the input edges.
struct InputEdgeInfo final : public EdgeInfoBase {
  ~InputEdgeInfo() override = default;
  // Indicate the index of the graph's input.
  size_t inputIndex = 0;
  void const* buffer = nullptr;
  size_t byteLength = 0;
  // Indicate if the input is from constant buffer which need to be
  // uploaded in the stage of initialization.
  bool isConstantInput = false;
};

// Represent the information of the intermediate edges and output edges.
struct EdgeInfo final : public EdgeInfoBase {
  ~EdgeInfo() override = default;
  // Indicate the index of the intermediate node from which this edge was
  // produced.
  uint32_t nodeIndex = 0;
  // Indicate the index of the intermediate node' output from which this edge
  // was produced.
  uint32_t outputNodeIndex = 0;
};

class GraphDMLNativeImpl {
 public:
  GraphDMLNativeImpl();
  ~GraphDMLNativeImpl();

  void AddConstant(OperandDescriptorPtr, const std::vector<uint8_t>&);
  void AddInput(const std::string&, OperandDescriptorPtr);
  void AddOutput(const std::string&, uint32_t);
  void AddElementWiseBinary(uint32_t,
                            uint32_t,
                            BinaryOperandType,
                            OperandDescriptorPtr);
  void AddConv2d(uint32_t, uint32_t, Conv2dOptionsPtr, OperandDescriptorPtr);
  void AddPool2d(uint32_t input_id,
                 Pool2dOptionsPtr options,
                 Pool2dType type,
                 OperandDescriptorPtr desc);
  void AddReshape(uint32_t, OperandDescriptorPtr);
  void AddUnary(uint32_t, UnaryOperandType, OperandDescriptorPtr);
  void AddGemm(uint32_t a_id,
               uint32_t b_id,
               GemmOptionsPtr options,
               OperandDescriptorPtr desc);
  void AddClamp(uint32_t, ClampOptionsPtr, OperandDescriptorPtr);
  void AddFusionClamp(ClampOptionsPtr options, uint32_t operator_id);
  BuildResult CompileImpl();
  ComputeResult ComputeImpl(NamedInputsPtr named_inputs,
                            NamedOutputsPtr& named_outputs);

  void AddEdgesToThisNode(
      std::vector<std::shared_ptr<EdgeInfoBase>> inputNodes);
  void FillUploadResourceAndInputBindings(
      uint64_t uploadResourceSize,
      std::vector<DML_BUFFER_BINDING>& inputBufferBinding,
      NamedInputsPtr namedInputs);
  std::shared_ptr<EdgeInfoBase> Clamp(std::shared_ptr<EdgeInfoBase> inputEdge,
                                      const ClampOptions* options);
  void EmulateFusedOperator(const FusionOperator* activation,
                            std::shared_ptr<EdgeInfoBase>& inputEdge,
                            const std::vector<UINT>& inputDims);
  void TransposeOutputToNhwc(std::shared_ptr<EdgeInfoBase>& inputEdge,
                             const std::vector<UINT>& nchwOutputDims);

 private:
  BuildResult Finish();

  // Represents a DirectML device, which is used to create operators, binding
  // tables, command recorders, and other objects.
  ComPtr<IDMLDevice> mDevice;
  // The IDMLDevice1 interface inherits from IDMLDevice.
  ComPtr<IDMLDevice1> mDevice1;
  // Represents a virtual adapter; it is used to create command allocators,
  // command lists, command queues, fences, resources, pipeline state objects,
  // heaps, root signatures, samplers, and many resource views.
  ComPtr<ID3D12Device> mD3D12Device;

  ComPtr<IDMLCommandRecorder> mCommandRecorder;
  ComPtr<ID3D12CommandQueue> mCommandQueue;
  ComPtr<ID3D12CommandAllocator> mCommandAllocator;
  ComPtr<ID3D12GraphicsCommandList> mCommandList;
  ComPtr<IDMLBindingTable> mBindingTable;
  ComPtr<ID3D12DescriptorHeap> mDescriptorHeap;

  ComPtr<ID3D12Resource> mUploadResource;
  ComPtr<ID3D12Resource> mInputResource;
  ComPtr<ID3D12Resource> mOutputResource;
  ComPtr<ID3D12Resource> mReadBackResource;
  ComPtr<ID3D12Resource> mTemporaryResource;
  ComPtr<ID3D12Resource> mPersistentResource;

  uint64_t mCommonInputsResourceSize = 0;
  uint64_t mOutputsResourceSize = 0;
  UINT64 mTemporaryResourceSize = 0;
  UINT64 mPersistentResourceSize = 0;

  // Describe a graph of DirectML operators used to compile a combined,
  // optimized operator.
  std::vector<std::shared_ptr<InputEdgeInfo>> mInputs;
  std::vector<EdgeInfo> mOutputs;
  std::vector<DML_GRAPH_NODE_DESC> mIntermediateNodes;
  std::vector<DML_GRAPH_EDGE_DESC> mInputEdges;
  std::vector<DML_GRAPH_EDGE_DESC> mOutputEdges;
  std::vector<DML_GRAPH_EDGE_DESC> mIntermediateEdges;

  // IDMLCompiledOperator represents the DirectML graph's output which need to
  // be initialized by IDMLOperatorInitializer.
  ComPtr<IDMLCompiledOperator> mCompiledOperator;
  DML_BINDING_TABLE_DESC mBindingTableDesc;

  std::map<uint32_t, std::shared_ptr<EdgeInfoBase>> mGraphEdgesMap;

  // Keep intermediate nodes here to avoid releasing too early.
  std::map<uint32_t, ComPtr<IDMLOperator>> mIntermediateNodesMap;
  // Keep the input tensors description here to avoid releasing too early.
  std::vector<std::shared_ptr<DmlTensorDesc>> mDmlTensorsDesc;
  // Keep the descriptions of nodes and edges here to avoid releasing too early.
  std::vector<std::unique_ptr<DML_OPERATOR_GRAPH_NODE_DESC>>
      mIntermediateNodesDesc;
  std::vector<std::unique_ptr<DML_INPUT_GRAPH_EDGE_DESC>> mInputEdgesDesc;
  std::vector<std::unique_ptr<DML_OUTPUT_GRAPH_EDGE_DESC>> mOutputEdgesDesc;
  std::vector<std::unique_ptr<DML_INTERMEDIATE_GRAPH_EDGE_DESC>>
      mIntermediateEdgesDesc;
  std::vector<std::unique_ptr<char>> mConstantsBuffer;

  std::string error_messages_;
  BuildResult build_result_;
  std::unique_ptr<FusionOperators> fusion_operators_;

  std::map<std::string, MemoryInfo> outputs_info_map_;
  std::map<std::string, mojo::ScopedSharedBufferMapping> named_buffer_map_;
  mojo::ScopedSharedBufferHandle outputs_buffer_;
};

}  // namespace webnn
}  // namespace content

#endif  // WEBNN_NATIVE_DML_MODEL_DML_H_