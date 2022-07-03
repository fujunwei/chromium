// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CONTENT_BROWSER_ML_WEBNN_GRAPH_DML_IMPL_H_
#define CONTENT_BROWSER_ML_WEBNN_GRAPH_DML_IMPL_H_

#include "components/ml/webnn/mojom/graph.mojom.h"
#include "mojo/public/cpp/bindings/pending_receiver.h"
#include "mojo/public/cpp/system/handle.h"

namespace content {

namespace {

using ml::webnn::mojom::BinaryOperandType;
using ml::webnn::mojom::ClampOptionsPtr;
using ml::webnn::mojom::Conv2dOptionsPtr;
using ml::webnn::mojom::GemmOptionsPtr;
using ml::webnn::mojom::NamedInputsPtr;
using ml::webnn::mojom::OperandDescriptorPtr;
using ml::webnn::mojom::Pool2dOptionsPtr;
using ml::webnn::mojom::Pool2dType;
using ml::webnn::mojom::UnaryOperandType;

}  // namespace

namespace webnn {

class GraphDMLNativeImpl;

class GraphDMLImpl : public ml::webnn::mojom::Graph {
 public:
  ~GraphDMLImpl() override;
  static void Create(mojo::PendingReceiver<ml::webnn::mojom::Graph> receiver);

  GraphDMLImpl(const GraphDMLImpl&) = delete;
  GraphDMLImpl& operator=(const GraphDMLImpl&) = delete;

 protected:
  GraphDMLImpl();

 private:
  // ml::webnn::mojom::Graph
  void AddInput(const std::string&, OperandDescriptorPtr) override;
  void AddConstant(OperandDescriptorPtr, const std::vector<uint8_t>&) override;
  void AddOutput(const std::string& name, uint32_t operand_id) override;
  void AddElementWiseBinary(uint32_t,
                            uint32_t,
                            BinaryOperandType,
                            OperandDescriptorPtr) override;
  void AddClamp(uint32_t input_id,
                ClampOptionsPtr options,
                OperandDescriptorPtr desc) override;
  void AddConv2d(uint32_t input_id,
                 uint32_t filter_id,
                 Conv2dOptionsPtr options,
                 OperandDescriptorPtr desc) override;
  void AddReshape(uint32_t input_id, OperandDescriptorPtr desc) override;
  void AddGemm(uint32_t,
               uint32_t,
               GemmOptionsPtr,
               OperandDescriptorPtr) override;
  void AddPool2d(uint32_t input_id,
                 Pool2dOptionsPtr options,
                 Pool2dType type,
                 OperandDescriptorPtr desc) override;
  void AddUnary(uint32_t input_id,
                UnaryOperandType type,
                OperandDescriptorPtr desc) override;
  void AddFusionClamp(ClampOptionsPtr options, uint32_t operator_id) override;

  void BuildGraph(BuildGraphCallback callback) override;
  void ComputeGraph(NamedInputsPtr named_inputs,
                    ComputeGraphCallback callback) override;
};

}  // namespace webnn
}  // namespace content

#endif  // CONTENT_BROWSER_ML_WEBNN_GRAPH_DML_IMPL_H_
