// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/browser/ml/webnn/graph_dml_impl.h"

#include "base/memory/ptr_util.h"
#include "mojo/public/cpp/bindings/self_owned_receiver.h"

namespace content {

namespace {

using ml::webnn::mojom::BuildResult;
using ml::webnn::mojom::ComputeResult;
using ml::webnn::mojom::Graph;

}  // namespace

namespace webnn {

// static
void GraphDMLImpl::Create(mojo::PendingReceiver<Graph> receiver) {
  mojo::MakeSelfOwnedReceiver<Graph>(base::WrapUnique(new GraphDMLImpl()),
                                     std::move(receiver));
}

GraphDMLImpl::~GraphDMLImpl() = default;

GraphDMLImpl::GraphDMLImpl() {}

void GraphDMLImpl::AddInput(const std::string& name,
                            OperandDescriptorPtr desc) {
  // TODO(crbug.com/1273291): Implement this with DirectML.
  NOTIMPLEMENTED();
}

void GraphDMLImpl::AddConstant(OperandDescriptorPtr desc,
                               const std::vector<uint8_t>& array_buffer) {
  // TODO(crbug.com/1273291): Implement this with DirectML.
  NOTIMPLEMENTED();
}

void GraphDMLImpl::AddOutput(const std::string& name, uint32_t operand_id) {
  // TODO(crbug.com/1273291): Implement this with DirectML.
  NOTIMPLEMENTED();
}

void GraphDMLImpl::AddElementWiseBinary(uint32_t a_id,
                                        uint32_t b_id,
                                        BinaryOperandType type,
                                        OperandDescriptorPtr output_desc) {
  // TODO(crbug.com/1273291): Implement this with DirectML.
  NOTIMPLEMENTED();
}

void GraphDMLImpl::AddClamp(uint32_t input_id,
                            ClampOptionsPtr options,
                            OperandDescriptorPtr output_desc) {
  // TODO(crbug.com/1273291): Implement this with DirectML.
  NOTIMPLEMENTED();
}

void GraphDMLImpl::AddConv2d(uint32_t input_id,
                             uint32_t filter_id,
                             Conv2dOptionsPtr options,
                             OperandDescriptorPtr output_desc) {
  // TODO(crbug.com/1273291): Implement this with DirectML.
  NOTIMPLEMENTED();
}

void GraphDMLImpl::AddReshape(uint32_t input_id,
                              OperandDescriptorPtr output_desc) {
  // TODO(crbug.com/1273291): Implement this with DirectML.
  NOTIMPLEMENTED();
}

void GraphDMLImpl::AddGemm(uint32_t a_id,
                           uint32_t b_id,
                           GemmOptionsPtr options,
                           OperandDescriptorPtr output_desc) {
  // TODO(crbug.com/1273291): Implement this with DirectML.
  NOTIMPLEMENTED();
}

void GraphDMLImpl::AddPool2d(uint32_t input_id,
                             Pool2dOptionsPtr options,
                             Pool2dType type,
                             OperandDescriptorPtr output_desc) {
  // TODO(crbug.com/1273291): Implement this with DirectML.
  NOTIMPLEMENTED();
}

void GraphDMLImpl::AddUnary(uint32_t input_id,
                            UnaryOperandType type,
                            OperandDescriptorPtr output_desc) {
  // TODO(crbug.com/1273291): Implement this with DirectML.
  NOTIMPLEMENTED();
}

void GraphDMLImpl::AddFusionClamp(ClampOptionsPtr options,
                                  uint32_t operator_id) {
  // TODO(crbug.com/1273291): Implement this with DirectML.
  NOTIMPLEMENTED();
}

void GraphDMLImpl::BuildGraph(BuildGraphCallback callback) {
  // TODO(crbug.com/1273291): Implement this with DirectML.
  NOTIMPLEMENTED();
  std::move(callback).Run(BuildResult::kOk);
}

void GraphDMLImpl::ComputeGraph(NamedInputsPtr named_inputs,
                                ComputeGraphCallback callback) {
  // TODO(crbug.com/1273291): Implement this with DirectML.
  NOTIMPLEMENTED();
  auto named_outputs = ml::webnn::mojom::NamedOutputs::New();
  std::move(callback).Run(ComputeResult::kOk, std::move(named_outputs));
}

}  // namespace webnn

}  // namespace content
