// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/browser/ml/webnn/dml/graph_dml_impl.h"

#include "base/memory/ptr_util.h"
#include "content/browser/ml/webnn/dml/native/GraphDML.h"
#include "mojo/public/cpp/bindings/self_owned_receiver.h"

namespace content {

namespace {

using ml::webnn::mojom::Graph;

}  // namespace

namespace webnn {

// static
void GraphDMLImpl::Create(mojo::PendingReceiver<Graph> receiver) {
  mojo::MakeSelfOwnedReceiver<Graph>(base::WrapUnique(new GraphDMLImpl()),
                                     std::move(receiver));
}

GraphDMLImpl::~GraphDMLImpl() = default;

GraphDMLImpl::GraphDMLImpl()
    : native_graph_dml_(std::make_unique<GraphDMLNativeImpl>()) {}

void GraphDMLImpl::AddInput(const std::string& name,
                            OperandDescriptorPtr desc) {
  // TODO: return directly if BuildResult has error message.
  native_graph_dml_->AddInput(name, std::move(desc));
}

void GraphDMLImpl::AddConstant(OperandDescriptorPtr desc,
                               const std::vector<uint8_t>& array_buffer) {
  // TODO: return directly if BuildResult has error message.
  native_graph_dml_->AddConstant(std::move(desc), array_buffer);
}

void GraphDMLImpl::AddOutput(const std::string& name, uint32_t operand_id) {
  // TODO: return directly if BuildResult has error message.
  native_graph_dml_->AddOutput(name, operand_id);
}

void GraphDMLImpl::AddElementWiseBinary(uint32_t a_id,
                                        uint32_t b_id,
                                        BinaryOperandType type,
                                        OperandDescriptorPtr desc) {
  // TODO: return directly if BuildResult has error message.
  native_graph_dml_->AddElementWiseBinary(a_id, b_id, type, std::move(desc));
}

void GraphDMLImpl::AddClamp(uint32_t input_id,
                            ClampOptionsPtr options,
                            OperandDescriptorPtr desc) {
  // TODO: return directly if BuildResult has error message.
  native_graph_dml_->AddClamp(input_id, std::move(options), std::move(desc));
}

void GraphDMLImpl::AddConv2d(uint32_t input_id,
                             uint32_t filter_id,
                             Conv2dOptionsPtr options,
                             OperandDescriptorPtr desc) {
  // TODO: return directly if BuildResult has error message.
  native_graph_dml_->AddConv2d(input_id, filter_id, std::move(options),
                               std::move(desc));
}

void GraphDMLImpl::AddReshape(uint32_t input_id, OperandDescriptorPtr desc) {
  // TODO: return directly if BuildResult has error message.
  native_graph_dml_->AddReshape(input_id, std::move(desc));
}

void GraphDMLImpl::AddGemm(uint32_t a_id,
                           uint32_t b_id,
                           GemmOptionsPtr options,
                           OperandDescriptorPtr desc) {
  // TODO: return directly if BuildResult has error message.
  native_graph_dml_->AddGemm(a_id, b_id, std::move(options), std::move(desc));
}

void GraphDMLImpl::AddPool2d(uint32_t input_id,
                             Pool2dOptionsPtr options,
                             Pool2dType type,
                             OperandDescriptorPtr desc) {
  // TODO: return directly if BuildResult has error message.
  native_graph_dml_->AddPool2d(input_id, std::move(options), type,
                               std::move(desc));
}

void GraphDMLImpl::AddUnary(uint32_t input_id,
                            UnaryOperandType type,
                            OperandDescriptorPtr desc) {
  // TODO: return directly if BuildResult has error message.
  native_graph_dml_->AddUnary(input_id, type, std::move(desc));
}

void GraphDMLImpl::AddFusionClamp(ClampOptionsPtr options,
                                  uint32_t operator_id) {
  native_graph_dml_->AddFusionClamp(std::move(options), operator_id);
}

void GraphDMLImpl::BuildAsync(BuildAsyncCallback callback) {
  std::move(callback).Run(native_graph_dml_->CompileImpl());
}

void GraphDMLImpl::ComputeAsync(NamedInputsPtr named_inputs,
                                const std::vector<std::string>& output_names,
                                ComputeAsyncCallback callback) {
  std::vector<std::vector<uint8_t>> output_buffers;
  ComputeResult result = native_graph_dml_->ComputeImpl(
      std::move(named_inputs), output_names, output_buffers);
  std::move(callback).Run(result, output_buffers);
}

}  // namespace webnn

}  // namespace content
