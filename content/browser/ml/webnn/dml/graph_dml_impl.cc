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

void GraphDMLImpl::BuildAsync(BuildAsyncCallback callback) {
  std::move(callback).Run(native_graph_dml_->CompileImpl());
}

void GraphDMLImpl::ComputeAsync(
    const base::flat_map<std::string, std::vector<uint8_t>>& named_inputs,
    const std::vector<std::string>& output_names,
    ComputeAsyncCallback callback) {
  std::vector<std::vector<uint8_t>> output_buffers;
  ComputeResult result = native_graph_dml_->ComputeImpl(
      named_inputs, output_names, output_buffers);
  std::move(callback).Run(result, output_buffers);
}

}  // namespace webnn

}  // namespace content
