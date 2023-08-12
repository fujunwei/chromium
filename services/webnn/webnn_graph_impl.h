// Copyright 2023 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef SERVICES_WEBNN_WEBNN_GRAPH_IMPL_H_
#define SERVICES_WEBNN_WEBNN_GRAPH_IMPL_H_

#include <memory>
#include <string>

#include "base/containers/flat_map.h"
#include "mojo/public/cpp/base/big_buffer.h"
#include "services/webnn/public/mojom/webnn_graph.mojom.h"
#include "services/webnn/public/mojom/webnn_service.mojom.h"

namespace webnn {

class WebNNGraphImpl : public mojom::WebNNGraph {
 public:
  // The members of `ComputeBufferValidator` are used to validate the inputs
  // of a graph execution. The input name and byte length of computation must
  // match graph's expectation, the output name and byte length are used to
  // create the result of computation.
  class ComputeBufferValidator {
   public:
    explicit ComputeBufferValidator(const mojom::GraphInfoPtr& graph_info);
    ~ComputeBufferValidator();

    ComputeBufferValidator(const ComputeBufferValidator&) = delete;
    ComputeBufferValidator& operator=(const ComputeBufferValidator&) = delete;

    // Validate the built graph's expected input matches what we received from a
    // compute call.
    bool Validate(
        const base::flat_map<std::string, mojo_base::BigBuffer>& inputs);

   private:
    std::map<std::string, size_t> input_byte_length_map;
    // TODO(crbug.com/1455278): Add output information.
    // std::map<std::string, size_t> output_byte_length_map_;
  };

  explicit WebNNGraphImpl(
      std::unique_ptr<ComputeBufferValidator> compute_buffer_validator);
  WebNNGraphImpl(const WebNNGraphImpl&) = delete;
  WebNNGraphImpl& operator=(const WebNNGraphImpl&) = delete;
  ~WebNNGraphImpl() override;

  // Return false if the graph is invalid.
  static bool ValidateGraph(const mojom::GraphInfoPtr& graph_info);

 private:
  // The validator is to make sure the inputs from a compute call match the
  // built graph's expected.
  std::unique_ptr<ComputeBufferValidator> compute_buffer_validator_;

  // mojom::WebNNGraph
  void Compute(base::flat_map<std::string, mojo_base::BigBuffer> named_inputs,
               mojom::WebNNGraph::ComputeCallback callback) override;

  // An WebNNGraph backend should implement this method to execute the compiled
  // platform graph asynchronously.
  virtual void ComputeImpl(
      base::flat_map<std::string, mojo_base::BigBuffer> named_inputs,
      mojom::WebNNGraph::ComputeCallback callback) = 0;
};

}  // namespace webnn

#endif  // SERVICES_WEBNN_WEBNN_GRAPH_IMPL_H_
