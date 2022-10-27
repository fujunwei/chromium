// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_MOJO_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_MOJO_H_

#include "components/ml/mojom/webnn_graph.mojom-blink.h"
#include "components/ml/mojom/webnn_service.mojom-blink.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operand.h"
#include "third_party/blink/renderer/platform/heap/visitor.h"
#include "third_party/blink/renderer/platform/mojo/heap_mojo_remote.h"

namespace blink {

using ml::webnn::mojom::blink::NamedResourcesPtr;

class MLContext;
class ScriptPromiseResolver;

class MODULES_EXPORT MLGraphMojo final : public MLGraph {
 public:
  // Create and build an MLGraphMojo object. Resolve the promise with
  // this concrete object if the graph builds successfully out of renderer
  // process. Launch WebNN service and bind `WebnnContext` mojo interface
  // to create `WebnnGraph` message pipe if needed.
  static void ValidateAndBuildAsync(MLContext* context,
                                    const MLNamedOperands& named_outputs,
                                    ScriptPromiseResolver* resolver);

  MLGraphMojo(ScriptState* script_state, MLContext* context);
  ~MLGraphMojo() override;

  void Trace(Visitor* visitor) const override;

 private:
  // Create `WebnnGraph` message pipe with `WebnnContext` mojo interface, then
  // build the computational graph with the hardware accelerated OS machine
  // learning API in the WebNN Service.
  void BuildAsyncImpl(const MLNamedOperands& outputs,
                      ScriptPromiseResolver* resolver) override;

  ScriptPromise ComputeImpl(ScriptState* script_state,
                            MLNamedArrayInputs inputs,
                            MLNamedArrayOutputs outputs,
                            ExceptionState& exception_state) override;
  void ComputeSyncImpl(MLNamedArrayInputs inputs,
                       MLNamedArrayOutputs outputs,
                       ExceptionState& exception_state) override;

  // The callback of creating `WebnnGraph` mojo interface from WebNN Service.
  // Return `CreatGraphResult::kNotSupported` with `mojo::NullRemote` on
  // non-supported platforms.
  void OnGraphCreated(const MLNamedOperands*,
                      ScriptPromiseResolver*,
                      ml::webnn::mojom::blink::CreateGraphResult result,
                      mojo::PendingRemote<ml::webnn::mojom::blink::WebnnGraph>);

  void OnGraphBuilt(ScriptPromiseResolver*,
                    ml::webnn::mojom::blink::BuildResult);
  void OnGraphComputed(ScriptPromiseResolver* resolver,
                       ComputeRequest* request,
                       ml::webnn::mojom::blink::ComputeResult result,
                       NamedResourcesPtr named_outputs);

  // The map of input name and input data offset.
  HashMap<String, size_t> inputs_byte_offset_;
  base::MappedReadOnlyRegion inputs_shm_region_;

  // The `WebnnGraph` mojo interface is used to build and execute graph in the
  // WebNN Service.
  HeapMojoRemote<ml::webnn::mojom::blink::WebnnGraph> remote_graph_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_MOJO_H_
