// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_MOJO_GRAPH_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_MOJO_GRAPH_H_

#include "base/memory/read_only_shared_memory_region.h"
#include "base/memory/shared_memory_mapping.h"
#include "components/ml/mojom/webnn_graph.mojom-blink.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph.h"
#include "third_party/blink/renderer/platform/heap/member.h"
#include "third_party/blink/renderer/platform/heap/visitor.h"
#include "third_party/blink/renderer/platform/mojo/heap_mojo_remote.h"

namespace blink {

class ScriptPromiseResolver;

using ml::webnn::mojom::blink::FusionOperatorPtr;
using ml::webnn::mojom::blink::NamedOutputsPtr;
using ml::webnn::mojom::blink::OperandDescriptorPtr;

class MojoGraph : public MLGraph {
 public:
  MojoGraph(ScriptState* script_state, MLContext* context);
  ~MojoGraph() override;

  void Trace(Visitor* visitor) const override;

  ScriptPromise BuildImpl(ScriptState* script_state,
                          MLNamedOperands named_outputs,
                          ExceptionState& exception_state) override;
  void BuildSyncImpl(MLNamedOperands named_outputs,
                     ExceptionState& exception_state) override;

  ScriptPromise ComputeImpl(ScriptState* script_state,
                            MLNamedArrayInputs inputs,
                            MLNamedArrayOutputs outputs,
                            ExceptionState& exception_state) override;
  void ComputeSyncImpl(MLNamedArrayInputs inputs,
                       MLNamedArrayOutputs outputs,
                       ExceptionState& exception_state) override;

 private:
  void OnGraphCreated(ScriptState*,
                      ScriptPromiseResolver*,
                      BuildRequest* request,
                      mojo::PendingRemote<ml::webnn::mojom::blink::Graph>);
  void OnGraphBuilt(ScriptPromiseResolver*,
                    ml::webnn::mojom::blink::BuildResult);
  void OnGraphComputed(ScriptPromiseResolver* resolver,
                       ComputeRequest* request,
                       ml::webnn::mojom::blink::ComputeResult result,
                       NamedOutputsPtr named_outputs);

  // The map of input name and input data offset.
  HashMap<String, size_t> inputs_byte_offset_;
  base::MappedReadOnlyRegion inputs_shm_region_;

  HeapMojoRemote<ml::webnn::mojom::blink::Graph> remote_graph_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_MOJO_GRAPH_H_
