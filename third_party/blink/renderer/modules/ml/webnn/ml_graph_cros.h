// Copyright 2023 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_CROS_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_CROS_H_

#include "components/ml/mojom/web_platform_model.mojom-blink.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_operand.h"
#include "third_party/blink/renderer/modules/modules_export.h"
#include "third_party/blink/renderer/platform/heap/member.h"
#include "third_party/blink/renderer/platform/heap/visitor.h"
#include "third_party/blink/renderer/platform/mojo/heap_mojo_remote.h"
#include "third_party/flatbuffers/src/include/flatbuffers/flatbuffers.h"
#include "third_party/tflite/src/tensorflow/lite/schema/schema_generated.h"

namespace blink {

using ml::model_loader::mojom::blink::ComputeResult;
using ml::model_loader::mojom::blink::CreateModelLoaderResult;
using ml::model_loader::mojom::blink::LoadModelResult;
using ml::model_loader::mojom::blink::Model;
using ml::model_loader::mojom::blink::ModelInfoPtr;
using ml::model_loader::mojom::blink::ModelLoader;

class ScriptPromiseResolver;

class MODULES_EXPORT MLGraphCrOS final : public MLGraph {
 public:
  // Create and build an MLGraphCrOS object. Resolve the promise with
  // this concrete object if the underlying TF-Lite model converted from WebNN
  // graph builds successfully.
  static void ValidateAndBuildAsync(MLContext* context,
                                    const MLNamedOperands& named_outputs,
                                    ScriptPromiseResolver* resolver);

  // The constructor shouldn't be called directly, use ValidateAndBuildAsync()
  // method instead, and the declaration must be public to be called by
  // MakeGarbageCollected.
  MLGraphCrOS(ExecutionContext* execution_context, MLContext* context);

  ~MLGraphCrOS() override;

  void Trace(Visitor* visitor) const override;

 private:
  void OnRemoteLoaderCreated(ScriptState* script_state,
                             ScriptPromiseResolver* resolver,
                             const MLNamedOperands* named_outputs,
                             CreateModelLoaderResult result,
                             mojo::PendingRemote<ModelLoader> pending_remote);
  void OnRemoteModelLoad(ExecutionContext* execution_context,
                         ScriptPromiseResolver* resolver,
                         LoadModelResult result,
                         mojo::PendingRemote<Model> pending_remote,
                         ModelInfoPtr model_info);
  void OnComputeResult(
      ScriptPromiseResolver* resolver,
      const MLNamedArrayBufferViews* named_inputs,
      const MLNamedArrayBufferViews* named_outputs,
      ComputeResult result,
      const absl::optional<HashMap<String, Vector<uint8_t>>>& outputs);

  // Load a WebNN graph in `MLService` with `ModelLoader` message pipe, the
  // operations of WebNN need to be converted into a TF-Lite model in
  // FlatBuffers.
  void BuildAsyncImpl(const MLNamedOperands& named_outputs,
                      ScriptPromiseResolver* resolver) override;

  // TODO(crbug.com/1273291): Support sync build.
  MLGraph* BuildSyncImpl(const MLNamedOperands& named_outputs,
                         ExceptionState& exception_state) override;

  // Compute the converted WebNN graph with the `Model` mojo interface.
  void ComputeAsyncImpl(const MLNamedArrayBufferViews& inputs,
                        const MLNamedArrayBufferViews& outputs,
                        ScriptPromiseResolver* resolver) override;

  void ComputeSyncImpl(const MLNamedArrayBufferViews& inputs,
                       const MLNamedArrayBufferViews& outputs,
                       ExceptionState& exception_state) override;

  HeapMojoRemote<ml::model_loader::mojom::blink::ModelLoader> remote_loader_;
  HeapMojoRemote<ml::model_loader::mojom::blink::Model> remote_model_;
  HashMap<String, ml::model_loader::mojom::blink::TensorInfoPtr>
      input_tensor_name_to_info_;
  HashMap<String, ml::model_loader::mojom::blink::TensorInfoPtr>
      output_tensor_name_to_info_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_CROS_H_
