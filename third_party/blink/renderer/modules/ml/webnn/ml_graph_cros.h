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

namespace blink {

using ml::model_loader::mojom::blink::CreateModelLoaderResult;
using ml::model_loader::mojom::blink::LoadModelResult;
using ml::model_loader::mojom::blink::Model;
using ml::model_loader::mojom::blink::ModelInfoPtr;
using ml::model_loader::mojom::blink::ModelLoader;
using ml::model_loader::mojom::blink::TensorInfoPtr;

// Map the MLGraph's input or output name to the TensorInfoPtr.
using TensorInfoMap = HashMap<String, TensorInfoPtr>;

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

  const TensorInfoMap& GetInputTensorInfoMapForTesting() const;
  const TensorInfoMap& GetOutputTensorInfoMapForTesting() const;

 private:
  // The callback of creating `ModelLoader` mojo interface, it will return
  // `kNotSupported` if the input configuration is not supported.
  void OnRemoteLoaderCreated(ScriptState* script_state,
                             ScriptPromiseResolver* resolver,
                             const MLNamedOperands* named_outputs,
                             CreateModelLoaderResult result,
                             mojo::PendingRemote<ModelLoader> pending_remote);
  // The callback of loading tflite model, it will bind the `Model` pending
  // remote if it's successful.
  void OnRemoteModelLoad(ExecutionContext* execution_context,
                         ScriptPromiseResolver* resolver,
                         LoadModelResult result,
                         mojo::PendingRemote<Model> pending_remote,
                         ModelInfoPtr model_info);

  // Load a WebNN graph in `MLService` with `ModelLoader` message pipe, the
  // operations of WebNN need to be converted into a TF-Lite model in
  // FlatBuffers.
  void BuildAsyncImpl(const MLNamedOperands& named_outputs,
                      ScriptPromiseResolver* resolver) override;

  // TODO(crbug.com/1273291): Support sync build.
  MLGraph* BuildSyncImpl(const MLNamedOperands& named_outputs,
                         ExceptionState& exception_state) override;

  // TODO(crbug.com/1273291): Support async compute.
  void ComputeAsyncImpl(const MLNamedArrayBufferViews& inputs,
                        const MLNamedArrayBufferViews& outputs,
                        ScriptPromiseResolver* resolver) override;

  // TODO(crbug.com/1273291): Support sync compute.
  void ComputeSyncImpl(const MLNamedArrayBufferViews& inputs,
                       const MLNamedArrayBufferViews& outputs,
                       ExceptionState& exception_state) override;

  HeapMojoRemote<ml::model_loader::mojom::blink::ModelLoader> remote_loader_;
  HeapMojoRemote<ml::model_loader::mojom::blink::Model> remote_model_;
  TensorInfoMap input_tensor_name_to_info_;
  TensorInfoMap output_tensor_name_to_info_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_CROS_H_
