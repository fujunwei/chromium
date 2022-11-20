// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_CROS_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_CROS_H_

#include "third_party/blink/renderer/modules/ml/webnn/ml_graph.h"
#include "third_party/blink/renderer/modules/modules_export.h"
#include "third_party/flatbuffers/src/include/flatbuffers/flatbuffers.h"
#include "third_party/tflite/src/tensorflow/lite/schema/schema_generated.h"

namespace blink {

class ScriptPromiseResolver;

class MODULES_EXPORT MLGraphCrOS final : public MLGraph {
 public:
  // Create and build an MLGraphCrOS object. Resolve the promise with
  // this concrete object if the underlying TF-Lite model builds successfully.
  static void ValidateAndBuildAsync(MLContext* context,
                                    const MLNamedOperands& named_outputs,
                                    ScriptPromiseResolver* resolver);

  // The constructor shouldn't be called directly. The callers should use
  // ValidateAndBuildAsync() method instead.
  explicit MLGraphCrOS(MLContext* context);

  ~MLGraphCrOS() override;

 private:
  // Load a WebNN graph in `MLService` with `ModelLoader` message pipe, the
  // operations of WebNN need to be converted into a TF-Lite model in
  // FlatBuffers.
  void BuildAsyncImpl(const MLNamedOperands& named_outputs,
                      ScriptPromiseResolver* resolver) override;

  // Compute the converted WebNN graph with the `Model` mojo interface.
  void ComputeAsyncImpl(const MLNamedArrayBufferViews& inputs,
                        const MLNamedArrayBufferViews& outputs,
                        ScriptPromiseResolver* resolver) override;

  // Must outlive this instance.
  flatbuffers::FlatBufferBuilder flat_builder_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_CROS_H_
