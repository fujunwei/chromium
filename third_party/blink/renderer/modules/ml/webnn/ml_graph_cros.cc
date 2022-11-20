// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_cros.h"

#include <algorithm>

#include "base/numerics/checked_math.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise_resolver.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"
#include "third_party/blink/renderer/modules/ml/ml_context.h"

namespace blink {

// static
void MLGraphCrOS::ValidateAndBuildAsync(MLContext* context,
                                        const MLNamedOperands& named_outputs,
                                        ScriptPromiseResolver* resolver) {
  auto* graph = MakeGarbageCollected<MLGraphCrOS>(context);
  graph->BuildAsync(named_outputs, resolver);
}

MLGraphCrOS::MLGraphCrOS(MLContext* context) : MLGraph(context) {}

MLGraphCrOS::~MLGraphCrOS() = default;

void MLGraphCrOS::BuildAsyncImpl(const MLNamedOperands& named_outputs,
                                 ScriptPromiseResolver* resolver) {
  // TODO::Remove the to the topological if TF-Lite doesn't need it.
  auto* toposorted_operators = GetOperatorsInTopologicalOrder(named_outputs);
  DCHECK(toposorted_operators != nullptr);

  flatbuffers::Offset<tflite::Conv2DOptions> conv2d_options =
      tflite::CreateConv2DOptions(flat_builder_);
  flat_builder_.Finish(conv2d_options);
}

void MLGraphCrOS::ComputeAsyncImpl(const MLNamedArrayBufferViews& inputs,
                                   const MLNamedArrayBufferViews& outputs,
                                   ScriptPromiseResolver* resolver) {
  resolver->Reject(MakeGarbageCollected<DOMException>(
      DOMExceptionCode::kNotSupportedError, "Not implemented."));
}

}  // namespace blink
