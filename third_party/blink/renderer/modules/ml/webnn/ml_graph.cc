// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/ml_graph.h"

#include "third_party/blink/renderer/modules/ml/ml_context.h"

namespace blink {

MLGraph::MLGraph(MLContext* context) : ml_context_(context) {}

MLGraph::~MLGraph() = default;

void MLGraph::Trace(Visitor* visitor) const {
  visitor->Trace(ml_context_);
  ScriptWrappable::Trace(visitor);
}

void MLGraph::compute(const MLNamedArrayInputs& inputs,
                      const MLNamedArrayOutputs& outputs,
                      ExceptionState& exception_state) {
  ComputeImpl(inputs, outputs, exception_state);
}

}  // namespace blink
