// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_ML_CONTEXT_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_ML_CONTEXT_H_

#include "components/ml/mojom/webnn_context.mojom-blink.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_device_preference.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_model_format.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_power_preference.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph.h"
#include "third_party/blink/renderer/modules/modules_export.h"
#include "third_party/blink/renderer/platform/bindings/script_wrappable.h"
#include "third_party/blink/renderer/platform/heap/member.h"
#include "third_party/blink/renderer/platform/heap/visitor.h"
#include "third_party/blink/renderer/platform/mojo/heap_mojo_remote.h"

namespace blink {

class ML;
class ScriptState;
class MLGraph;

class MODULES_EXPORT MLContext : public ScriptWrappable {
  DEFINE_WRAPPERTYPEINFO();

 public:
  MLContext(const V8MLDevicePreference device_preference,
            const V8MLPowerPreference power_preference,
            const V8MLModelFormat model_format,
            const unsigned int num_threads,
            ExecutionContext* execution_context,
            ML* ml);

  MLContext(const MLContext&) = delete;
  MLContext& operator=(const MLContext&) = delete;

  ~MLContext() override;

  V8MLDevicePreference GetDevicePreference() const;
  V8MLPowerPreference GetPowerPreference() const;
  V8MLModelFormat GetModelFormat() const;
  unsigned int GetNumThreads() const;

  ML* GetML();

  void Trace(Visitor* visitor) const override;

  // The CPU backend of WebNN is implementing in renderer process with Xnnpack,
  // other hardware acceleration such as DirectML on Windows will run in GPU
  // process, the runtime enable feature is used to disable the cross process
  // hardware acceleration by default.
  bool IsWebnnMojoContextEnabled() const;
  // Create WebNN mojo context in server side and await the callback to resolve
  // the ml context.
  void CreateWebnnMojoContext(ScriptPromiseResolver* resolver);
  // Create WebNN graph message pipe with WebNN mojo context interface, the
  // graph mojo interface is used to build and compute the computational graph.
  void CreateWebnnGraph(ScriptPromiseResolver*,
                        ml::webnn::mojom::blink::Context::CreateGraphCallback);

  // ml_context.idl
  ScriptPromise compute(ScriptState* script_state,
                        MLGraph* graph,
                        const MLNamedArrayInputs& inputs,
                        const MLNamedArrayOutputs& outputs,
                        ExceptionState& exception_state);

  void computeSync(MLGraph* graph,
                   const MLNamedArrayInputs& inputs,
                   const MLNamedArrayOutputs& outputs,
                   ExceptionState& exception_state);

 private:
  // The callback of creating context called from server side.
  void OnWebnnContextCreated(
      ScriptPromiseResolver* resolver,
      mojo::PendingRemote<ml::webnn::mojom::blink::Context>);

  V8MLDevicePreference device_preference_;
  V8MLPowerPreference power_preference_;
  V8MLModelFormat model_format_;
  unsigned int num_threads_;

  Member<ML> ml_;
  // Webnn support multiple types of neural network inference hardware
  // acceleration such as CPU, GPU, VPU, the context of webnn in server side is
  // used to map different device and represent a state of graph execution
  // processes.
  HeapMojoRemote<ml::webnn::mojom::blink::Context> webnn_context_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_ML_CONTEXT_H_
