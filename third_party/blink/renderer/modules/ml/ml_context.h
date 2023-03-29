// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_ML_CONTEXT_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_ML_CONTEXT_H_

#include "services/webnn/buildflags.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_device_preference.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_model_format.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_power_preference.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph.h"
#include "third_party/blink/renderer/modules/modules_export.h"
#include "third_party/blink/renderer/platform/bindings/script_wrappable.h"
#include "third_party/blink/renderer/platform/heap/member.h"
#include "third_party/blink/renderer/platform/heap/visitor.h"

#if BUILDFLAG(BUILD_WEBNN_WITH_SERVICE)
#include "services/webnn/public/mojom/webnn_graph.mojom-blink.h"
#include "services/webnn/public/mojom/webnn_service.mojom-blink.h"
#include "third_party/blink/renderer/platform/mojo/heap_mojo_remote.h"
#endif

namespace blink {

class ML;

class MODULES_EXPORT MLContext final : public ScriptWrappable {
  DEFINE_WRAPPERTYPEINFO();

 public:
  MLContext(const V8MLDevicePreference device_preference,
            const V8MLPowerPreference power_preference,
            const V8MLModelFormat model_format,
            const unsigned int num_threads,
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

  // IDL interface:
  ScriptPromise compute(ScriptState* script_state,
                        MLGraph* graph,
                        const MLNamedArrayBufferViews& inputs,
                        const MLNamedArrayBufferViews& outputs,
                        ExceptionState& exception_state);

  void computeSync(MLGraph* graph,
                   const MLNamedArrayBufferViews& inputs,
                   const MLNamedArrayBufferViews& outputs,
                   ExceptionState& exception_state);

  bool IsWebnnContextBound() const { return webnn_context_.is_bound(); }
  void CreateWebnnGraph(
      ScriptPromiseResolver* resolver,
      webnn::mojom::blink::WebnnContext::CreateGraphCallback callback);

 private:
  V8MLDevicePreference device_preference_;
  V8MLPowerPreference power_preference_;
  V8MLModelFormat model_format_;
  unsigned int num_threads_;

  Member<ML> ml_;

#if BUILDFLAG(BUILD_WEBNN_WITH_SERVICE)
  // The callback of creating context called from server side.
  void OnWebnnContextCreated(
      ScriptPromiseResolver* resolver,
      webnn::mojom::blink::CreateContextResult result,
      mojo::PendingRemote<webnn::mojom::blink::WebnnContext>);

  webnn::mojom::blink::WebnnContext::CreateGraphCallback create_graph_callback_;
  // Webnn support multiple types of neural network inference hardware
  // acceleration, the context of webnn in server side is used to map different
  // device and represent a state of graph execution processes.
  HeapMojoRemote<webnn::mojom::blink::WebnnContext> webnn_context_;
#endif
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_ML_CONTEXT_H_
