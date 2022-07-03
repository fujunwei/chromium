// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_ML_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_ML_H_

#include "base/memory/scoped_refptr.h"
#include "components/ml/mojom/ml_service.mojom-blink.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise.h"
#include "third_party/blink/renderer/core/execution_context/execution_context.h"
#include "third_party/blink/renderer/core/execution_context/navigator_base.h"
#include "third_party/blink/renderer/modules/ml/ml_context.h"
#include "third_party/blink/renderer/platform/bindings/script_wrappable.h"
#include "third_party/blink/renderer/platform/heap/member.h"
#include "third_party/blink/renderer/platform/heap/visitor.h"
#include "third_party/blink/renderer/platform/mojo/heap_mojo_remote.h"
#include "third_party/blink/renderer/platform/supplementable.h"

namespace blink {

class MLContext;
class MLContextOptions;
class ScriptPromise;
class ScriptState;
class ScriptPromise;
class MojoClient;

// This class represents the "Machine Learning" object "navigator.ml" and will
// be shared between the Model Loader API and WebNN API.
class ML final : public ScriptWrappable, public Supplement<NavigatorBase> {
  DEFINE_WRAPPERTYPEINFO();

 public:
  static const char kSupplementName[];
  // Getter for navigator.ml
  static ML* ml(NavigatorBase& navigator);

  explicit ML(NavigatorBase& navigator);

  ML(const ML&) = delete;
  ML& operator=(const ML&) = delete;

  void CreateModelLoader(
      ScriptState* script_state,
      ExceptionState& exception_state,
      ml::model_loader::mojom::blink::CreateModelLoaderOptionsPtr options,
      ml::model_loader::mojom::blink::MLService::CreateModelLoaderCallback
          callback);

  // Create the MojoClient if it's not ready.
  scoped_refptr<MojoClient> GetMojoClient();

  void Trace(blink::Visitor*) const override;

  // IDL interface:
  ScriptPromise createContext(ScriptState* state,
                              MLContextOptions* option,
                              ExceptionState& exception_state);

  MLContext* createContextSync(MLContextOptions* option,
                               ExceptionState& exception_state);

 private:
  // Binds the Mojo connection to browser process if needed.
  // Returns false when the execution context is not valid (e.g., the frame is
  // detached) and an exception will be thrown.
  // Otherwise returns true.
  bool BootstrapMojoConnectionIfNeeded(ScriptState* script_state,
                                       ExceptionState& exception_state);

  Member<ExecutionContext> execution_context_;

  HeapMojoRemote<ml::model_loader::mojom::blink::MLService> remote_service_;

  // WebNN Wire Client to create native Context and manage Object ID that talk
  // with server side.
  scoped_refptr<MojoClient> webnn_mojo_client_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_ML_H_
