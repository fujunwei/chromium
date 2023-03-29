// Copyright 2023 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_cros.h"

#include <numeric>

#include "base/numerics/checked_math.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise_resolver.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_compute_result.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"
#include "third_party/blink/renderer/modules/ml/ml.h"
#include "third_party/blink/renderer/modules/ml/ml_context.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_builder.h"
#include "third_party/blink/renderer/platform/bindings/exception_state.h"
#include "third_party/flatbuffers/src/include/flatbuffers/flatbuffers.h"

namespace blink {

namespace {

using ml::model_loader::mojom::blink::CreateModelLoaderOptions;
using ml::model_loader::mojom::blink::DevicePreference;
using ml::model_loader::mojom::blink::ModelFormat;
using ml::model_loader::mojom::blink::ModelLoader;

// The callback of loading model is used to bind the pending remote of `Model`
// interface if the model is loaded successfully.
using WebnnLoadModelCallback =
    base::OnceCallback<void(WebnnLoadModelResult result,
                            mojo::PendingRemote<Model> pending_remote,
                            ModelInfoPtr model_info)>;

void LoadCrOSModel(MLContext* ml_context,
                   ScriptState* script_state,
                   flatbuffers::DetachedBuffer flat_buffer,
                   WebnnLoadModelCallback callback) {
  auto& remote_loader = ml_context->GetModelLoaderRemote();
  if (!remote_loader.is_bound()) {
    // Needs to bootstrap the mojo connection first.
    auto options_mojo = CreateModelLoaderOptions::New();
    options_mojo->num_threads = ml_context->GetNumThreads();
    // Hardcode the preference of creating ModelLoader mojo interface because
    // `MLService` only support TF-Lite model and CPU device at current stage.
    // TODO(crbug.com/1273291): Support power preference, other model format and
    // device preference.
    options_mojo->model_format = ModelFormat::kTfLite;
    options_mojo->device_preference = DevicePreference::kCpu;

    // Use `ModelLoader` mojo interface to load WebNN computational graphs that
    // is converted in FlatBuffer.
    ml_context->GetML()->CreateModelLoader(
        script_state, std::move(options_mojo),
        WTF::BindOnce(
            [](ScriptState* script_state, MLContext* ml_context,
               flatbuffers::DetachedBuffer flat_buffer,
               WebnnLoadModelCallback callback, CreateModelLoaderResult result,
               mojo::PendingRemote<ModelLoader> pending_remote) {
              if (result != CreateModelLoaderResult::kOk) {
                std::move(callback).Run(WebnnLoadModelResult::kError,
                                        mojo::NullRemote(), nullptr);
                return;
              }
              auto* execution_context = ExecutionContext::From(script_state);
              auto& remote_loader = ml_context->GetModelLoaderRemote();
              remote_loader.Bind(
                  std::move(pending_remote),
                  execution_context->GetTaskRunner(TaskType::kInternalDefault));

              remote_loader->Load(
                  base::make_span(
                      static_cast<const uint8_t*>(flat_buffer.data()),
                      flat_buffer.size()),
                  WTF::BindOnce(
                      [](WebnnLoadModelCallback callback,
                         LoadModelResult result,
                         mojo::PendingRemote<Model> pending_remote,
                         ModelInfoPtr model_info) {
                        if (result != LoadModelResult::kOk) {
                          std::move(callback).Run(WebnnLoadModelResult::kError,
                                                  mojo::NullRemote(), nullptr);
                          return;
                        }
                        std::move(callback).Run(WebnnLoadModelResult::kOk,
                                                std::move(pending_remote),
                                                std::move(model_info));
                      },
                      std::move(callback)));
            },
            WrapPersistent(script_state), WrapPersistent(ml_context),
            std::move(flat_buffer), std::move(callback)));
  } else {
    // Directly use `remote_loader`.
    remote_loader->Load(
        base::make_span(static_cast<const uint8_t*>(flat_buffer.data()),
                        flat_buffer.size()),
        WTF::BindOnce(
            [](WebnnLoadModelCallback callback, LoadModelResult result,
               mojo::PendingRemote<Model> pending_remote,
               ModelInfoPtr model_info) {
              if (result != LoadModelResult::kOk) {
                std::move(callback).Run(WebnnLoadModelResult::kError,
                                        mojo::NullRemote(), nullptr);
                return;
              }
              std::move(callback).Run(WebnnLoadModelResult::kOk,
                                      std::move(pending_remote),
                                      std::move(model_info));
            },
            std::move(callback)));
  }
}

}  // namespace

// static
void MLGraphCrOS::ValidateAndBuildAsync(MLContext* ml_context,
                                        const MLNamedOperands& named_outputs,
                                        ScriptPromiseResolver* resolver) {
  auto* script_state = resolver->GetScriptState();
  auto* execution_context = ExecutionContext::From(script_state);
  auto* graph =
      MakeGarbageCollected<MLGraphCrOS>(execution_context, ml_context);
  graph->BuildAsync(named_outputs, resolver);
}

MLGraphCrOS::MLGraphCrOS(ExecutionContext* execution_context,
                         MLContext* ml_context)
    : MLGraph(ml_context), remote_model_(execution_context) {}

MLGraphCrOS::~MLGraphCrOS() = default;

void MLGraphCrOS::Trace(Visitor* visitor) const {
  visitor->Trace(remote_model_);
  MLGraph::Trace(visitor);
}

void MLGraphCrOS::BuildAsyncImpl(const MLNamedOperands& outputs,
                                 ScriptPromiseResolver* resolver) {
  auto* script_state = resolver->GetScriptState();
  auto* execution_context = ExecutionContext::From(script_state);
  // TODO(crbug.com/1273291): Convert WebNN graph to tflite model instead of
  // the empty buffer.
  LoadCrOSModel(
      ml_context_, script_state, flatbuffers::DetachedBuffer(),
      WTF::BindOnce(&MLGraphCrOS::OnRemoteModelLoad, WrapPersistent(this),
                    WrapPersistent(execution_context),
                    WrapPersistent(resolver)));
}

void MLGraphCrOS::OnRemoteModelLoad(ExecutionContext* execution_context,
                                    ScriptPromiseResolver* resolver,
                                    WebnnLoadModelResult result,
                                    mojo::PendingRemote<Model> pending_remote,
                                    ModelInfoPtr tensor_info) {
  switch (result) {
    case WebnnLoadModelResult::kError:
      resolver->Reject(MakeGarbageCollected<DOMException>(
          DOMExceptionCode::kUnknownError, "Unknown error."));
      return;
    case WebnnLoadModelResult::kOk:
      remote_model_.Bind(
          std::move(pending_remote),
          execution_context->GetTaskRunner(TaskType::kInternalDefault));

      // Stores input tensor information of loaded model to verify the input
      // data by user including name and byte length.
      input_tensor_name_to_info_ = std::move(tensor_info->input_tensor_info);
      // Stores output tensor information of loaded model to verify the output
      // data returned from `MLService` after computing.
      output_tensor_name_to_info_ = std::move(tensor_info->output_tensor_info);

      resolver->Resolve(this);
      return;
  }
}

const TensorInfoMap& MLGraphCrOS::GetInputTensorInfoMapForTesting() const {
  return input_tensor_name_to_info_;
}

const TensorInfoMap& MLGraphCrOS::GetOutputTensorInfoMapForTesting() const {
  return output_tensor_name_to_info_;
}

MLGraph* MLGraphCrOS::BuildSyncImpl(const MLNamedOperands& named_outputs,
                                    ExceptionState& exception_state) {
  // TODO(crbug.com/1273291): Support sync build.
  exception_state.ThrowDOMException(DOMExceptionCode::kNotSupportedError,
                                    "Not implemented.");
  return nullptr;
}

void MLGraphCrOS::ComputeAsyncImpl(const MLNamedArrayBufferViews& inputs,
                                   const MLNamedArrayBufferViews& outputs,
                                   ScriptPromiseResolver* resolver) {
  // TODO(crbug.com/1273291): Support async compute.
  resolver->Reject(MakeGarbageCollected<DOMException>(
      DOMExceptionCode::kNotSupportedError, "Not implemented."));
}

void MLGraphCrOS::ComputeSyncImpl(const MLNamedArrayBufferViews& inputs,
                                  const MLNamedArrayBufferViews& outputs,
                                  ExceptionState& exception_state) {
  // TODO(crbug.com/1273291): Support sync compute.
  exception_state.ThrowDOMException(DOMExceptionCode::kNotSupportedError,
                                    "Not implemented.");
}

}  // namespace blink
