// Copyright 2023 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "components/ml/mojom/ml_service.mojom-blink.h"
#include "components/ml/mojom/web_platform_model.mojom-blink.h"
#include "mojo/public/cpp/bindings/pending_remote.h"
#include "mojo/public/cpp/bindings/receiver.h"
#include "third_party/blink/public/common/browser_interface_broker_proxy.h"
#include "third_party/blink/renderer/bindings/core/v8/v8_binding_for_testing.h"
#include "third_party/blink/renderer/bindings/modules/v8/v8_ml_tensor.h"

namespace blink {

namespace blink_mojom = ml::model_loader::mojom::blink;

// A fake MLService that intercepts Blink's browser interface request to the
// ml.model_loader.MLService interface.
class FakeMLService : public blink_mojom::MLService {
 public:
  FakeMLService() = default;
  FakeMLService(const FakeMLService&) = delete;
  FakeMLService(FakeMLService&&) = delete;
  ~FakeMLService() override = default;

  using CreateModelLoaderFn = base::OnceCallback<void(
      blink_mojom::CreateModelLoaderOptionsPtr,
      blink_mojom::MLService::CreateModelLoaderCallback)>;

  void SetCreateModelLoader(CreateModelLoaderFn fn);

  void BindFakeService(mojo::ScopedMessagePipeHandle pipe);

 private:
 private:
  // Override methods from ml::blink_mojom::MLService.
  void CreateModelLoader(blink_mojom::CreateModelLoaderOptionsPtr opts,
                         CreateModelLoaderCallback callback) override;

  CreateModelLoaderFn create_model_loader_;
  mojo::Receiver<blink_mojom::MLService> receiver_{this};
};

class ScopedSetMLServiceBinder {
 public:
  ScopedSetMLServiceBinder(FakeMLService* ml_service,
                           const V8TestingScope& scope);
  ~ScopedSetMLServiceBinder();

 private:
  const BrowserInterfaceBrokerProxy& interface_broker_;
};

// A fake MLModelLoader Mojo interface implementation that backs a Blink
// MLModelLoader object.
class FakeMLModelLoader : public blink_mojom::ModelLoader {
 public:
  FakeMLModelLoader() = default;
  FakeMLModelLoader(const FakeMLModelLoader&) = delete;
  FakeMLModelLoader(FakeMLModelLoader&&) = delete;
  ~FakeMLModelLoader() override = default;

  using LoadFn =
      base::OnceCallback<void(mojo_base::BigBuffer,
                              blink_mojom::ModelLoader::LoadCallback)>;
  void SetLoad(LoadFn fn);

  FakeMLService::CreateModelLoaderFn CreateFromThis();

  FakeMLService::CreateModelLoaderFn CreateForUnsupportedContext();

  void OnCreateModelLoader(
      blink_mojom::CreateModelLoaderOptionsPtr,
      blink_mojom::MLService::CreateModelLoaderCallback callback);

 private:
  // Override methods from blink_mojom::ModelLoader.
  void Load(mojo_base::BigBuffer buffer,
            blink_mojom::ModelLoader::LoadCallback callback) override;

  LoadFn load_;
  mojo::Receiver<blink_mojom::ModelLoader> receiver_{this};
};

// Helper struct to create faked MLTensors.
struct TensorInfo {
  uint32_t byte_size;
  blink_mojom::DataType data_type;
  WTF::Vector<uint32_t> dimensions;

  blink_mojom::TensorInfoPtr ToMojom() const;

  MLTensor* ToMLTensor() const;
};

// A fake MLModel Mojo interface implementation that backs a Blink MLModel
// object.
class FakeMLModel : public blink_mojom::Model {
 public:
  FakeMLModel() = default;
  FakeMLModel(const FakeMLModel&) = delete;
  FakeMLModel(FakeMLModel&&) = delete;
  ~FakeMLModel() override = default;

  using ComputeFn = base::OnceCallback<void(
      const WTF::HashMap<WTF::String, WTF::Vector<uint8_t>>&,
      blink_mojom::Model::ComputeCallback callback)>;

  FakeMLModelLoader::LoadFn CreateFromInfo(
      const std::map<std::string, TensorInfo>& inputs,
      const std::map<std::string, TensorInfo>& outputs);

  void OnCreateModel(mojo_base::BigBuffer,
                     blink_mojom::ModelLoader::LoadCallback callback);

  void SetModelInfo(blink_mojom::ModelInfoPtr info);

  void SetComputeResult(
      const std::map<std::string, WTF::Vector<uint8_t>>& output);

  void SetComputeFailure(const blink_mojom::ComputeResult result);

 private:
  // Override methods from blink_mojom::Model.
  void Compute(const WTF::HashMap<WTF::String, WTF::Vector<uint8_t>>& input,
               blink_mojom::Model::ComputeCallback callback) override;

  blink_mojom::ModelInfoPtr info_;
  ComputeFn compute_;

  mojo::Receiver<blink_mojom::Model> receiver_{this};
};

}  // namespace blink
