// Copyright 2023 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_TEST_CROS_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_TEST_CROS_H_

#include "components/ml/mojom/ml_service.mojom-blink.h"
#include "components/ml/mojom/web_platform_model.mojom-blink.h"
#include "mojo/public/cpp/bindings/pending_remote.h"
#include "mojo/public/cpp/bindings/receiver.h"
#include "mojo/public/cpp/bindings/self_owned_receiver.h"
#include "third_party/blink/public/common/browser_interface_broker_proxy.h"
#include "third_party/blink/renderer/bindings/core/v8/v8_binding_for_testing.h"

namespace blink {

namespace blink_mojom = ml::model_loader::mojom::blink;

class FakeMLService;

class ScopedMLServiceBinder {
 public:
  explicit ScopedMLServiceBinder(const V8TestingScope& scope);

  ~ScopedMLServiceBinder();

 private:
  std::unique_ptr<FakeMLService> ml_service_;
  const BrowserInterfaceBrokerProxy& interface_broker_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_TEST_CROS_H_
