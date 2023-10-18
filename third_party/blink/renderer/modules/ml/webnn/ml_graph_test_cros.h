// Copyright 2023 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_TEST_CROS_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_TEST_CROS_H_

#include "third_party/blink/renderer/bindings/core/v8/v8_binding_for_testing.h"
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph_test_base.h"

namespace blink {

class FakeMLService;
class ScopedSetMLServiceBinder;
class FakeMLModelLoader;
class FakeWebNNModel;

class MLGraphTestCrOS : public MLGraphTestBase {
 public:
  MLGraphTestCrOS();
  ~MLGraphTestCrOS();

  ScopedSetMLServiceBinder SetUpMLService(V8TestingScope& scope);

 private:
  std::unique_ptr<FakeMLService> service_;
  std::unique_ptr<FakeMLModelLoader> loader_;
  std::unique_ptr<FakeWebNNModel> model_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_ML_GRAPH_TEST_CROS_H_
