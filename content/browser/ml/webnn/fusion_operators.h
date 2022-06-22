// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CONTENT_BROWSER_ML_WEBNN_FUSION_OPERATORS_H_
#define CONTENT_BROWSER_ML_WEBNN_FUSION_OPERATORS_H_

#include "components/ml/webnn/mojom/graph.mojom.h"

#include <map>

namespace content {

namespace webnn {

namespace {

using ml::webnn::mojom::ClampOptions;
using ml::webnn::mojom::ClampOptionsPtr;

}  // namespace

class FusionOperators final {
 public:
  FusionOperators();
  ~FusionOperators();

  FusionOperators(const FusionOperators&) = delete;
  FusionOperators& operator=(const FusionOperators&) = delete;

  void AddClampOption(uint32_t operator_id, ClampOptionsPtr options);
  const ClampOptions* GetClampOption(uint32_t operator_id);

 private:
  std::map<uint32_t, ClampOptionsPtr> clamp_options_;
};

}  // namespace webnn
}  // namespace content

#endif  // CONTENT_BROWSER_ML_WEBNN_FUSION_OPERATORS_H_
