// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/browser/ml/webnn/fusion_operators.h"

namespace content::webnn {

FusionOperators::~FusionOperators() = default;

FusionOperators::FusionOperators() = default;

void FusionOperators::AddClampOption(UINT64 operator_id,
                                     ClampOptionsPtr options) {
  clamp_options_[operator_id] = std::move(options);
}

const ClampOptions* FusionOperators::GetClampOption(UINT64 operator_id) {
  if (clamp_options_.find(operator_id) == clamp_options_.end()) {
    return nullptr;
  }
  return clamp_options_[operator_id].get();
}

}  // namespace content::webnn
