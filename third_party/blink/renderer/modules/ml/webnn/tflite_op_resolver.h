// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_TF_LITE_MODEL_INFO_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_TF_LITE_MODEL_INFO_H_

#include "third_party/tflite/src/tensorflow/lite/model.h"
#include "third_party/tflite/src/tensorflow/lite/mutable_op_resolver.h"

namespace blink {

// This class maintains all the currently supported TFLite
// operations for the Chromium build of TFLite and registers them for use.
class TFLiteOpResolver : public tflite::MutableOpResolver {
 public:
  TFLiteOpResolver();
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_WEBNN_TF_LITE_MODEL_INFO_H_
