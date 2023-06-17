// Copyright 2023 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef COMPONENTS_ML_WEBNN_GRAPH_VALIDATION_UTILS_H_
#define COMPONENTS_ML_WEBNN_GRAPH_VALIDATION_UTILS_H_

#include <vector>

#include "base/containers/span.h"
#include "base/types/expected.h"
#include "third_party/abseil-cpp/absl/types/optional.h"

namespace webnn {

// The struct defined in this file need to be synced with,
// - "services/webnn/public/mojom/webnn_graph.mojom"
//
// Represents the `MLOperand` which describes not only input and constant
// operand, but also the output operand of operator.
struct Operand {
  // Represents the `MLOperandType` in the WebIDL definition.
  enum DataType {
    kFloat32,
    kFloat16,
    kInt32,
    kUint32,
    kInt8,
    kUint8,
  };

  Operand(DataType data_type, std::vector<uint32_t> dimensions);
  // Used for converting MLOperand to the component::Operand.
  Operand(DataType data_type, base::span<const uint32_t> dimensions);
  ~Operand();

  Operand(Operand&& other);
  Operand& operator=(Operand&& other);

  bool operator==(const Operand& other) const;
  bool operator!=(const Operand& other) const;

  Operand(const Operand&) = delete;
  Operand& operator=(const Operand&) = delete;

  // The data type of the operand.
  DataType data_type;
  // The dimensions of the operand.
  std::vector<uint32_t> dimensions;
};

// Represents the `MLInputOperandLayout` that specifies the layout format of
// the input tensor. N is the batch, C is input channels, H is height and W is
// the width of the tensor.
enum InputOperandLayout { kNchw, kNhwc };

// Represents the `MLAutoPad`. `Explicit` means that the values in the
// options.padding array should be used for input padding, with `SameUpper`
// and `SameLower` options, the padding values are automatically computed.
enum AutoPad { kExplicit, kSameUpper, kSameLower };

// Represents the `MLRoundingType` that is used to compute the output shape.
enum RoundingType { kFloor, kCeil };

// Contains the attributes of pool2d operator.
struct Pool2dAttributes {
  Pool2dAttributes();
  ~Pool2dAttributes();

  Pool2dAttributes(Pool2dAttributes&& other);
  Pool2dAttributes& operator=(Pool2dAttributes&& other);

  Pool2dAttributes(const Pool2dAttributes&) = delete;
  Pool2dAttributes& operator=(const Pool2dAttributes&) = delete;

  // The dimensions of the sliding window.
  absl::optional<std::vector<uint32_t>> window_dimensions;
  // The additional rows and columns added to the beginning and ending of each
  // spatial dimension of input.
  std::vector<uint32_t> padding;
  // The stride of the sliding window for each spatial dimension of input.
  std::vector<uint32_t> strides;
  // The dilation factor for each spatial dimension of input.
  std::vector<uint32_t> dilations;
  // The automatic input padding options.
  AutoPad auto_pad = AutoPad::kExplicit;
  // The layout format of the input.
  InputOperandLayout layout = InputOperandLayout::kNchw;
  // The rounding function used to compute the output shape.
  RoundingType rounding_type = RoundingType::kFloor;
  // The sizes of the two spacial dimensions of the output tensor.
  absl::optional<std::vector<uint32_t>> output_sizes;
};

// Contains the attributes of gemm operator.
struct GemmAttributes {
  GemmAttributes();
  ~GemmAttributes();

  GemmAttributes(GemmAttributes&& other);
  GemmAttributes& operator=(GemmAttributes&& other);

  GemmAttributes(const GemmAttributes&) = delete;
  GemmAttributes& operator=(const GemmAttributes&) = delete;

  // The id of third input tensor.
  absl::optional<Operand> c_operand;
  // A float scalar multiplier for the first input
  float alpha = 1.0;
  // A float scalar multiplier for the third input
  float beta = 1.0;
  // True is to transpose the first input prior to calculating the output.
  bool a_transpose = false;
  // True is to transpose the second input prior to calculating the output.
  bool b_transpose = false;
};

// Validate softmax operator defined in WebIDL here
// https://www.w3.org/TR/webnn/#api-mlgraphbuilder-softmax
base::expected<Operand, std::string> ValidateSoftmax(Operand input);

// Validate a mean, L2 norm, or max reduction operator defined in WebIDL here
// https://www.w3.org/TR/webnn/#api-mlgraphbuilder-pool2d
base::expected<Operand, std::string> ValidatePool2d(
    Operand input,
    Pool2dAttributes attributes);

// Validate gemm operator defined in WebIDL here
// https://www.w3.org/TR/webnn/#api-mlgraphbuilder-gemm
base::expected<Operand, std::string> ValidateGemm(Operand a,
                                                  Operand b,
                                                  GemmAttributes attributes);

base::expected<size_t, std::string> ValidateAndCalculateElementsNumber(
    base::span<const uint32_t> dimensions);

base::expected<size_t, std::string> ValidateAndCalculateByteLength(
    size_t type_bytes,
    base::span<const uint32_t> dimensions);

// Broadcast the input shapes and return the output shape.
// If bidirectional is true, its behavior follows the numpy-broadcasting-rule:
// https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules.
// Otherwise, it unidirectionally broadcasts the lhs to the rhs.
absl::optional<std::vector<uint32_t>> BroadcastShapes(
    base::span<const uint32_t> dims_lhs,
    base::span<const uint32_t> dims_rhs,
    bool bidirectional = true);

// TODO(crbug.com/1273291): Don't export FloatSize2D when moving the validation
// of Conv2d to the shared library.
struct FloatSize2D {
  double height;
  double width;
};

// TODO(crbug.com/1273291): Don't export PaddingSizes when moving the validation
// of ConvTransposed2d to the shared library.
struct PaddingSizes {
  uint32_t begin;
  uint32_t end;
};

// TODO(crbug.com/1273291): Don't export this heler function when moving the
// validation of Conv2d to the shared library.
base::expected<FloatSize2D, std::string> ValidateAndCalculateConv2dOutputSizes(
    const uint32_t input_height,
    const uint32_t input_width,
    const uint32_t filter_height,
    const uint32_t filter_width,
    base::span<const uint32_t> padding,
    base::span<const uint32_t> strides,
    base::span<const uint32_t> dilations,
    const AutoPad auto_pad);

struct Padding2D {
  uint32_t top;
  uint32_t bottom;
  uint32_t left;
  uint32_t right;
};

// Helper to get padding sizes for convolution 2d or pooling 2d operators.
Padding2D GetPadding2D(const AutoPad auto_pad,
                       base::span<const uint32_t> ml_padding,
                       uint32_t input_height,
                       uint32_t input_width,
                       uint32_t filter_height,
                       uint32_t filter_width,
                       uint32_t stride_height,
                       uint32_t stride_width,
                       uint32_t dilation_height,
                       uint32_t dilation_width);

}  // namespace webnn

#endif  // COMPONENTS_ML_WEBNN_GRAPH_VALIDATION_UTILS_H_
