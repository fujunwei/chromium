
TEST_P(MLGraphCrOSTest, ConvertWebNNGraphToTFLiteModel) {
  V8TestingScope scope;
  ScopedMLServiceBinder scoped_setup_binder(scope);
  auto* context_options = MLContextOptions::Create();
  context_options->setDevicePreference(V8MLDevicePreference::Enum::kGpu);
  auto* builder = CreateMLGraphBuilder(scope, context_options);
  Vector<uint32_t> shape({1, 4, 4, 3});
  wtf_size_t data_size = base::checked_cast<wtf_size_t>(std::accumulate(
      shape.begin(), shape.end(), size_t(1), std::multiplies<uint32_t>()));
  Vector<float> default_buffer(data_size, 1);
  {
    // Test the TF-Lite model converted from the following topology:
    //       [input0] [input1]
    //           \   /
    //            add
    //             |
    //          [output]
    auto* input0 = BuildInput(scope, builder, "input0", shape,
                              V8MLOperandType::Enum::kFloat32);
    auto* input1 = BuildInput(scope, builder, "input1", shape,
                              V8MLOperandType::Enum::kFloat32);
    auto* output = BuildElementWiseBinary(
        scope, builder, ElementWiseBinaryKind::kAdd, input0, input1);
    auto [graph, exception] = BuildGraph(scope, builder, {{"output", output}});
    EXPECT_NE(graph, nullptr);
    MLNamedArrayBufferViews inputs;
    inputs.emplace_back(
        "input0", CreateArrayBufferViewForOperand(input0, default_buffer));
    inputs.emplace_back(
        "input1", CreateArrayBufferViewForOperand(input1, default_buffer));
    MLNamedArrayBufferViews outputs;
    NotShared<DOMArrayBufferView> output_array_buffer =
        CreateArrayBufferViewForOperand(output);
    outputs.emplace_back("output", output_array_buffer);
    exception = ComputeGraph(scope, graph, inputs, outputs);
    EXPECT_EQ(exception, nullptr);
    auto results = GetArrayBufferViewValues<float>(output_array_buffer);
    EXPECT_EQ(results, Vector<float>(data_size, 2));
  }
  {
    // Test the TF-Lite model converted from the following topology:
    //       [input] [constant]
    //           \   /
    //            add
    //             |
    //          [output]
    auto* input = BuildInput(scope, builder, "input", shape,
                             V8MLOperandType::Enum::kFloat32);
    auto* constant = BuildConstant(
        scope, builder, shape, V8MLOperandType::Enum::kFloat32, default_buffer);
    auto* output = BuildElementWiseBinary(
        scope, builder, ElementWiseBinaryKind::kAdd, input, constant);
    auto [graph, exception] = BuildGraph(scope, builder, {{"output", output}});
    EXPECT_NE(graph, nullptr);
    MLNamedArrayBufferViews inputs;
    inputs.emplace_back("input",
                        CreateArrayBufferViewForOperand(input, default_buffer));
    MLNamedArrayBufferViews outputs;
    NotShared<DOMArrayBufferView> output_array_buffer =
        CreateArrayBufferViewForOperand(output);
    outputs.emplace_back("output", output_array_buffer);
    exception = ComputeGraph(scope, graph, inputs, outputs);
    EXPECT_EQ(exception, nullptr);
    auto results = GetArrayBufferViewValues<float>(output_array_buffer);
    EXPECT_EQ(results, Vector<float>(data_size, 2));
  }
  {
    // Test the TF-Lite model converted from the following topology:
    //       [input] [constant0]
    //           \   /
    //            add
    //             |
    //      [intermediate]  [constant1]
    //                  \   /
    //                   add
    //                    |
    //                 [output]
    auto* input = BuildInput(scope, builder, "input", shape,
                             V8MLOperandType::Enum::kFloat32);
    auto* constant0 = BuildConstant(
        scope, builder, shape, V8MLOperandType::Enum::kFloat32, default_buffer);
    auto* intermediate = BuildElementWiseBinary(
        scope, builder, ElementWiseBinaryKind::kAdd, input, constant0);
    auto* constant1 = BuildConstant(
        scope, builder, shape, V8MLOperandType::Enum::kFloat32, default_buffer);
    auto* output = BuildElementWiseBinary(
        scope, builder, ElementWiseBinaryKind::kAdd, intermediate, constant1);
    auto [graph, exception] = BuildGraph(scope, builder, {{"output", output}});
    EXPECT_NE(graph, nullptr);
    MLNamedArrayBufferViews inputs;
    inputs.emplace_back("input",
                        CreateArrayBufferViewForOperand(input, default_buffer));
    MLNamedArrayBufferViews outputs;
    NotShared<DOMArrayBufferView> output_array_buffer =
        CreateArrayBufferViewForOperand(output);
    outputs.emplace_back("output", output_array_buffer);
    exception = ComputeGraph(scope, graph, inputs, outputs);
    EXPECT_EQ(exception, nullptr);
    auto results = GetArrayBufferViewValues<float>(output_array_buffer);
    EXPECT_EQ(results, Vector<float>(data_size, 3));
  }
  {
    // Test the TF-Lite model converted from the following topology:
    //     [input0] [input1]
    //           \   /
    //            add
    //             |
    //      [intermediate]  [input2]
    //             |     \   /
    //            relu    add
    //             |       |
    //       [output0]   [output1]
    auto* input0 = BuildInput(scope, builder, "input0", shape,
                              V8MLOperandType::Enum::kFloat32);
    auto* input1 = BuildInput(scope, builder, "input1", shape,
                              V8MLOperandType::Enum::kFloat32);
    auto* intermediate = BuildElementWiseBinary(
        scope, builder, ElementWiseBinaryKind::kAdd, input0, input1);
    auto* output0 = builder->relu(intermediate, scope.GetExceptionState());
    auto* input2 = BuildInput(scope, builder, "input2", shape,
                              V8MLOperandType::Enum::kFloat32);
    auto* output1 = BuildElementWiseBinary(
        scope, builder, ElementWiseBinaryKind::kAdd, intermediate, input2);
    auto [graph, exception] = BuildGraph(
        scope, builder, {{"output0", output0}, {"output1", output1}});
    EXPECT_NE(graph, nullptr);
    MLNamedArrayBufferViews inputs;
    inputs.emplace_back(
        "input0", CreateArrayBufferViewForOperand(input0, default_buffer));
    inputs.emplace_back(
        "input1", CreateArrayBufferViewForOperand(input1, default_buffer));
    inputs.emplace_back(
        "input2", CreateArrayBufferViewForOperand(input2, default_buffer));
    MLNamedArrayBufferViews outputs;
    NotShared<DOMArrayBufferView> output0_array_buffer =
        CreateArrayBufferViewForOperand(output0);
    outputs.emplace_back("output0", output0_array_buffer);
    NotShared<DOMArrayBufferView> output1_array_buffer =
        CreateArrayBufferViewForOperand(output1);
    outputs.emplace_back("output1", output1_array_buffer);
    exception = ComputeGraph(scope, graph, inputs, outputs);
    EXPECT_EQ(exception, nullptr);
    auto results = GetArrayBufferViewValues<float>(output0_array_buffer);
    EXPECT_EQ(results, Vector<float>(data_size, 2));
    results = GetArrayBufferViewValues<float>(output1_array_buffer);
    EXPECT_EQ(results, Vector<float>(data_size, 3));
  }
}

template <typename T>
struct OperandInfo {
  V8MLOperandType::Enum type;
  Vector<uint32_t> dimensions;
  Vector<T> values;
};

template <typename T>
void SetArrayBufferViewValues(NotShared<DOMArrayBufferView> array_buffer_view,
                              const Vector<T>& values) {
  DCHECK_EQ(array_buffer_view->byteLength(), values.size() * sizeof(T));
  memcpy(array_buffer_view->BaseAddress(), values.data(),
         values.size() * sizeof(T));
}
