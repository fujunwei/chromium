
#include "third_party/blink/renderer/modules/ml/webnn/ml_graph.h"


  // ml_context.idl
  ScriptPromise compute(ScriptState* script_state,
                        MLGraph* graph,
                        const MLNamedArrayBufferViews& inputs,
                        const MLNamedArrayBufferViews& outputs,
                        ExceptionState& exception_state);


#include "third_party/blink/renderer/bindings/core/v8/script_promise_resolver.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"
#include "third_party/blink/renderer/platform/bindings/exception_state.h"

ScriptPromise MLContext::compute(ScriptState* script_state,
                                 MLGraph* graph,
                                 const MLNamedArrayBufferViews& inputs,
                                 const MLNamedArrayBufferViews& outputs,
                                 ExceptionState& exception_state) {
  if (!script_state->ContextIsValid()) {
    exception_state.ThrowDOMException(DOMExceptionCode::kInvalidStateError,
                                      "Invalid script state");
    return ScriptPromise();
  }
  auto* resolver = MakeGarbageCollected<ScriptPromiseResolver>(script_state);
  auto promise = resolver->Promise();
  graph->ComputeAsync(inputs, outputs, resolver);
  return promise;
}

bool Input1Static() {
  return true;
}

bool Input2Static() {
  return true;
}

std::vector<int32_t> Input1Shape() {
  return std::vector<int32_t>({1, 2, 2, 3});
}

std::vector<int32_t> Input2Shape() {
  return std::vector<int32_t>({1, 2, 2, 3});
}

std::vector<int32_t> OutputShape() {
  return std::vector<int32_t>({1, 2, 2, 3});
}

int32_t ComputeSize(const std::vector<int32_t>& shape) {
  return std::accumulate(shape.cbegin(), shape.cend(), 1,
                         std::multiplies<int32_t>());
}

std::vector<const uint8_t> CreateTfLiteModel(
    tflite::BuiltinOperator binary_op) {
  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<tflite::OperatorCode>> operator_codes{
      {tflite::CreateOperatorCode(builder, binary_op)}};

  std::vector<flatbuffers::Offset<tflite::Buffer>> buffers{{
      tflite::CreateBuffer(builder, builder.CreateVector({})),
  }};

  uint32_t input1_buffer = 0;
  if (Input1Static()) {
    std::vector<float> input1_data(ComputeSize(Input1Shape()), 2);
    // std::generate(input1_data.begin(), input1_data.end(), input1_rng);
    input1_buffer = static_cast<uint32_t>(buffers.size());

    buffers.push_back(tflite::CreateBuffer(
        builder, builder.CreateVector(
                     reinterpret_cast<const uint8_t*>(input1_data.data()),
                     sizeof(float) * input1_data.size())));
  }

  uint32_t input2_buffer = 0;
  if (Input2Static()) {
    std::vector<float> input2_data(ComputeSize(Input2Shape()), 3);
    // std::generate(input2_data.begin(), input2_data.end(), input2_rng);

    input2_buffer = static_cast<uint32_t>(buffers.size());

    buffers.push_back(tflite::CreateBuffer(
        builder, builder.CreateVector(
                     reinterpret_cast<const uint8_t*>(input2_data.data()),
                     sizeof(float) * input2_data.size())));
  }

  const std::vector<int32_t> output_shape = OutputShape();
  std::vector<flatbuffers::Offset<tflite::Tensor>> tensors;
  std::vector<flatbuffers::Offset<tflite::Operator>> operators;
  tensors.emplace_back(tflite::CreateTensor(
      builder, builder.CreateVector<int32_t>(Input1Shape()),
      tflite::TensorType_FLOAT32, input1_buffer));
  tensors.emplace_back(tflite::CreateTensor(
      builder, builder.CreateVector<int32_t>(Input2Shape()),
      tflite::TensorType_FLOAT32, input2_buffer));
  tensors.emplace_back(
      tflite::CreateTensor(builder, builder.CreateVector<int32_t>(output_shape),
                           tflite::TensorType_FLOAT32));

  // TF-Lite support activation in elementwise binary operations, but WebNN Spec
  // doesn't define the feature, so the options of elementwise binary doesn't
  // need to be configured.
  tflite::BuiltinOptions builtin_options_type = tflite::BuiltinOptions_NONE;
  flatbuffers::Offset<void> builtin_options = 0;

  const std::array<int32_t, 2> op_inputs{
      {static_cast<int>(tensors.size()) - 3,
       static_cast<int>(tensors.size()) - 2}};
  const std::array<int32_t, 1> op_outputs{
      {static_cast<int>(tensors.size()) - 1}};
  operators.emplace_back(tflite::CreateOperator(
      builder, /*opcode_index=*/0, builder.CreateVector<int32_t>(op_inputs),
      builder.CreateVector<int32_t>(op_outputs), builtin_options_type,
      builtin_options));

  std::vector<int32_t> subgraph_inputs;
  if (!Input1Static()) {
    subgraph_inputs.push_back(static_cast<int32_t>(tensors.size() - 3));
  }
  if (!Input2Static()) {
    subgraph_inputs.push_back(static_cast<int32_t>(tensors.size() - 2));
  }
  const std::array<int32_t, 1> subgraph_outputs{
      {static_cast<int>(tensors.size()) - 1}};
  flatbuffers::Offset<tflite::SubGraph> subgraph =
      tflite::CreateSubGraph(builder, builder.CreateVector(tensors),
                             builder.CreateVector<int32_t>(subgraph_inputs),
                             builder.CreateVector<int32_t>(subgraph_outputs),
                             builder.CreateVector(operators));

  flatbuffers::Offset<flatbuffers::String> description =
      builder.CreateString("Binary model");

  flatbuffers::Offset<tflite::Model> model_buffer = tflite::CreateModel(
      builder, TFLITE_SCHEMA_VERSION, builder.CreateVector(operator_codes),
      builder.CreateVector(&subgraph, 1), description,
      builder.CreateVector(buffers));

  tflite::FinishModelBuffer(builder, model_buffer);

  return std::vector<const uint8_t>(
      builder.GetBufferPointer(),
      builder.GetBufferPointer() + builder.GetSize());
}
  // // std::vector<const uint8_t> buffer =
  // //     CreateTfLiteModel(tflite::BuiltinOperator_ADD);


Remove SecureContext from navigator.ml IDL definition to use IP address to load webnn-polyfill
[
   Exposed=Window,
-  SecureContext,
   ImplementedAs=NavigatorML,
   RuntimeEnabled=MachineLearningCommon
 ] partial interface Navigator {


=================================
#include <fstream>
// Test the tf-lite model converted from WebNN Graph
  TfLiteConverter* tf_lite_model = BuildTFLiteModel(*named_outputs);
  auto& builder = tf_lite_model->GetFlatBufferBuilder();
  // =====================
  std::ofstream file("/home/junwei/workspace/webnn/ops_test/add.tflite",
                     std::ios::binary);
  if (!file.is_open()) {
    return;
  }
  file << std::string(builder.GetBufferPointer(),
                      builder.GetBufferPointer() + builder.GetSize());
  file.close();
  LOG(ERROR) << "==========Write model to the file";


