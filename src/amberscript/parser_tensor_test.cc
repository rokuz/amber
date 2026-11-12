// Copyright 2026 The Amber Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstring>

#include "gtest/gtest.h"
#include "src/amberscript/parser.h"

namespace amber {
namespace amberscript {

using AmberScriptParserTest = testing::Test;

TEST_F(AmberScriptParserTest, TensorData) {
  std::string in = R"(
TENSOR t DATA_TYPE float DIMS 2 3 DATA
1 2 3
4 5 6
END
)";

  Parser parser;
  Result r = parser.Parse(in);
  ASSERT_TRUE(r.IsSuccess()) << r.Error();

  auto script = parser.GetScript();
  const auto& buffers = script->GetBuffers();
  ASSERT_EQ(1U, buffers.size());
  const auto* buffer = buffers[0].get();
  EXPECT_EQ("t", buffer->GetName());
  EXPECT_TRUE(buffer->IsTensor());
  EXPECT_TRUE(buffer->GetFormat()->IsFloat32());
  EXPECT_EQ((std::vector<uint64_t>{2, 3}), buffer->GetTensorDims());
  EXPECT_TRUE(buffer->GetTensorStrides().empty());
  EXPECT_EQ(TensorEncoding::kNone, buffer->GetTensorEncoding());
  EXPECT_EQ(6U, buffer->ElementCount());
  ASSERT_EQ(24U, buffer->ValuePtr()->size());
  float values[6];
  std::memcpy(values, buffer->ValuePtr()->data(), sizeof(values));
  for (int i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(static_cast<float>(i + 1), values[i]);
  }
}

TEST_F(AmberScriptParserTest, TensorFillSeriesAndDefault) {
  std::string in = R"(
TENSOR filled DATA_TYPE int16 DIMS 2 2 2 FILL 7
TENSOR series DATA_TYPE uint32 DIMS 4 SERIES_FROM 3 INC_BY 2
TENSOR zero DATA_TYPE int64 DIMS 3
)";

  Parser parser;
  Result r = parser.Parse(in);
  ASSERT_TRUE(r.IsSuccess()) << r.Error();

  auto script = parser.GetScript();
  const auto* filled = script->GetBuffer("filled");
  ASSERT_TRUE(filled != nullptr);
  EXPECT_EQ(8U, filled->ElementCount());
  ASSERT_EQ(16U, filled->ValuePtr()->size());
  EXPECT_EQ(7, filled->GetValues<int16_t>()[7]);

  const auto* series = script->GetBuffer("series");
  ASSERT_TRUE(series != nullptr);
  ASSERT_EQ(16U, series->ValuePtr()->size());
  EXPECT_EQ(3U, series->GetValues<uint32_t>()[0]);
  EXPECT_EQ(9U, series->GetValues<uint32_t>()[3]);

  const auto* zero = script->GetBuffer("zero");
  ASSERT_TRUE(zero != nullptr);
  EXPECT_EQ(3U, zero->ElementCount());
  ASSERT_EQ(24U, zero->ValuePtr()->size());
  EXPECT_EQ(0, zero->GetValues<int64_t>()[2]);
}

TEST_F(AmberScriptParserTest, TensorEncodingAndStrides) {
  std::string in = R"(
TENSOR flags DATA_TYPE uint8 ENCODING BOOL DIMS 4 DATA 1 0 1 1 END
TENSOR bf DATA_TYPE uint16 ENCODING BFLOAT16 DIMS 1 FILL 16256
TENSOR e4 DATA_TYPE uint8 ENCODING FLOAT8_E4M3 DIMS 1 FILL 56
TENSOR e5 DATA_TYPE uint8 ENCODING FLOAT8_E5M2 DIMS 1 FILL 60
TENSOR padded DATA_TYPE float DIMS 2 3 STRIDES 32 4 FILL 1
)";

  Parser parser;
  Result r = parser.Parse(in);
  ASSERT_TRUE(r.IsSuccess()) << r.Error();

  auto script = parser.GetScript();
  EXPECT_EQ(TensorEncoding::kBool,
            script->GetBuffer("flags")->GetTensorEncoding());
  EXPECT_EQ(TensorEncoding::kBFloat16,
            script->GetBuffer("bf")->GetTensorEncoding());
  EXPECT_EQ(TensorEncoding::kFloat8E4M3,
            script->GetBuffer("e4")->GetTensorEncoding());
  EXPECT_EQ(TensorEncoding::kFloat8E5M2,
            script->GetBuffer("e5")->GetTensorEncoding());
  const auto* padded = script->GetBuffer("padded");
  EXPECT_EQ((std::vector<uint64_t>{32, 4}), padded->GetTensorStrides());
  EXPECT_EQ(6U, padded->ElementCount());
  EXPECT_EQ(24U, padded->ValuePtr()->size());
}

TEST_F(AmberScriptParserTest, TensorErrors) {
  struct {
    const char* in;
    const char* err;
  } cases[] = {
      {"TENSOR", "1: invalid TENSOR name provided"},
      {"TENSOR t DIMS 2", "1: TENSOR missing DATA_TYPE"},
      {"TENSOR t DATA_TYPE vec2<float> DIMS 2",
       "1: TENSOR data type must be a scalar number type: vec2<float>"},
      {"TENSOR t DATA_TYPE float FILL 0", "1: TENSOR missing DIMS"},
      {"TENSOR t DATA_TYPE float DIMS FILL 0",
       "1: TENSOR DIMS needs at least one dimension"},
      {"TENSOR t DATA_TYPE float DIMS 2 0",
       "1: TENSOR dimensions must be positive"},
      {"TENSOR t DATA_TYPE float ENCODING BOOL DIMS 2",
       "1: TENSOR ENCODING BOOL needs a 8-bit integer DATA_TYPE"},
      {"TENSOR t DATA_TYPE uint8 ENCODING BFLOAT16 DIMS 2",
       "1: TENSOR ENCODING BFLOAT16 needs a 16-bit integer DATA_TYPE"},
      {"TENSOR t DATA_TYPE uint8 ENCODING HALF DIMS 2",
       "1: unknown TENSOR ENCODING: HALF"},
      {"TENSOR t DATA_TYPE float DIMS 2 3 STRIDES 12",
       "1: TENSOR STRIDES needs one stride per dimension"},
      {"TENSOR t DATA_TYPE float DIMS 2 3 STRIDES 16 8",
       "1: TENSOR innermost stride must be the element size (4)"},
      {"TENSOR t DATA_TYPE float DIMS 2 3 STRIDES 8 4",
       "1: TENSOR stride 0 overlaps the next dimension"},
      {"TENSOR t DATA_TYPE float DIMS 2 3 STRIDES 14 4",
       "1: TENSOR strides must be multiples of the element size"},
      {"TENSOR t DATA_TYPE float DIMS 2 DATA 1 2 3 END",
       "1: TENSOR t has 12 bytes of data but its DIMS need 8"},
      {"TENSOR t DATA_TYPE float DIMS 2 SIZE 2",
       "1: unknown TENSOR initializer: SIZE"},
  };

  for (const auto& test : cases) {
    Parser parser;
    Result r = parser.Parse(test.in);
    ASSERT_FALSE(r.IsSuccess()) << test.in;
    EXPECT_EQ(std::string(test.err), r.Error()) << test.in;
  }
}

TEST_F(AmberScriptParserTest, BindTensors) {
  std::string in = R"(
SHADER compute my_shader GLSL
#version 460
void main() {}
END
TENSOR t DATA_TYPE float DIMS 2 FILL 0
TENSOR a0 DATA_TYPE int32 DIMS 3 FILL 1
TENSOR a1 DATA_TYPE int32 DIMS 4 FILL 2

PIPELINE compute my_pipeline
  ATTACH my_shader
  BIND BUFFER t AS tensor DESCRIPTOR_SET 0 BINDING 1
  BIND BUFFER_ARRAY a0 a1 AS tensor DESCRIPTOR_SET 1 BINDING 0
END
PIPELINE compute dump
  ATTACH my_shader
  BIND BUFFER t AS storage DESCRIPTOR_SET 0 BINDING 0
END
)";

  Parser parser;
  Result r = parser.Parse(in);
  ASSERT_TRUE(r.IsSuccess()) << r.Error();

  auto script = parser.GetScript();
  const auto& pipelines = script->GetPipelines();
  ASSERT_EQ(2U, pipelines.size());
  const auto& bufs = pipelines[0]->GetBuffers();
  ASSERT_EQ(3U, bufs.size());
  EXPECT_EQ(BufferType::kTensor, bufs[0].type);
  EXPECT_EQ(0U, bufs[0].descriptor_set);
  EXPECT_EQ(1U, bufs[0].binding);
  for (size_t i = 1; i < 3; ++i) {
    EXPECT_EQ(BufferType::kTensor, bufs[i].type);
    EXPECT_EQ(1U, bufs[i].descriptor_set);
    EXPECT_EQ(0U, bufs[i].binding);
  }
  EXPECT_EQ("a1", bufs[2].buffer->GetName());
  EXPECT_EQ(BufferType::kStorage, pipelines[1]->GetBuffers()[0].type);
}

TEST_F(AmberScriptParserTest, BindBufferAsTensorNeedsATensor) {
  std::string in = R"(
SHADER compute my_shader GLSL
#version 460
void main() {}
END
BUFFER b DATA_TYPE float SIZE 2 FILL 0

PIPELINE compute my_pipeline
  ATTACH my_shader
  BIND BUFFER b AS tensor DESCRIPTOR_SET 0 BINDING 0
END
)";

  Parser parser;
  Result r = parser.Parse(in);
  ASSERT_FALSE(r.IsSuccess());
  EXPECT_EQ("10: buffer b bound AS tensor must be declared with TENSOR",
            r.Error());
}

TEST_F(AmberScriptParserTest, TensorFeatures) {
  std::string in = R"(
DEVICE_FEATURE TensorFeaturesARM.tensors
DEVICE_FEATURE TensorFeaturesARM.shaderTensorAccess
DEVICE_FEATURE TensorFeaturesARM.tensorNonPacked
DEVICE_FEATURE TensorFeaturesARM.shaderStorageTensorArrayDynamicIndexing
DEVICE_FEATURE TensorFeaturesARM.shaderStorageTensorArrayNonUniformIndexing
DEVICE_FEATURE TensorFeaturesARM.descriptorBindingStorageTensorUpdateAfterBind
)";

  Parser parser;
  Result r = parser.Parse(in);
  ASSERT_TRUE(r.IsSuccess()) << r.Error();
  EXPECT_EQ(6U, parser.GetScript()->GetRequiredFeatures().size());
}

}  // namespace amberscript
}  // namespace amber
