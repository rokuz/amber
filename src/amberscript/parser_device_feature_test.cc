// Copyright 2019 The Amber Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or parseried.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "gtest/gtest.h"
#include "src/amberscript/parser.h"

namespace amber {
namespace amberscript {

using AmberScriptParserTest = testing::Test;

TEST_F(AmberScriptParserTest, DeviceFeature) {
  std::string in = R"(
DEVICE_FEATURE vertexPipelineStoresAndAtomics
DEVICE_FEATURE VariablePointerFeatures.variablePointersStorageBuffer
DEVICE_FEATURE Float16Int8Features.shaderFloat16
DEVICE_FEATURE Float16Int8Features.shaderInt8
DEVICE_FEATURE Storage8BitFeatures.storageBuffer8BitAccess
DEVICE_FEATURE Storage8BitFeatures.uniformAndStorageBuffer8BitAccess
DEVICE_FEATURE Storage8BitFeatures.storagePushConstant8
DEVICE_FEATURE Storage16BitFeatures.storageBuffer16BitAccess
DEVICE_FEATURE Storage16BitFeatures.uniformAndStorageBuffer16BitAccess
DEVICE_FEATURE Storage16BitFeatures.storagePushConstant16
DEVICE_FEATURE Storage16BitFeatures.storageInputOutput16
DEVICE_FEATURE SubgroupSizeControl.subgroupSizeControl
DEVICE_FEATURE SubgroupSizeControl.computeFullSubgroups
DEVICE_FEATURE VulkanMemoryModelFeatures.vulkanMemoryModel
DEVICE_FEATURE VulkanMemoryModelFeatures.vulkanMemoryModelDeviceScope
DEVICE_FEATURE CooperativeMatrixFeaturesKHR.cooperativeMatrix
DEVICE_FEATURE ZeroInitializeWorkgroupMemoryFeatures.shaderZeroInitializeWorkgroupMemory
DEVICE_FEATURE ShaderLongVectorFeaturesEXT.longVector)";

  Parser parser;
  Result r = parser.Parse(in);
  ASSERT_TRUE(r.IsSuccess()) << r.Error();

  auto script = parser.GetScript();
  const auto& features = script->GetRequiredFeatures();
  ASSERT_EQ(18U, features.size());
  EXPECT_EQ("vertexPipelineStoresAndAtomics", features[0]);
  EXPECT_EQ("VariablePointerFeatures.variablePointersStorageBuffer",
            features[1]);
  EXPECT_EQ("Float16Int8Features.shaderFloat16", features[2]);
  EXPECT_EQ("Float16Int8Features.shaderInt8", features[3]);
  EXPECT_EQ("Storage8BitFeatures.storageBuffer8BitAccess", features[4]);
  EXPECT_EQ("Storage8BitFeatures.uniformAndStorageBuffer8BitAccess",
            features[5]);
  EXPECT_EQ("Storage8BitFeatures.storagePushConstant8", features[6]);
  EXPECT_EQ("Storage16BitFeatures.storageBuffer16BitAccess", features[7]);
  EXPECT_EQ("Storage16BitFeatures.uniformAndStorageBuffer16BitAccess",
            features[8]);
  EXPECT_EQ("Storage16BitFeatures.storagePushConstant16", features[9]);
  EXPECT_EQ("Storage16BitFeatures.storageInputOutput16", features[10]);
  EXPECT_EQ("SubgroupSizeControl.subgroupSizeControl", features[11]);
  EXPECT_EQ("SubgroupSizeControl.computeFullSubgroups", features[12]);
  EXPECT_EQ("VulkanMemoryModelFeatures.vulkanMemoryModel", features[13]);
  EXPECT_EQ("VulkanMemoryModelFeatures.vulkanMemoryModelDeviceScope",
    features[14]);
  EXPECT_EQ("CooperativeMatrixFeaturesKHR.cooperativeMatrix", features[15]);
  EXPECT_EQ("ZeroInitializeWorkgroupMemoryFeatures."
            "shaderZeroInitializeWorkgroupMemory",
    features[16]);
  EXPECT_EQ("ShaderLongVectorFeaturesEXT.longVector", features[17]);
}

TEST_F(AmberScriptParserTest, DeviceFeatureExtensionStructs) {
  std::string in = R"(
DEVICE_FEATURE RayTracingPipelineFeaturesKHR.rayTraversalPrimitiveCulling
DEVICE_FEATURE RayTracingPositionFetchFeaturesKHR.rayTracingPositionFetch
DEVICE_FEATURE RayTracingMaintenance1FeaturesKHR.rayTracingMaintenance1
DEVICE_FEATURE RayTracingMaintenance1FeaturesKHR.rayTracingPipelineTraceRaysIndirect2
DEVICE_FEATURE RayTracingMotionBlurFeaturesNV.rayTracingMotionBlur
DEVICE_FEATURE RayTracingMotionBlurFeaturesNV.rayTracingMotionBlurPipelineTraceRaysIndirect
DEVICE_FEATURE MeshShaderFeaturesEXT.taskShader
DEVICE_FEATURE MeshShaderFeaturesEXT.meshShader
DEVICE_FEATURE MeshShaderFeaturesEXT.multiviewMeshShader
DEVICE_FEATURE MeshShaderFeaturesEXT.primitiveFragmentShadingRateMeshShader
DEVICE_FEATURE MeshShaderFeaturesEXT.meshShaderQueries
DEVICE_FEATURE MeshShaderFeaturesNV.taskShader
DEVICE_FEATURE MeshShaderFeaturesNV.meshShader
DEVICE_FEATURE FragmentShadingRateFeaturesKHR.pipelineFragmentShadingRate
DEVICE_FEATURE FragmentShadingRateFeaturesKHR.primitiveFragmentShadingRate
DEVICE_FEATURE FragmentShadingRateFeaturesKHR.attachmentFragmentShadingRate
DEVICE_FEATURE DescriptorIndexingFeatures.shaderInputAttachmentArrayDynamicIndexing
DEVICE_FEATURE DescriptorIndexingFeatures.shaderUniformTexelBufferArrayDynamicIndexing
DEVICE_FEATURE DescriptorIndexingFeatures.shaderStorageTexelBufferArrayDynamicIndexing
DEVICE_FEATURE DescriptorIndexingFeatures.shaderUniformBufferArrayNonUniformIndexing
DEVICE_FEATURE DescriptorIndexingFeatures.shaderSampledImageArrayNonUniformIndexing
DEVICE_FEATURE DescriptorIndexingFeatures.shaderStorageBufferArrayNonUniformIndexing
DEVICE_FEATURE DescriptorIndexingFeatures.shaderStorageImageArrayNonUniformIndexing
DEVICE_FEATURE DescriptorIndexingFeatures.shaderInputAttachmentArrayNonUniformIndexing
DEVICE_FEATURE DescriptorIndexingFeatures.shaderUniformTexelBufferArrayNonUniformIndexing
DEVICE_FEATURE DescriptorIndexingFeatures.shaderStorageTexelBufferArrayNonUniformIndexing
DEVICE_FEATURE DescriptorIndexingFeatures.descriptorBindingUniformBufferUpdateAfterBind
DEVICE_FEATURE DescriptorIndexingFeatures.descriptorBindingSampledImageUpdateAfterBind
DEVICE_FEATURE DescriptorIndexingFeatures.descriptorBindingStorageImageUpdateAfterBind
DEVICE_FEATURE DescriptorIndexingFeatures.descriptorBindingStorageBufferUpdateAfterBind
DEVICE_FEATURE DescriptorIndexingFeatures.descriptorBindingUniformTexelBufferUpdateAfterBind
DEVICE_FEATURE DescriptorIndexingFeatures.descriptorBindingStorageTexelBufferUpdateAfterBind
DEVICE_FEATURE DescriptorIndexingFeatures.descriptorBindingUpdateUnusedWhilePending
DEVICE_FEATURE DescriptorIndexingFeatures.descriptorBindingPartiallyBound
DEVICE_FEATURE DescriptorIndexingFeatures.descriptorBindingVariableDescriptorCount
DEVICE_FEATURE DescriptorIndexingFeatures.runtimeDescriptorArray
DEVICE_FEATURE ShaderDemoteToHelperInvocationFeatures.shaderDemoteToHelperInvocation
DEVICE_FEATURE ShaderAtomicInt64Features.shaderBufferInt64Atomics
DEVICE_FEATURE ShaderAtomicInt64Features.shaderSharedInt64Atomics)";

  Parser parser;
  Result r = parser.Parse(in);
  ASSERT_TRUE(r.IsSuccess()) << r.Error();

  auto script = parser.GetScript();
  const auto& features = script->GetRequiredFeatures();
  ASSERT_EQ(39U, features.size());
  EXPECT_EQ("RayTracingPipelineFeaturesKHR.rayTraversalPrimitiveCulling",
            features[0]);
  EXPECT_EQ("RayTracingPositionFetchFeaturesKHR.rayTracingPositionFetch",
            features[1]);
  EXPECT_EQ("RayTracingMaintenance1FeaturesKHR.rayTracingMaintenance1",
            features[2]);
  EXPECT_EQ(
      "RayTracingMaintenance1FeaturesKHR."
      "rayTracingPipelineTraceRaysIndirect2",
      features[3]);
  EXPECT_EQ("RayTracingMotionBlurFeaturesNV.rayTracingMotionBlur", features[4]);
  EXPECT_EQ(
      "RayTracingMotionBlurFeaturesNV."
      "rayTracingMotionBlurPipelineTraceRaysIndirect",
      features[5]);
  EXPECT_EQ("MeshShaderFeaturesEXT.taskShader", features[6]);
  EXPECT_EQ("MeshShaderFeaturesEXT.meshShader", features[7]);
  EXPECT_EQ("MeshShaderFeaturesEXT.multiviewMeshShader", features[8]);
  EXPECT_EQ("MeshShaderFeaturesEXT.primitiveFragmentShadingRateMeshShader",
            features[9]);
  EXPECT_EQ("MeshShaderFeaturesEXT.meshShaderQueries", features[10]);
  EXPECT_EQ("MeshShaderFeaturesNV.taskShader", features[11]);
  EXPECT_EQ("MeshShaderFeaturesNV.meshShader", features[12]);
  EXPECT_EQ("FragmentShadingRateFeaturesKHR.pipelineFragmentShadingRate",
            features[13]);
  EXPECT_EQ("FragmentShadingRateFeaturesKHR.primitiveFragmentShadingRate",
            features[14]);
  EXPECT_EQ("FragmentShadingRateFeaturesKHR.attachmentFragmentShadingRate",
            features[15]);
  EXPECT_EQ(
      "DescriptorIndexingFeatures."
      "shaderInputAttachmentArrayDynamicIndexing",
      features[16]);
  EXPECT_EQ(
      "DescriptorIndexingFeatures."
      "shaderUniformTexelBufferArrayDynamicIndexing",
      features[17]);
  EXPECT_EQ(
      "DescriptorIndexingFeatures."
      "shaderStorageTexelBufferArrayDynamicIndexing",
      features[18]);
  EXPECT_EQ(
      "DescriptorIndexingFeatures."
      "shaderUniformBufferArrayNonUniformIndexing",
      features[19]);
  EXPECT_EQ(
      "DescriptorIndexingFeatures."
      "shaderSampledImageArrayNonUniformIndexing",
      features[20]);
  EXPECT_EQ(
      "DescriptorIndexingFeatures."
      "shaderStorageBufferArrayNonUniformIndexing",
      features[21]);
  EXPECT_EQ(
      "DescriptorIndexingFeatures."
      "shaderStorageImageArrayNonUniformIndexing",
      features[22]);
  EXPECT_EQ(
      "DescriptorIndexingFeatures."
      "shaderInputAttachmentArrayNonUniformIndexing",
      features[23]);
  EXPECT_EQ(
      "DescriptorIndexingFeatures."
      "shaderUniformTexelBufferArrayNonUniformIndexing",
      features[24]);
  EXPECT_EQ(
      "DescriptorIndexingFeatures."
      "shaderStorageTexelBufferArrayNonUniformIndexing",
      features[25]);
  EXPECT_EQ(
      "DescriptorIndexingFeatures."
      "descriptorBindingUniformBufferUpdateAfterBind",
      features[26]);
  EXPECT_EQ(
      "DescriptorIndexingFeatures."
      "descriptorBindingSampledImageUpdateAfterBind",
      features[27]);
  EXPECT_EQ(
      "DescriptorIndexingFeatures."
      "descriptorBindingStorageImageUpdateAfterBind",
      features[28]);
  EXPECT_EQ(
      "DescriptorIndexingFeatures."
      "descriptorBindingStorageBufferUpdateAfterBind",
      features[29]);
  EXPECT_EQ(
      "DescriptorIndexingFeatures."
      "descriptorBindingUniformTexelBufferUpdateAfterBind",
      features[30]);
  EXPECT_EQ(
      "DescriptorIndexingFeatures."
      "descriptorBindingStorageTexelBufferUpdateAfterBind",
      features[31]);
  EXPECT_EQ(
      "DescriptorIndexingFeatures."
      "descriptorBindingUpdateUnusedWhilePending",
      features[32]);
  EXPECT_EQ("DescriptorIndexingFeatures.descriptorBindingPartiallyBound",
            features[33]);
  EXPECT_EQ(
      "DescriptorIndexingFeatures."
      "descriptorBindingVariableDescriptorCount",
      features[34]);
  EXPECT_EQ("DescriptorIndexingFeatures.runtimeDescriptorArray", features[35]);
  EXPECT_EQ(
      "ShaderDemoteToHelperInvocationFeatures."
      "shaderDemoteToHelperInvocation",
      features[36]);
  EXPECT_EQ("ShaderAtomicInt64Features.shaderBufferInt64Atomics", features[37]);
  EXPECT_EQ("ShaderAtomicInt64Features.shaderSharedInt64Atomics", features[38]);
}

TEST_F(AmberScriptParserTest, DeviceFeatureSubgroupStages) {
  std::string in = R"(
DEVICE_FEATURE SubgroupSupportedStages.task
DEVICE_FEATURE SubgroupSupportedStages.mesh
DEVICE_FEATURE SubgroupSupportedStages.rayGeneration
DEVICE_FEATURE SubgroupSupportedStages.anyHit
DEVICE_FEATURE SubgroupSupportedStages.closestHit
DEVICE_FEATURE SubgroupSupportedStages.miss
DEVICE_FEATURE SubgroupSupportedStages.intersection
DEVICE_FEATURE SubgroupSupportedStages.callable)";

  Parser parser;
  Result r = parser.Parse(in);
  ASSERT_TRUE(r.IsSuccess()) << r.Error();
  EXPECT_EQ(8U, parser.GetScript()->GetRequiredFeatures().size());
}

TEST_F(AmberScriptParserTest, DeviceFeatureMissingFeature) {
  std::string in = "DEVICE_FEATURE";

  Parser parser;
  Result r = parser.Parse(in);
  ASSERT_FALSE(r.IsSuccess());
  EXPECT_EQ("1: missing feature name for DEVICE_FEATURE command", r.Error());
}

TEST_F(AmberScriptParserTest, DeviceFeatureUnknown) {
  std::string in = "DEVICE_FEATURE unknown";

  Parser parser;
  Result r = parser.Parse(in);
  ASSERT_FALSE(r.IsSuccess());
  EXPECT_EQ("1: unknown feature name for DEVICE_FEATURE command", r.Error());
}

TEST_F(AmberScriptParserTest, DeviceFeatureInvalid) {
  std::string in = "DEVICE_FEATURE 12345";

  Parser parser;
  Result r = parser.Parse(in);
  ASSERT_FALSE(r.IsSuccess());
  EXPECT_EQ("1: invalid feature name for DEVICE_FEATURE command", r.Error());
}

TEST_F(AmberScriptParserTest, DeviceFeatureExtraParams) {
  std::string in = "DEVICE_FEATURE vertexPipelineStoresAndAtomics EXTRA";

  Parser parser;
  Result r = parser.Parse(in);
  ASSERT_FALSE(r.IsSuccess());
  EXPECT_EQ("1: extra parameters after DEVICE_FEATURE command: EXTRA",
            r.Error());
}

}  // namespace amberscript
}  // namespace amber
