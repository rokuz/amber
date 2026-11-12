// Copyright 2018 The Amber Authors.
// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "src/script.h"

#include "src/type_parser.h"

namespace amber {

Script::Script() : virtual_files_(std::make_unique<VirtualFileStore>()) {}

Script::~Script() = default;

std::vector<ShaderInfo> Script::GetShaderInfo() const {
  std::vector<ShaderInfo> ret;
  for (const auto& shader : shaders_) {
    bool in_pipeline = false;
    // A given shader could be in multiple pipelines with different
    // optimizations so make sure we check and report all pipelines.
    for (const auto& pipeline : pipelines_) {
      auto shader_info = pipeline->GetShader(shader.get());
      if (shader_info) {
        ret.emplace_back(
            ShaderInfo{shader->GetFormat(), shader->GetType(),
                       pipeline->GetName() + "-" + shader->GetName(),
                       shader->GetData(), shader_info->GetShaderOptimizations(),
                       shader->GetTargetEnv(), shader_info->GetData()});

        in_pipeline = true;
      }
    }

    if (!in_pipeline) {
      ret.emplace_back(ShaderInfo{shader->GetFormat(),
                                  shader->GetType(),
                                  shader->GetName(),
                                  shader->GetData(),
                                  {},
                                  shader->GetTargetEnv(),
                                  {}});
    }
  }
  return ret;
}

void Script::AddRequiredExtension(const std::string& ext) {
  // Make this smarter when we have more instance extensions to match.
  if (ext == "VK_KHR_get_physical_device_properties2") {
    AddRequiredInstanceExtension(ext);
  } else {
    AddRequiredDeviceExtension(ext);
  }
}

bool Script::IsKnownFeature(const std::string& name) const {
  return name == "robustBufferAccess" || name == "fullDrawIndexUint32" ||
         name == "imageCubeArray" || name == "independentBlend" ||
         name == "geometryShader" || name == "tessellationShader" ||
         name == "sampleRateShading" || name == "dualSrcBlend" ||
         name == "logicOp" || name == "multiDrawIndirect" ||
         name == "drawIndirectFirstInstance" || name == "depthClamp" ||
         name == "depthBiasClamp" || name == "fillModeNonSolid" ||
         name == "depthBounds" || name == "wideLines" ||
         name == "largePoints" || name == "alphaToOne" ||
         name == "multiViewport" || name == "samplerAnisotropy" ||
         name == "textureCompressionETC2" ||
         name == "textureCompressionASTC_LDR" ||
         name == "textureCompressionBC" || name == "occlusionQueryPrecise" ||
         name == "pipelineStatisticsQuery" ||
         name == "vertexPipelineStoresAndAtomics" ||
         name == "fragmentStoresAndAtomics" ||
         name == "shaderTessellationAndGeometryPointSize" ||
         name == "shaderImageGatherExtended" ||
         name == "shaderStorageImageExtendedFormats" ||
         name == "shaderStorageImageMultisample" ||
         name == "shaderStorageImageReadWithoutFormat" ||
         name == "shaderStorageImageWriteWithoutFormat" ||
         name == "shaderUniformBufferArrayDynamicIndexing" ||
         name == "shaderSampledImageArrayDynamicIndexing" ||
         name == "shaderStorageBufferArrayDynamicIndexing" ||
         name == "shaderStorageImageArrayDynamicIndexing" ||
         name == "shaderClipDistance" || name == "shaderCullDistance" ||
         name == "shaderFloat64" || name == "shaderInt64" ||
         name == "shaderInt16" || name == "shaderResourceResidency" ||
         name == "shaderResourceMinLod" || name == "sparseBinding" ||
         name == "sparseResidencyBuffer" || name == "sparseResidencyImage2D" ||
         name == "sparseResidencyImage3D" ||
         name == "sparseResidency2Samples" ||
         name == "sparseResidency4Samples" ||
         name == "sparseResidency8Samples" ||
         name == "sparseResidency16Samples" ||
         name == "sparseResidencyAliased" ||
         name == "variableMultisampleRate" || name == "inheritedQueries" ||
         name == "VariablePointerFeatures.variablePointers" ||
         name == "VariablePointerFeatures.variablePointersStorageBuffer" ||
         name == "Float16Int8Features.shaderFloat16" ||
         name == "VulkanMemoryModelFeatures.vulkanMemoryModel" ||
         name == "VulkanMemoryModelFeatures.vulkanMemoryModelDeviceScope" ||
         name ==
             "ZeroInitializeWorkgroupMemoryFeatures."
             "shaderZeroInitializeWorkgroupMemory" ||
         name == "Float16Int8Features.shaderInt8" ||
         name == "Storage8BitFeatures.storageBuffer8BitAccess" ||
         name == "Storage8BitFeatures.uniformAndStorageBuffer8BitAccess" ||
         name == "Storage8BitFeatures.storagePushConstant8" ||
         name == "Storage16BitFeatures.storageBuffer16BitAccess" ||
         name == "Storage16BitFeatures.uniformAndStorageBuffer16BitAccess" ||
         name == "Storage16BitFeatures.storagePushConstant16" ||
         name == "Storage16BitFeatures.storageInputOutput16" ||
         name == "DepthClampZeroOneFeatures.depthClampZeroOne" ||
         name == "SubgroupSizeControl.subgroupSizeControl" ||
         name == "SubgroupSizeControl.computeFullSubgroups" ||
         name == "SubgroupSupportedOperations.basic" ||
         name == "SubgroupSupportedOperations.vote" ||
         name == "SubgroupSupportedOperations.arithmetic" ||
         name == "SubgroupSupportedOperations.ballot" ||
         name == "SubgroupSupportedOperations.shuffle" ||
         name == "SubgroupSupportedOperations.shuffleRelative" ||
         name == "SubgroupSupportedOperations.clustered" ||
         name == "SubgroupSupportedOperations.quad" ||
         name == "SubgroupSupportedStages.vertex" ||
         name == "SubgroupSupportedStages.tessellationControl" ||
         name == "SubgroupSupportedStages.tessellationEvaluation" ||
         name == "SubgroupSupportedStages.geometry" ||
         name == "SubgroupSupportedStages.fragment" ||
         name == "SubgroupSupportedStages.compute" ||
         name == "SubgroupSupportedStages.task" ||
         name == "SubgroupSupportedStages.mesh" ||
         name == "SubgroupSupportedStages.rayGeneration" ||
         name == "SubgroupSupportedStages.anyHit" ||
         name == "SubgroupSupportedStages.closestHit" ||
         name == "SubgroupSupportedStages.miss" ||
         name == "SubgroupSupportedStages.intersection" ||
         name == "SubgroupSupportedStages.callable" ||
         name == "IndexTypeUint8Features.indexTypeUint8" ||
         name ==
             "ShaderSubgroupExtendedTypesFeatures"
             ".shaderSubgroupExtendedTypes" ||
         name == "CooperativeMatrixFeaturesKHR.cooperativeMatrix" ||
         name == "RayTracingPipelineFeaturesKHR.rayTracingPipeline" ||
         name == "RayQueryFeaturesKHR.rayQuery" ||
         name == "AccelerationStructureFeaturesKHR.accelerationStructure" ||
         name == "BufferDeviceAddressFeatures.bufferDeviceAddress" ||
         name == "RayTracingPipelineFeaturesKHR.rayTraversalPrimitiveCulling" ||
         name == "RayTracingPositionFetchFeaturesKHR.rayTracingPositionFetch" ||
         name == "RayTracingMaintenance1FeaturesKHR.rayTracingMaintenance1" ||
         name ==
             "RayTracingMaintenance1FeaturesKHR."
             "rayTracingPipelineTraceRaysIndirect2" ||
         name == "RayTracingMotionBlurFeaturesNV.rayTracingMotionBlur" ||
         name ==
             "RayTracingMotionBlurFeaturesNV."
             "rayTracingMotionBlurPipelineTraceRaysIndirect" ||
         name == "MeshShaderFeaturesEXT.taskShader" ||
         name == "MeshShaderFeaturesEXT.meshShader" ||
         name == "MeshShaderFeaturesEXT.multiviewMeshShader" ||
         name ==
             "MeshShaderFeaturesEXT."
             "primitiveFragmentShadingRateMeshShader" ||
         name == "MeshShaderFeaturesEXT.meshShaderQueries" ||
         name == "MeshShaderFeaturesNV.taskShader" ||
         name == "MeshShaderFeaturesNV.meshShader" ||
         name == "FragmentShadingRateFeaturesKHR.pipelineFragmentShadingRate" ||
         name ==
             "FragmentShadingRateFeaturesKHR."
             "primitiveFragmentShadingRate" ||
         name ==
             "FragmentShadingRateFeaturesKHR."
             "attachmentFragmentShadingRate" ||
         name ==
             "DescriptorIndexingFeatures."
             "shaderInputAttachmentArrayDynamicIndexing" ||
         name ==
             "DescriptorIndexingFeatures."
             "shaderUniformTexelBufferArrayDynamicIndexing" ||
         name ==
             "DescriptorIndexingFeatures."
             "shaderStorageTexelBufferArrayDynamicIndexing" ||
         name ==
             "DescriptorIndexingFeatures."
             "shaderUniformBufferArrayNonUniformIndexing" ||
         name ==
             "DescriptorIndexingFeatures."
             "shaderSampledImageArrayNonUniformIndexing" ||
         name ==
             "DescriptorIndexingFeatures."
             "shaderStorageBufferArrayNonUniformIndexing" ||
         name ==
             "DescriptorIndexingFeatures."
             "shaderStorageImageArrayNonUniformIndexing" ||
         name ==
             "DescriptorIndexingFeatures."
             "shaderInputAttachmentArrayNonUniformIndexing" ||
         name ==
             "DescriptorIndexingFeatures."
             "shaderUniformTexelBufferArrayNonUniformIndexing" ||
         name ==
             "DescriptorIndexingFeatures."
             "shaderStorageTexelBufferArrayNonUniformIndexing" ||
         name ==
             "DescriptorIndexingFeatures."
             "descriptorBindingUniformBufferUpdateAfterBind" ||
         name ==
             "DescriptorIndexingFeatures."
             "descriptorBindingSampledImageUpdateAfterBind" ||
         name ==
             "DescriptorIndexingFeatures."
             "descriptorBindingStorageImageUpdateAfterBind" ||
         name ==
             "DescriptorIndexingFeatures."
             "descriptorBindingStorageBufferUpdateAfterBind" ||
         name ==
             "DescriptorIndexingFeatures."
             "descriptorBindingUniformTexelBufferUpdateAfterBind" ||
         name ==
             "DescriptorIndexingFeatures."
             "descriptorBindingStorageTexelBufferUpdateAfterBind" ||
         name ==
             "DescriptorIndexingFeatures."
             "descriptorBindingUpdateUnusedWhilePending" ||
         name == "DescriptorIndexingFeatures.descriptorBindingPartiallyBound" ||
         name ==
             "DescriptorIndexingFeatures."
             "descriptorBindingVariableDescriptorCount" ||
         name == "DescriptorIndexingFeatures.runtimeDescriptorArray" ||
         name ==
             "ShaderDemoteToHelperInvocationFeatures."
             "shaderDemoteToHelperInvocation" ||
         name == "ShaderAtomicInt64Features.shaderBufferInt64Atomics" ||
         name == "ShaderAtomicInt64Features.shaderSharedInt64Atomics" ||
         name == "ShaderExpectAssumeFeaturesKHR.shaderExpectAssume" ||
         name == "ShaderAtomicFloatFeaturesEXT.shaderBufferFloat32Atomics" ||
         name == "ShaderAtomicFloatFeaturesEXT.shaderBufferFloat32AtomicAdd" ||
         name == "ShaderAtomicFloatFeaturesEXT.shaderBufferFloat64Atomics" ||
         name == "ShaderAtomicFloatFeaturesEXT.shaderBufferFloat64AtomicAdd" ||
         name == "ShaderAtomicFloatFeaturesEXT.shaderSharedFloat32Atomics" ||
         name == "ShaderAtomicFloatFeaturesEXT.shaderSharedFloat32AtomicAdd" ||
         name == "ShaderAtomicFloatFeaturesEXT.shaderSharedFloat64Atomics" ||
         name == "ShaderAtomicFloatFeaturesEXT.shaderSharedFloat64AtomicAdd" ||
         name == "ShaderAtomicFloatFeaturesEXT.shaderImageFloat32Atomics" ||
         name == "ShaderAtomicFloatFeaturesEXT.shaderImageFloat32AtomicAdd" ||
         name == "ShaderAtomicFloatFeaturesEXT.sparseImageFloat32Atomics" ||
         name == "ShaderAtomicFloatFeaturesEXT.sparseImageFloat32AtomicAdd" ||
         name == "ShaderAtomicFloat2FeaturesEXT.shaderBufferFloat16Atomics" ||
         name == "ShaderAtomicFloat2FeaturesEXT.shaderBufferFloat16AtomicAdd" ||
         name == "ShaderAtomicFloat2FeaturesEXT.shaderBufferFloat16AtomicMinMax" ||
         name == "ShaderAtomicFloat2FeaturesEXT.shaderBufferFloat32AtomicMinMax" ||
         name == "ShaderAtomicFloat2FeaturesEXT.shaderBufferFloat64AtomicMinMax" ||
         name == "ShaderAtomicFloat2FeaturesEXT.shaderSharedFloat16Atomics" ||
         name == "ShaderAtomicFloat2FeaturesEXT.shaderSharedFloat16AtomicAdd" ||
         name == "ShaderAtomicFloat2FeaturesEXT.shaderSharedFloat16AtomicMinMax" ||
         name == "ShaderAtomicFloat2FeaturesEXT.shaderSharedFloat32AtomicMinMax" ||
         name == "ShaderAtomicFloat2FeaturesEXT.shaderSharedFloat64AtomicMinMax" ||
         name == "ShaderAtomicFloat2FeaturesEXT.shaderImageFloat32AtomicMinMax" ||
         name == "ShaderAtomicFloat2FeaturesEXT.sparseImageFloat32AtomicMinMax" ||
         name == "ShaderImageAtomicInt64FeaturesEXT.shaderImageInt64Atomics" ||
         name == "ShaderImageAtomicInt64FeaturesEXT.sparseImageInt64Atomics" ||
         name == "MultiviewFeatures.multiview" ||
         name == "MultiviewFeatures.multiviewGeometryShader" ||
         name == "MultiviewFeatures.multiviewTessellationShader" ||
         name == "TransformFeedbackFeaturesEXT.transformFeedback" ||
         name == "TransformFeedbackFeaturesEXT.geometryStreams" ||
         name == "ShaderReplicatedCompositesFeaturesEXT.shaderReplicatedComposites" ||
         name == "ComputeShaderDerivativesFeaturesKHR.computeDerivativeGroupQuads" ||
         name == "ComputeShaderDerivativesFeaturesKHR.computeDerivativeGroupLinear" ||
         name == "ShaderSubgroupRotateFeaturesKHR.shaderSubgroupRotate" ||
         name == "ShaderSubgroupRotateFeaturesKHR.shaderSubgroupRotateClustered" ||
         name == "ShaderQuadControlFeaturesKHR.shaderQuadControl" ||
         name == "FragmentShaderInterlockFeaturesEXT.fragmentShaderSampleInterlock" ||
         name == "FragmentShaderInterlockFeaturesEXT.fragmentShaderPixelInterlock" ||
         name == "FragmentShaderInterlockFeaturesEXT.fragmentShaderShadingRateInterlock" ||
         name == "ShaderImageFootprintFeaturesNV.imageFootprint" ||
         name == "ShaderRelaxedExtendedInstructionFeaturesKHR."
                 "shaderRelaxedExtendedInstruction" ||
         name == "FragmentDensityMapFeaturesEXT.fragmentDensityMap" ||
         name == "ShadingRateImageFeaturesNV.shadingRateImage" ||
         name == "ComputeShaderDerivativesPropertiesKHR."
                 "meshAndTaskShaderDerivatives" ||
         name == "SubgroupSupportedOperations.partitionedNV" ||
         name == "ShaderLongVectorFeaturesEXT.longVector" ||
         name == "CooperativeMatrixFeaturesNV.cooperativeMatrix" ||
         name == "CooperativeMatrixFeaturesNV.cooperativeMatrixRobustBufferAccess" ||
         name == "CooperativeMatrix2FeaturesNV.cooperativeMatrixWorkgroupScope" ||
         name == "CooperativeMatrix2FeaturesNV.cooperativeMatrixFlexibleDimensions" ||
         name == "CooperativeMatrix2FeaturesNV.cooperativeMatrixReductions" ||
         name == "CooperativeMatrix2FeaturesNV.cooperativeMatrixConversions" ||
         name == "CooperativeMatrix2FeaturesNV.cooperativeMatrixPerElementOperations" ||
         name == "CooperativeMatrix2FeaturesNV.cooperativeMatrixTensorAddressing" ||
         name == "CooperativeMatrix2FeaturesNV.cooperativeMatrixBlockLoads" ||
         name == "CooperativeVectorFeaturesNV.cooperativeVector" ||
         name == "CooperativeVectorFeaturesNV.cooperativeVectorTraining" ||
         name == "ShaderBfloat16FeaturesKHR.shaderBFloat16Type" ||
         name == "ShaderBfloat16FeaturesKHR.shaderBFloat16DotProduct" ||
         name == "ShaderBfloat16FeaturesKHR.shaderBFloat16CooperativeMatrix" ||
         name == "ShaderFloat8FeaturesEXT.shaderFloat8" ||
         name == "ShaderFloat8FeaturesEXT.shaderFloat8CooperativeMatrix";
}

bool Script::IsKnownProperty(const std::string& name) const {
  return name ==
             "FloatControlsProperties.shaderSignedZeroInfNanPreserveFloat16" ||
         name ==
             "FloatControlsProperties.shaderSignedZeroInfNanPreserveFloat32" ||
         name ==
             "FloatControlsProperties.shaderSignedZeroInfNanPreserveFloat64" ||
         name == "FloatControlsProperties.shaderDenormPreserveFloat16" ||
         name == "FloatControlsProperties.shaderDenormPreserveFloat32" ||
         name == "FloatControlsProperties.shaderDenormPreserveFloat64" ||
         name == "FloatControlsProperties.shaderDenormFlushToZeroFloat16" ||
         name == "FloatControlsProperties.shaderDenormFlushToZeroFloat32" ||
         name == "FloatControlsProperties.shaderDenormFlushToZeroFloat64" ||
         name == "FloatControlsProperties.shaderRoundingModeRTEFloat16" ||
         name == "FloatControlsProperties.shaderRoundingModeRTEFloat32" ||
         name == "FloatControlsProperties.shaderRoundingModeRTEFloat64" ||
         name == "FloatControlsProperties.shaderRoundingModeRTZFloat16" ||
         name == "FloatControlsProperties.shaderRoundingModeRTZFloat32" ||
         name == "FloatControlsProperties.shaderRoundingModeRTZFloat64";
}

type::Type* Script::ParseType(const std::string& str) {
  auto type = GetType(str);
  if (type) {
    return type;
  }

  TypeParser parser;
  auto new_type = parser.Parse(str);
  if (new_type != nullptr) {
    type = new_type.get();
    RegisterType(std::move(new_type));
  }
  return type;
}

}  // namespace amber
