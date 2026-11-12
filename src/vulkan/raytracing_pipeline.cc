// Copyright 2024 The Amber Authors.
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

#include <utility>

#include "src/vulkan/raytracing_pipeline.h"

#include "src/vulkan/blas.h"
#include "src/vulkan/command_pool.h"
#include "src/vulkan/device.h"
#include "src/vulkan/sbt.h"
#include "src/vulkan/tlas.h"

namespace amber {
namespace vulkan {

inline VkStridedDeviceAddressRegionKHR makeStridedDeviceAddressRegionKHR(
    VkDeviceAddress deviceAddress,
    VkDeviceSize stride,
    VkDeviceSize size) {
  VkStridedDeviceAddressRegionKHR res;
  res.deviceAddress = deviceAddress;
  res.stride = stride;
  res.size = size;
  return res;
}

RayTracingPipeline::RayTracingPipeline(
    Device* device,
    BlasesMap* blases,
    TlasesMap* tlases,
    uint32_t fence_timeout_ms,
    bool pipeline_runtime_layer_enabled,
    const std::vector<VkPipelineShaderStageCreateInfo>& shader_stage_info,
    VkPipelineCreateFlags create_flags)
    : Pipeline(PipelineType::kRayTracing,
               device,
               fence_timeout_ms,
               pipeline_runtime_layer_enabled,
               shader_stage_info,
               create_flags),
      shader_group_create_info_(),
      blases_(blases),
      tlases_(tlases) {}

RayTracingPipeline::~RayTracingPipeline() = default;

Result RayTracingPipeline::Initialize(
    CommandPool* pool,
    std::vector<VkRayTracingShaderGroupCreateInfoKHR>&
        shader_group_create_info) {
  shader_group_create_info_.swap(shader_group_create_info);

  return Pipeline::Initialize(pool);
}

Result RayTracingPipeline::CreateVkRayTracingPipeline(
    const VkPipelineLayout& pipeline_layout,
    VkPipeline* pipeline,
    const std::vector<VkPipeline>& libs,
    uint32_t maxPipelineRayPayloadSize,
    uint32_t maxPipelineRayHitAttributeSize,
    uint32_t maxPipelineRayRecursionDepth) {
  if (nv_) {
    return CreateVkRayTracingPipelineNV(pipeline_layout, pipeline,
                                        maxPipelineRayRecursionDepth);
  }

  std::vector<VkPipelineShaderStageCreateInfo> shader_stage_info =
      GetVkShaderStageInfo();

  for (auto& info : shader_stage_info) {
    info.pName = GetEntryPointName(info.stage);
  }

  const bool lib = (create_flags_ & VK_PIPELINE_CREATE_LIBRARY_BIT_KHR) != 0;
  const VkPipelineLibraryCreateInfoKHR libraryInfo = {
      VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR, nullptr,
      static_cast<uint32_t>(libs.size()), libs.size() ? &libs[0] : nullptr};
  const VkRayTracingPipelineInterfaceCreateInfoKHR libraryInterface = {
      VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_INTERFACE_CREATE_INFO_KHR, nullptr,
      maxPipelineRayPayloadSize, maxPipelineRayHitAttributeSize};

  VkRayTracingPipelineCreateInfoKHR pipelineCreateInfo{
      VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
      nullptr,
      create_flags_,
      static_cast<uint32_t>(shader_stage_info.size()),
      shader_stage_info.data(),
      static_cast<uint32_t>(shader_group_create_info_.size()),
      shader_group_create_info_.data(),
      maxPipelineRayRecursionDepth,
      libs.empty() ? nullptr : &libraryInfo,
      lib || !libs.empty() ? &libraryInterface : nullptr,
      nullptr,
      pipeline_layout,
      VK_NULL_HANDLE,
      0,
  };

  VkResult r = device_->GetPtrs()->vkCreateRayTracingPipelinesKHR(
      device_->GetVkDevice(), VK_NULL_HANDLE, VK_NULL_HANDLE, 1u,
      &pipelineCreateInfo, nullptr, pipeline);
  if (r != VK_SUCCESS) {
    return Result("Vulkan::Calling vkCreateRayTracingPipelinesKHR Fail");
  }

  return {};
}

Result RayTracingPipeline::CreateVkRayTracingPipelineNV(
    const VkPipelineLayout& pipeline_layout,
    VkPipeline* pipeline,
    uint32_t maxPipelineRayRecursionDepth) {
  if (device_->GetPtrs()->vkCreateRayTracingPipelinesNV == nullptr ||
      device_->GetPtrs()->vkCmdTraceRaysNV == nullptr) {
    return Result("Vulkan: NV ray tracing pipelines require VK_NV_ray_tracing");
  }

  std::vector<VkPipelineShaderStageCreateInfo> shader_stage_info =
      GetVkShaderStageInfo();
  for (auto& info : shader_stage_info) {
    info.pName = GetEntryPointName(info.stage);
  }

  std::vector<VkRayTracingShaderGroupCreateInfoNV> groups;
  for (const auto& group : shader_group_create_info_) {
    VkRayTracingShaderGroupCreateInfoNV nv_group = {};
    nv_group.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_NV;
    nv_group.type = group.type;
    nv_group.generalShader = group.generalShader;
    nv_group.closestHitShader = group.closestHitShader;
    nv_group.anyHitShader = group.anyHitShader;
    nv_group.intersectionShader = group.intersectionShader;
    groups.push_back(nv_group);
  }

  VkRayTracingPipelineCreateInfoNV create_info = {};
  create_info.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_NV;
  create_info.flags = create_flags_;
  create_info.stageCount = static_cast<uint32_t>(shader_stage_info.size());
  create_info.pStages = shader_stage_info.data();
  create_info.groupCount = static_cast<uint32_t>(groups.size());
  create_info.pGroups = groups.data();
  create_info.maxRecursionDepth = maxPipelineRayRecursionDepth;
  create_info.layout = pipeline_layout;
  create_info.basePipelineIndex = -1;

  VkResult r = device_->GetPtrs()->vkCreateRayTracingPipelinesNV(
      device_->GetVkDevice(), VK_NULL_HANDLE, 1u, &create_info, nullptr,
      pipeline);
  if (r != VK_SUCCESS) {
    return Result("Vulkan::Calling vkCreateRayTracingPipelinesNV Fail");
  }

  return {};
}

Result RayTracingPipeline::getVulkanSBTRegion(
    VkPipeline pipeline,
    amber::SBT* aSBT,
    VkStridedDeviceAddressRegionKHR* region) {
  if (aSBT != nullptr) {
    SBT* vSBT = nullptr;
    auto x = sbtses_.find(aSBT);

    if (x == sbtses_.end()) {
      auto p = std::make_unique<amber::vulkan::SBT>(device_);
      sbts_.push_back(std::move(p));
      auto sbt_vulkan = sbtses_.emplace(aSBT, sbts_.back().get());

      vSBT = sbt_vulkan.first->second;

      Result r = vSBT->Create(aSBT, pipeline, nv_);
      if (!r.IsSuccess()) {
        return r;
      }
    } else {
      vSBT = x->second;
    }

    *region = makeStridedDeviceAddressRegionKHR(
        vSBT->GetDeviceAddress(), vSBT->GetStride(), vSBT->GetSize());
  } else {
    *region = makeStridedDeviceAddressRegionKHR(0, 0, 0);
  }

  return {};
}

Result RayTracingPipeline::WriteShaderRecordAddresses(amber::SBT* aSBT) {
  if (aSBT == nullptr) {
    return {};
  }

  for (auto& record : aSBT->GetSBTRecords()) {
    Result r =
        WriteAccelerationStructureAddresses(record->GetShaderRecordBuffer());
    if (!r.IsSuccess()) {
      return r;
    }
  }

  return {};
}

Result RayTracingPipeline::InitLibrary(const std::vector<VkPipeline>& libs,
                                       uint32_t maxPipelineRayPayloadSize,
                                       uint32_t maxPipelineRayHitAttributeSize,
                                       uint32_t maxPipelineRayRecursionDepth) {
  assert(pipeline_layout_ == VK_NULL_HANDLE);
  Result r = CreateVkPipelineLayout(&pipeline_layout_);
  if (!r.IsSuccess()) {
    return r;
  }

  assert(pipeline_ == VK_NULL_HANDLE);
  r = CreateVkRayTracingPipeline(
      pipeline_layout_, &pipeline_, libs, maxPipelineRayPayloadSize,
      maxPipelineRayHitAttributeSize, maxPipelineRayRecursionDepth);
  if (!r.IsSuccess()) {
    return r;
  }

  return {};
}

Result RayTracingPipeline::TraceRays(amber::SBT* rSBT,
                                     amber::SBT* mSBT,
                                     amber::SBT* hSBT,
                                     amber::SBT* cSBT,
                                     uint32_t x,
                                     uint32_t y,
                                     uint32_t z,
                                     uint32_t maxPipelineRayPayloadSize,
                                     uint32_t maxPipelineRayHitAttributeSize,
                                     uint32_t maxPipelineRayRecursionDepth,
                                     const std::vector<VkPipeline>& libs,
                                     bool is_timed_execution) {
  Result r = SendDescriptorDataToDeviceIfNeeded();
  if (!r.IsSuccess()) {
    return r;
  }

  for (amber::SBT* sbt : {rSBT, mSBT, hSBT, cSBT}) {
    r = WriteShaderRecordAddresses(sbt);
    if (!r.IsSuccess()) {
      return r;
    }
  }

  r = InitLibrary(libs, maxPipelineRayPayloadSize,
                  maxPipelineRayHitAttributeSize, maxPipelineRayRecursionDepth);
  if (!r.IsSuccess()) {
    return r;
  }

  // Note that a command updating a descriptor set and a command using
  // it must be submitted separately, because using a descriptor set
  // while updating it is not safe.
  UpdateDescriptorSetsIfNeeded();
  CreateTimingQueryObjectIfNeeded(is_timed_execution);
  {
    CommandBufferGuard guard(GetCommandBuffer());
    if (!guard.IsRecording()) {
      return guard.GetResult();
    }

    for (auto& i : *blases_) {
      i.second->BuildBLAS(GetCommandBuffer());
    }
    for (auto& i : *tlases_) {
      Result tlas_result = i.second->BuildTLAS(GetCommandBuffer());
      if (!tlas_result.IsSuccess()) {
        return tlas_result;
      }
    }

    BindVkDescriptorSets(pipeline_layout_);

    r = RecordPushConstant(pipeline_layout_);
    if (!r.IsSuccess()) {
      return r;
    }

    device_->GetPtrs()->vkCmdBindPipeline(
        command_->GetVkCommandBuffer(), VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
        pipeline_);

    VkStridedDeviceAddressRegionKHR rSBTRegion = {};
    VkStridedDeviceAddressRegionKHR mSBTRegion = {};
    VkStridedDeviceAddressRegionKHR hSBTRegion = {};
    VkStridedDeviceAddressRegionKHR cSBTRegion = {};

    r = getVulkanSBTRegion(pipeline_, rSBT, &rSBTRegion);
    if (!r.IsSuccess()) {
      return r;
    }

    r = getVulkanSBTRegion(pipeline_, mSBT, &mSBTRegion);
    if (!r.IsSuccess()) {
      return r;
    }

    r = getVulkanSBTRegion(pipeline_, hSBT, &hSBTRegion);
    if (!r.IsSuccess()) {
      return r;
    }

    r = getVulkanSBTRegion(pipeline_, cSBT, &cSBTRegion);
    if (!r.IsSuccess()) {
      return r;
    }

    if (nv_) {
      if (x == 0 || y == 0 || z == 0) {
        return Result("Vulkan: NV ray tracing needs a non-empty launch size");
      }
      auto table = [this](amber::SBT* sbt, VkBuffer* buffer,
                          VkDeviceSize* offset, VkDeviceSize* stride) {
        *buffer = VK_NULL_HANDLE;
        *offset = 0;
        *stride = 0;
        auto found = sbtses_.find(sbt);
        if (sbt != nullptr && found != sbtses_.end()) {
          *buffer = found->second->getBuffer()->GetVkBuffer();
          *offset = found->second->GetOffset();
          *stride = found->second->GetStride();
        }
      };
      VkBuffer buffers[4];
      VkDeviceSize offsets[4];
      VkDeviceSize strides[4];
      amber::SBT* tables[4] = {rSBT, mSBT, hSBT, cSBT};
      for (size_t i = 0; i < 4; ++i) {
        table(tables[i], &buffers[i], &offsets[i], &strides[i]);
      }
      if (buffers[0] == VK_NULL_HANDLE) {
        return Result("Vulkan: NV ray tracing needs a ray generation table");
      }
      device_->GetPtrs()->vkCmdTraceRaysNV(
          command_->GetVkCommandBuffer(), buffers[0], offsets[0], buffers[1],
          offsets[1], strides[1], buffers[2], offsets[2], strides[2],
          buffers[3], offsets[3], strides[3], x, y, z);
    } else {
      device_->GetPtrs()->vkCmdTraceRaysKHR(command_->GetVkCommandBuffer(),
                                            &rSBTRegion, &mSBTRegion,
                                            &hSBTRegion, &cSBTRegion, x, y, z);
    }
    BeginTimerQuery();
    r = guard.Submit(GetFenceTimeout(), GetPipelineRuntimeLayerEnabled());
    EndTimerQuery();
    if (!r.IsSuccess()) {
      return r;
    }
  }
  DestroyTimingQueryObjectIfNeeded();
  r = ReadbackDescriptorsToHostDataQueue();
  if (!r.IsSuccess()) {
    return r;
  }

  return {};
}

}  // namespace vulkan
}  // namespace amber
