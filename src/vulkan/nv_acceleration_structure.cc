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

#include "src/vulkan/nv_acceleration_structure.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>

#include "src/vulkan/command_buffer.h"

namespace amber {
namespace vulkan {
namespace {

VkDeviceSize AlignUp(VkDeviceSize value, VkDeviceSize alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

VkTransformMatrixKHR MakeTransform(const float* m) {
  VkTransformMatrixKHR v = {{{1.0f, 0.0f, 0.0f, 0.0f},
                             {0.0f, 1.0f, 0.0f, 0.0f},
                             {0.0f, 0.0f, 1.0f, 0.0f}}};
  if (m == nullptr) {
    return v;
  }
  for (size_t i = 0; i < 12; i++) {
    v.matrix[i / 4][i % 4] = m[i];
  }
  return v;
}

Result CreateHostBuffer(Device* device,
                        VkDeviceSize size,
                        std::unique_ptr<TransferBuffer>* buffer) {
  if (size > std::numeric_limits<uint32_t>::max()) {
    return Result("Vulkan: NV acceleration structure buffer is too large");
  }
  *buffer = std::make_unique<TransferBuffer>(
      device, static_cast<uint32_t>(std::max<VkDeviceSize>(size, 4)), nullptr);
  (*buffer)->AddUsageFlags(VK_BUFFER_USAGE_RAY_TRACING_BIT_NV);
  return (*buffer)->Initialize();
}

}  // namespace

NVAccelerationStructure::NVAccelerationStructure(Device* device)
    : device_(device) {}

NVAccelerationStructure::~NVAccelerationStructure() {
  if (structure_ != VK_NULL_HANDLE &&
      device_->GetPtrs()->vkDestroyAccelerationStructureNV != nullptr) {
    device_->GetPtrs()->vkDestroyAccelerationStructureNV(device_->GetVkDevice(),
                                                         structure_, nullptr);
  }
  if (memory_ != VK_NULL_HANDLE) {
    device_->GetPtrs()->vkFreeMemory(device_->GetVkDevice(), memory_, nullptr);
  }
}

Result NVAccelerationStructure::CreateBottomLevel(amber::BLAS* blas) {
  if (structure_ != VK_NULL_HANDLE) {
    return Result("Cannot recreate acceleration structure");
  }

  auto& geometries = blas->GetGeometries();
  std::vector<VkDeviceSize> offsets(geometries.size());
  VkDeviceSize size = 0;
  for (size_t i = 0; i < geometries.size(); ++i) {
    offsets[i] = size;
    size += AlignUp(geometries[i]->GetData().size() * sizeof(float), 8);
  }

  Result r = CreateHostBuffer(device_, size, &data_buffer_);
  if (!r.IsSuccess()) {
    return r;
  }
  uint8_t* memory =
      static_cast<uint8_t*>(data_buffer_->HostAccessibleMemoryPtr());

  geometries_.resize(geometries.size());
  for (size_t i = 0; i < geometries.size(); ++i) {
    const auto& data = geometries[i]->GetData();
    std::memcpy(memory + offsets[i], data.data(), data.size() * sizeof(float));

    VkGeometryNV geometry = {};
    geometry.sType = VK_STRUCTURE_TYPE_GEOMETRY_NV;
    geometry.geometry.triangles.sType = VK_STRUCTURE_TYPE_GEOMETRY_TRIANGLES_NV;
    geometry.geometry.triangles.indexType = VK_INDEX_TYPE_NONE_NV;
    geometry.geometry.aabbs.sType = VK_STRUCTURE_TYPE_GEOMETRY_AABB_NV;
    geometry.flags = geometries[i]->GetFlags();
    if (geometries[i]->IsTriangle()) {
      geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_NV;
      geometry.geometry.triangles.vertexData = data_buffer_->GetVkBuffer();
      geometry.geometry.triangles.vertexOffset = offsets[i];
      geometry.geometry.triangles.vertexCount =
          static_cast<uint32_t>(geometries[i]->getVertexCount());
      geometry.geometry.triangles.vertexStride = 3 * sizeof(float);
      geometry.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    } else if (geometries[i]->IsAABB()) {
      geometry.geometryType = VK_GEOMETRY_TYPE_AABBS_NV;
      geometry.geometry.aabbs.aabbData = data_buffer_->GetVkBuffer();
      geometry.geometry.aabbs.numAABBs =
          static_cast<uint32_t>(geometries[i]->getPrimitiveCount());
      geometry.geometry.aabbs.stride = sizeof(VkAabbPositionsKHR);
      geometry.geometry.aabbs.offset = offsets[i];
    } else {
      return Result("Vulkan: unknown geometry type");
    }
    geometries_[i] = geometry;
  }

  info_ = {};
  info_.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV;
  info_.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_NV;
  info_.geometryCount = static_cast<uint32_t>(geometries_.size());
  info_.pGeometries = geometries_.data();
  return CreateStructure(info_);
}

Result NVAccelerationStructure::CreateTopLevel(amber::TLAS* tlas) {
  if (structure_ != VK_NULL_HANDLE) {
    return {};
  }

  auto& instances = tlas->GetInstances();
  std::map<amber::BLAS*, uint64_t> handles;
  std::vector<VkAccelerationStructureInstanceKHR> records;
  for (auto& instance : instances) {
    amber::BLAS* blas = instance->GetUsedBLAS();
    if (blas == nullptr) {
      return Result("Vulkan: instance without a bottom level structure");
    }
    auto found = handles.find(blas);
    if (found == handles.end()) {
      bottom_levels_.push_back(
          std::make_unique<NVAccelerationStructure>(device_));
      Result r = bottom_levels_.back()->CreateBottomLevel(blas);
      if (!r.IsSuccess()) {
        return r;
      }
      uint64_t handle = 0;
      r = bottom_levels_.back()->GetHandle(&handle);
      if (!r.IsSuccess()) {
        return r;
      }
      found = handles.emplace(blas, handle).first;
    }
    records.push_back(VkAccelerationStructureInstanceKHR{
        MakeTransform(instance->GetTransform()), instance->GetInstanceIndex(),
        instance->GetMask(), instance->GetOffset(), instance->GetFlags(),
        found->second});
  }

  Result r = CreateHostBuffer(
      device_, records.size() * sizeof(VkAccelerationStructureInstanceKHR),
      &data_buffer_);
  if (!r.IsSuccess()) {
    return r;
  }
  if (!records.empty()) {
    std::memcpy(data_buffer_->HostAccessibleMemoryPtr(), records.data(),
                records.size() * sizeof(VkAccelerationStructureInstanceKHR));
  }

  info_ = {};
  info_.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV;
  info_.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_NV;
  info_.instanceCount = static_cast<uint32_t>(records.size());
  return CreateStructure(info_);
}

Result NVAccelerationStructure::CreateStructure(
    const VkAccelerationStructureInfoNV& info) {
  const VulkanPtrs* ptrs = device_->GetPtrs();
  if (ptrs->vkCreateAccelerationStructureNV == nullptr ||
      ptrs->vkGetAccelerationStructureMemoryRequirementsNV == nullptr ||
      ptrs->vkBindAccelerationStructureMemoryNV == nullptr ||
      ptrs->vkCmdBuildAccelerationStructureNV == nullptr ||
      ptrs->vkGetAccelerationStructureHandleNV == nullptr) {
    return Result(
        "Vulkan: NV acceleration structures require VK_NV_ray_tracing");
  }

  VkAccelerationStructureCreateInfoNV create_info = {};
  create_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_NV;
  create_info.info = info;
  if (ptrs->vkCreateAccelerationStructureNV(device_->GetVkDevice(),
                                            &create_info, nullptr,
                                            &structure_) != VK_SUCCESS) {
    return Result("Vulkan::Calling vkCreateAccelerationStructureNV failed");
  }

  VkAccelerationStructureMemoryRequirementsInfoNV requirements_info = {};
  requirements_info.sType =
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO_NV;
  requirements_info.type =
      VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_OBJECT_NV;
  requirements_info.accelerationStructure = structure_;
  VkMemoryRequirements2 requirements = {};
  requirements.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
  ptrs->vkGetAccelerationStructureMemoryRequirementsNV(
      device_->GetVkDevice(), &requirements_info, &requirements);
  Result r = AllocateMemory(requirements.memoryRequirements);
  if (!r.IsSuccess()) {
    return r;
  }

  VkBindAccelerationStructureMemoryInfoNV bind = {};
  bind.sType = VK_STRUCTURE_TYPE_BIND_ACCELERATION_STRUCTURE_MEMORY_INFO_NV;
  bind.accelerationStructure = structure_;
  bind.memory = memory_;
  if (ptrs->vkBindAccelerationStructureMemoryNV(device_->GetVkDevice(), 1,
                                                &bind) != VK_SUCCESS) {
    return Result("Vulkan::Calling vkBindAccelerationStructureMemoryNV failed");
  }

  requirements_info.type =
      VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_BUILD_SCRATCH_NV;
  ptrs->vkGetAccelerationStructureMemoryRequirementsNV(
      device_->GetVkDevice(), &requirements_info, &requirements);
  return CreateHostBuffer(device_, requirements.memoryRequirements.size,
                          &scratch_buffer_);
}

Result NVAccelerationStructure::AllocateMemory(
    const VkMemoryRequirements& requirements) {
  uint32_t chosen = std::numeric_limits<uint32_t>::max();
  for (uint32_t index = 0; index < 32; ++index) {
    if ((requirements.memoryTypeBits & (1u << index)) == 0) {
      continue;
    }
    if (chosen == std::numeric_limits<uint32_t>::max()) {
      chosen = index;
    }
    if (device_->HasMemoryFlags(index, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
      chosen = index;
      break;
    }
  }
  if (chosen == std::numeric_limits<uint32_t>::max()) {
    return Result("Vulkan: no memory type for an NV acceleration structure");
  }

  VkMemoryAllocateInfo allocate_info = {};
  allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocate_info.allocationSize = requirements.size;
  allocate_info.memoryTypeIndex = chosen;
  if (device_->GetPtrs()->vkAllocateMemory(device_->GetVkDevice(),
                                           &allocate_info, nullptr,
                                           &memory_) != VK_SUCCESS) {
    return Result("Vulkan::Calling vkAllocateMemory failed");
  }
  return {};
}

Result NVAccelerationStructure::GetHandle(uint64_t* handle) const {
  if (device_->GetPtrs()->vkGetAccelerationStructureHandleNV(
          device_->GetVkDevice(), structure_, sizeof(*handle), handle) !=
      VK_SUCCESS) {
    return Result("Vulkan::Calling vkGetAccelerationStructureHandleNV failed");
  }
  return {};
}

Result NVAccelerationStructure::Build(CommandBuffer* command_buffer) {
  if (structure_ == VK_NULL_HANDLE) {
    return Result("Acceleration structure should be created first");
  }
  if (built_) {
    return {};
  }

  for (auto& bottom : bottom_levels_) {
    Result r = bottom->Build(command_buffer);
    if (!r.IsSuccess()) {
      return r;
    }
  }

  data_buffer_->CopyToDevice(command_buffer);

  const bool top = info_.type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_NV;
  VkCommandBuffer cmd = command_buffer->GetVkCommandBuffer();
  device_->GetPtrs()->vkCmdBuildAccelerationStructureNV(
      cmd, &info_, top ? data_buffer_->GetVkBuffer() : VK_NULL_HANDLE, 0,
      VK_FALSE, structure_, VK_NULL_HANDLE, scratch_buffer_->GetVkBuffer(), 0);

  const VkAccessFlags access = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_NV |
                               VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_NV;
  const VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                access, access};
  device_->GetPtrs()->vkCmdPipelineBarrier(
      cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV,
      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &barrier, 0, nullptr, 0,
      nullptr);

  built_ = true;
  return {};
}

}  // namespace vulkan
}  // namespace amber
