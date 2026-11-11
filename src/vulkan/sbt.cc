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

#include <algorithm>
#include <cstring>

#include "src/buffer.h"
#include "src/vulkan/pipeline.h"
#include "src/vulkan/sbt.h"

namespace amber {
namespace vulkan {
namespace {

VkDeviceSize AlignUp(VkDeviceSize value, VkDeviceSize alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

}  // namespace

SBT::SBT(Device* device) : device_(device) {}

Result SBT::Create(amber::SBT* sbt, VkPipeline pipeline) {
  uint32_t handles_count = 0;
  uint32_t record_size = 0;
  for (auto& x : sbt->GetSBTRecords()) {
    handles_count += x->GetCount();
    Buffer* record = x->GetShaderRecordBuffer();
    if (record != nullptr) {
      record_size = std::max(record_size, record->GetSizeInBytes());
    }
  }

  if (handles_count == 0) {
    return Result("SBT must contain at least one record");
  }

  const uint32_t handle_size = device_->GetRayTracingShaderGroupHandleSize();
  const VkDeviceSize handle_alignment =
      std::max(1u, device_->GetRayTracingShaderGroupHandleAlignment());
  const VkDeviceSize base_alignment =
      std::max(1u, device_->GetRayTracingShaderGroupBaseAlignment());
  stride_ = AlignUp(handle_size + record_size, handle_alignment);

  const uint32_t max_stride = device_->GetRayTracingMaxShaderGroupStride();
  if (record_size > 0 && max_stride > 0 && stride_ > max_stride) {
    return Result("SBT record stride exceeds maxShaderGroupStride");
  }

  size_ = stride_ * handles_count;
  const VkDeviceSize buffer_size = size_ + base_alignment;
  std::vector<uint8_t> handles(handle_size * handles_count);

  buffer_ = std::make_unique<TransferBuffer>(
      device_, static_cast<uint32_t>(buffer_size), nullptr);
  buffer_->AddUsageFlags(VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                         VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR |
                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
  buffer_->AddAllocateFlags(VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT);
  Result r = buffer_->Initialize();
  if (!r.IsSuccess()) {
    return r;
  }

  size_t start = 0;
  for (auto& x : sbt->GetSBTRecords()) {
    const uint32_t index = x->GetIndex();
    const uint32_t count = x->GetCount();
    if (index != static_cast<uint32_t>(-1)) {
      VkResult vr = device_->GetPtrs()->vkGetRayTracingShaderGroupHandlesKHR(
          device_->GetVkDevice(), pipeline, index, count, count * handle_size,
          &handles[start * handle_size]);

      if (vr != VK_SUCCESS) {
        return Result("vkGetRayTracingShaderGroupHandlesKHR has failed");
      }
    }

    start += count;
  }

  const VkBufferDeviceAddressInfo address_info = {
      VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_KHR,
      nullptr,
      buffer_->GetVkBuffer(),
  };
  const VkDeviceAddress buffer_address =
      device_->GetPtrs()->vkGetBufferDeviceAddress(device_->GetVkDevice(),
                                                   &address_info);
  address_ = AlignUp(buffer_address, base_alignment);

  uint8_t* memory = static_cast<uint8_t*>(buffer_->HostAccessibleMemoryPtr());
  memset(memory, 0, static_cast<size_t>(buffer_size));
  uint8_t* table = memory + (address_ - buffer_address);

  start = 0;
  for (auto& x : sbt->GetSBTRecords()) {
    Buffer* record = x->GetShaderRecordBuffer();
    for (uint32_t i = 0; i < x->GetCount(); ++i, ++start) {
      uint8_t* slot = table + start * stride_;
      memcpy(slot, &handles[start * handle_size], handle_size);
      if (record != nullptr && !record->ValuePtr()->empty()) {
        memcpy(slot + handle_size, record->ValuePtr()->data(),
               std::min(static_cast<size_t>(record->GetSizeInBytes()),
                        record->ValuePtr()->size()));
      }
    }
  }

  // Skip flush as memory allocated for buffer is coherent

  return r;
}

SBT::~SBT() = default;

}  // namespace vulkan
}  // namespace amber
