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

#ifndef SRC_VULKAN_NV_ACCELERATION_STRUCTURE_H_
#define SRC_VULKAN_NV_ACCELERATION_STRUCTURE_H_

#include <memory>
#include <vector>

#include "src/acceleration_structure.h"
#include "src/vulkan/device.h"
#include "src/vulkan/transfer_buffer.h"

namespace amber {
namespace vulkan {

class CommandBuffer;

/// An acceleration structure built with VK_NV_ray_tracing. A top level
/// structure owns the bottom level structures of its instances.
class NVAccelerationStructure {
 public:
  explicit NVAccelerationStructure(Device* device);
  ~NVAccelerationStructure();

  Result CreateBottomLevel(amber::BLAS* blas);
  Result CreateTopLevel(amber::TLAS* tlas);
  Result Build(CommandBuffer* command_buffer);

  VkAccelerationStructureNV GetVkAccelerationStructure() const {
    return structure_;
  }

 private:
  Result CreateStructure(const VkAccelerationStructureInfoNV& info);
  Result AllocateMemory(const VkMemoryRequirements& requirements);
  Result GetHandle(uint64_t* handle) const;

  Device* device_ = nullptr;
  VkAccelerationStructureNV structure_ = VK_NULL_HANDLE;
  VkDeviceMemory memory_ = VK_NULL_HANDLE;
  bool built_ = false;
  VkAccelerationStructureInfoNV info_ = {};
  std::vector<VkGeometryNV> geometries_;
  std::unique_ptr<TransferBuffer> data_buffer_;
  std::unique_ptr<TransferBuffer> scratch_buffer_;
  std::vector<std::unique_ptr<NVAccelerationStructure>> bottom_levels_;
};

}  // namespace vulkan
}  // namespace amber

#endif  // SRC_VULKAN_NV_ACCELERATION_STRUCTURE_H_
