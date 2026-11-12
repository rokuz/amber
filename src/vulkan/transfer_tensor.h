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

#ifndef SRC_VULKAN_TRANSFER_TENSOR_H_
#define SRC_VULKAN_TRANSFER_TENSOR_H_

#include <cstdint>
#include <vector>

#include "amber/result.h"
#include "amber/vulkan_header.h"
#include "src/buffer.h"
#include "src/vulkan/resource.h"

namespace amber {
namespace vulkan {

class CommandBuffer;
class Device;

/// A linear VK_ARM_tensors tensor bound to host visible, host coherent
/// memory. Amber buffers hold the tensor elements packed in row-major order;
/// the tensor memory follows the tensor strides, so uploads scatter and
/// readbacks gather the elements.
class TransferTensor : public Resource {
 public:
  TransferTensor(Device* device, const Buffer& buffer);
  ~TransferTensor() override;

  TransferTensor* AsTransferTensor() override { return this; }
  Result Initialize() override;

  VkTensorARM GetVkTensor() const { return tensor_; }
  VkTensorViewARM GetVkTensorView() const { return view_; }

  /// Records a memory barrier: the memory is host coherent, so host writes
  /// are visible to the device once the command buffer is submitted.
  void CopyToDevice(CommandBuffer* command_buffer) override;
  /// Records a memory barrier making shader writes visible to the host.
  void CopyToHost(CommandBuffer* command_buffer) override;

  /// Writes packed row-major elements into the tensor memory.
  void WritePacked(const std::vector<uint8_t>& packed);
  /// Reads the tensor memory into packed row-major elements.
  void ReadPacked(std::vector<uint8_t>* packed) const;

 private:
  uint64_t ElementOffset(uint64_t index) const;

  VkFormat format_ = VK_FORMAT_UNDEFINED;
  uint32_t element_size_ = 0;
  uint64_t element_count_ = 0;
  std::vector<int64_t> dims_;
  std::vector<int64_t> strides_;
  bool packed_ = true;
  VkTensorARM tensor_ = VK_NULL_HANDLE;
  VkDeviceMemory memory_ = VK_NULL_HANDLE;
  VkTensorViewARM view_ = VK_NULL_HANDLE;
};

}  // namespace vulkan
}  // namespace amber

#endif  // SRC_VULKAN_TRANSFER_TENSOR_H_
