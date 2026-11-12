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

#include "src/vulkan/transfer_tensor.h"

#include <cstring>
#include <limits>

#include "src/vulkan/command_buffer.h"
#include "src/vulkan/device.h"

namespace amber {
namespace vulkan {
namespace {

// The bfloat16 and 8-bit float tensor formats are newer than the Vulkan
// headers Amber is built with.
const VkFormat kFormatBool = static_cast<VkFormat>(1000460000);
const VkFormat kFormatBFloat16 = static_cast<VkFormat>(1000460001);
const VkFormat kFormatFloat8E4M3 = static_cast<VkFormat>(1000460002);
const VkFormat kFormatFloat8E5M2 = static_cast<VkFormat>(1000460003);

VkFormat TensorFormat(Device* device, const Buffer& buffer) {
  switch (buffer.GetTensorEncoding()) {
    case TensorEncoding::kBool:
      return kFormatBool;
    case TensorEncoding::kBFloat16:
      return kFormatBFloat16;
    case TensorEncoding::kFloat8E4M3:
      return kFormatFloat8E4M3;
    case TensorEncoding::kFloat8E5M2:
      return kFormatFloat8E5M2;
    case TensorEncoding::kNone:
      break;
  }
  return device->GetVkFormat(*buffer.GetFormat());
}

uint32_t TensorSize(const Buffer& buffer) {
  const auto& dims = buffer.GetTensorDims();
  const auto& strides = buffer.GetTensorStrides();
  if (!strides.empty() && !dims.empty()) {
    return static_cast<uint32_t>(dims[0] * strides[0]);
  }
  return buffer.GetSizeInBytes();
}

}  // namespace

TransferTensor::TransferTensor(Device* device, const Buffer& buffer)
    : Resource(device, TensorSize(buffer)),
      format_(TensorFormat(device, buffer)),
      element_size_(buffer.GetFormat()->SizeInBytes()),
      element_count_(buffer.ElementCount()) {
  for (uint64_t dim : buffer.GetTensorDims()) {
    dims_.push_back(static_cast<int64_t>(dim));
  }
  const auto& strides = buffer.GetTensorStrides();
  packed_ = strides.empty();
  if (packed_) {
    strides_.resize(dims_.size());
    int64_t stride = element_size_;
    for (size_t i = dims_.size(); i > 0; --i) {
      strides_[i - 1] = stride;
      stride *= dims_[i - 1];
    }
  } else {
    for (uint64_t stride : strides) {
      strides_.push_back(static_cast<int64_t>(stride));
    }
  }
}

TransferTensor::~TransferTensor() {
  if (!device_) {
    return;
  }
  if (view_ != VK_NULL_HANDLE) {
    device_->GetPtrs()->vkDestroyTensorViewARM(device_->GetVkDevice(), view_,
                                               nullptr);
  }
  if (tensor_ != VK_NULL_HANDLE) {
    device_->GetPtrs()->vkDestroyTensorARM(device_->GetVkDevice(), tensor_,
                                           nullptr);
  }
  if (memory_ != VK_NULL_HANDLE) {
    UnMapMemory(memory_);
    device_->GetPtrs()->vkFreeMemory(device_->GetVkDevice(), memory_, nullptr);
  }
}

Result TransferTensor::Initialize() {
  if (tensor_ != VK_NULL_HANDLE) {
    return Result("Vulkan: TransferTensor::Initialize() already initialized");
  }
  const auto* ptrs = device_->GetPtrs();
  if (!ptrs->vkCreateTensorARM || !ptrs->vkGetTensorMemoryRequirementsARM ||
      !ptrs->vkBindTensorMemoryARM || !ptrs->vkCreateTensorViewARM) {
    return Result(
        "Vulkan: tensors need VK_ARM_tensors (DEVICE_FEATURE "
        "TensorFeaturesARM.tensors)");
  }

  VkTensorDescriptionARM description = {};
  description.sType = VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM;
  description.tiling = VK_TENSOR_TILING_LINEAR_ARM;
  description.format = format_;
  description.dimensionCount = static_cast<uint32_t>(dims_.size());
  description.pDimensions = dims_.data();
  description.pStrides = packed_ ? nullptr : strides_.data();
  description.usage = VK_TENSOR_USAGE_SHADER_BIT_ARM |
                      VK_TENSOR_USAGE_TRANSFER_SRC_BIT_ARM |
                      VK_TENSOR_USAGE_TRANSFER_DST_BIT_ARM;

  VkTensorCreateInfoARM create_info = {};
  create_info.sType = VK_STRUCTURE_TYPE_TENSOR_CREATE_INFO_ARM;
  create_info.pDescription = &description;
  create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (ptrs->vkCreateTensorARM(device_->GetVkDevice(), &create_info, nullptr,
                              &tensor_) != VK_SUCCESS) {
    return Result("Vulkan::Calling vkCreateTensorARM Fail");
  }

  VkTensorMemoryRequirementsInfoARM requirements_info = {};
  requirements_info.sType =
      VK_STRUCTURE_TYPE_TENSOR_MEMORY_REQUIREMENTS_INFO_ARM;
  requirements_info.tensor = tensor_;
  VkMemoryRequirements2 requirements = {};
  requirements.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
  ptrs->vkGetTensorMemoryRequirementsARM(device_->GetVkDevice(),
                                         &requirements_info, &requirements);

  const uint32_t memory_type_index =
      ChooseMemory(requirements.memoryRequirements.memoryTypeBits,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                   true);
  if (memory_type_index == std::numeric_limits<uint32_t>::max()) {
    return Result(
        "Vulkan: TransferTensor::Initialize() no host visible and host "
        "coherent memory for the tensor");
  }
  Result r = AllocateMemory(&memory_, requirements.memoryRequirements.size,
                            memory_type_index);
  if (!r.IsSuccess()) {
    return r;
  }

  VkBindTensorMemoryInfoARM bind_info = {};
  bind_info.sType = VK_STRUCTURE_TYPE_BIND_TENSOR_MEMORY_INFO_ARM;
  bind_info.tensor = tensor_;
  bind_info.memory = memory_;
  bind_info.memoryOffset = 0;
  if (ptrs->vkBindTensorMemoryARM(device_->GetVkDevice(), 1, &bind_info) !=
      VK_SUCCESS) {
    return Result("Vulkan::Calling vkBindTensorMemoryARM Fail");
  }

  VkTensorViewCreateInfoARM view_info = {};
  view_info.sType = VK_STRUCTURE_TYPE_TENSOR_VIEW_CREATE_INFO_ARM;
  view_info.tensor = tensor_;
  view_info.format = format_;
  if (ptrs->vkCreateTensorViewARM(device_->GetVkDevice(), &view_info, nullptr,
                                  &view_) != VK_SUCCESS) {
    return Result("Vulkan::Calling vkCreateTensorViewARM Fail");
  }

  r = MapMemory(memory_);
  if (!r.IsSuccess()) {
    return r;
  }
  std::memset(HostAccessibleMemoryPtr(), 0, GetSizeInBytes());
  return {};
}

uint64_t TransferTensor::ElementOffset(uint64_t index) const {
  uint64_t offset = 0;
  for (size_t i = dims_.size(); i > 0; --i) {
    const auto dim = static_cast<uint64_t>(dims_[i - 1]);
    offset += (index % dim) * static_cast<uint64_t>(strides_[i - 1]);
    index /= dim;
  }
  return offset;
}

void TransferTensor::WritePacked(const std::vector<uint8_t>& packed) {
  auto* memory = static_cast<uint8_t*>(HostAccessibleMemoryPtr());
  for (uint64_t i = 0; i < element_count_; ++i) {
    const uint64_t source = i * element_size_;
    if (source + element_size_ > packed.size()) {
      break;
    }
    std::memcpy(memory + ElementOffset(i), packed.data() + source,
                element_size_);
  }
}

void TransferTensor::ReadPacked(std::vector<uint8_t>* packed) const {
  const auto* memory = static_cast<const uint8_t*>(HostAccessibleMemoryPtr());
  packed->resize(element_count_ * element_size_);
  for (uint64_t i = 0; i < element_count_; ++i) {
    std::memcpy(packed->data() + i * element_size_, memory + ElementOffset(i),
                element_size_);
  }
}

void TransferTensor::CopyToDevice(CommandBuffer* command_buffer) {
  MemoryBarrier(command_buffer);
}

void TransferTensor::CopyToHost(CommandBuffer* command_buffer) {
  MemoryBarrier(command_buffer);
}

}  // namespace vulkan
}  // namespace amber
