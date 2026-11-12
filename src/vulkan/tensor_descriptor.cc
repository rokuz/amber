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

#include "src/vulkan/tensor_descriptor.h"

#include <memory>
#include <vector>

#include "src/vulkan/device.h"
#include "src/vulkan/transfer_tensor.h"

namespace amber {
namespace vulkan {

TensorDescriptor::TensorDescriptor(Buffer* buffer,
                                   Device* device,
                                   uint32_t desc_set,
                                   uint32_t binding,
                                   Pipeline* pipeline)
    : BufferBackedDescriptor(buffer,
                             DescriptorType::kTensor,
                             device,
                             desc_set,
                             binding,
                             pipeline) {}

TensorDescriptor::~TensorDescriptor() = default;

Result TensorDescriptor::CreateResourceIfNeeded() {
  auto& transfer_resources = pipeline_->GetDescriptorTransferResources();
  for (const auto& amber_buffer : GetAmberBuffers()) {
    if (!amber_buffer->IsTensor()) {
      return Result("Vulkan: buffer " + amber_buffer->GetName() +
                    " bound as a tensor is not a TENSOR");
    }
    if (transfer_resources.count(amber_buffer) == 0) {
      transfer_resources[amber_buffer] =
          std::make_unique<TransferTensor>(device_, *amber_buffer);
    } else if (!transfer_resources[amber_buffer]->AsTransferTensor()) {
      return Result("Vulkan: tensor " + amber_buffer->GetName() +
                    " is also bound as another kind of resource");
    }
  }
  is_descriptor_set_update_needed_ = true;
  return {};
}

void TensorDescriptor::UpdateDescriptorSetIfNeeded(
    VkDescriptorSet descriptor_set) {
  if (!is_descriptor_set_update_needed_) {
    return;
  }

  std::vector<VkTensorViewARM> views;
  for (const auto& amber_buffer : GetAmberBuffers()) {
    views.push_back(pipeline_->GetDescriptorTransferResources()[amber_buffer]
                        ->AsTransferTensor()
                        ->GetVkTensorView());
  }

  VkWriteDescriptorSetTensorARM tensor_write = {};
  tensor_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_TENSOR_ARM;
  tensor_write.tensorViewCount = static_cast<uint32_t>(views.size());
  tensor_write.pTensorViews = views.data();

  VkWriteDescriptorSet write = VkWriteDescriptorSet();
  write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write.pNext = &tensor_write;
  write.dstSet = descriptor_set;
  write.dstBinding = binding_;
  write.dstArrayElement = 0;
  write.descriptorCount = static_cast<uint32_t>(views.size());
  write.descriptorType = GetVkDescriptorType();

  device_->GetPtrs()->vkUpdateDescriptorSets(device_->GetVkDevice(), 1, &write,
                                             0, nullptr);
  is_descriptor_set_update_needed_ = false;
}

}  // namespace vulkan
}  // namespace amber
