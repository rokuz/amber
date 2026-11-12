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

#ifndef SRC_VULKAN_TENSOR_DESCRIPTOR_H_
#define SRC_VULKAN_TENSOR_DESCRIPTOR_H_

#include "amber/result.h"
#include "amber/vulkan_header.h"
#include "src/buffer.h"
#include "src/vulkan/buffer_backed_descriptor.h"
#include "src/vulkan/pipeline.h"

namespace amber {
namespace vulkan {

class Device;

/// A VK_DESCRIPTOR_TYPE_TENSOR_ARM descriptor, or an array of them.
class TensorDescriptor : public BufferBackedDescriptor {
 public:
  TensorDescriptor(Buffer* buffer,
                   Device* device,
                   uint32_t desc_set,
                   uint32_t binding,
                   Pipeline* pipeline);
  ~TensorDescriptor() override;

  void UpdateDescriptorSetIfNeeded(VkDescriptorSet descriptor_set) override;
  Result CreateResourceIfNeeded() override;
};

}  // namespace vulkan
}  // namespace amber

#endif  // SRC_VULKAN_TENSOR_DESCRIPTOR_H_
