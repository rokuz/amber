// Copyright 2018 The Amber Authors.
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

#ifndef SRC_VULKAN_TRANSFER_IMAGE_H_
#define SRC_VULKAN_TRANSFER_IMAGE_H_

#include "amber/result.h"
#include "amber/vulkan_header.h"
#include "src/format.h"
#include "src/vulkan/resource.h"

namespace amber {
namespace vulkan {

class CommandBuffer;
class Device;

/// Wrapper around a Vulkan VkImage.
class TransferImage : public Resource {
 public:
  TransferImage(Device* device,
                const Format& format,
                VkImageAspectFlags aspect,
                VkImageType image_type,
                VkImageUsageFlags image_usage_flags,
                uint32_t x,
                uint32_t y,
                uint32_t z,
                uint32_t mip_levels,
                uint32_t base_mip_level,
                uint32_t used_mip_levels,
                uint32_t samples,
                uint32_t layers = 1,
                VkImageViewType view_type = VK_IMAGE_VIEW_TYPE_MAX_ENUM);
  ~TransferImage() override;

  TransferImage* AsTransferImage() override { return this; }
  Result Initialize() override;

  /// Lets a multisampled image take data from and give data to the host:
  /// copies run as compute dispatches through a storage view of the image,
  /// one sample after the other per texel. Must be called before Initialize();
  /// the device needs shaderStorageImageMultisample.
  void EnableMultisampleTransfer();
  VkImageView GetVkImageView() const { return view_; }

  void ImageBarrier(CommandBuffer* command_buffer,
                    VkImageLayout to_layout,
                    VkPipelineStageFlags to_stage);

  /// Records a command on |command_buffer| to copy the buffer contents from the
  /// host to the device.
  void CopyToDevice(CommandBuffer* command_buffer) override;
  /// Records a command on |command_buffer| to copy the buffer contents from the
  /// device to the host.
  void CopyToHost(CommandBuffer* command_buffer) override;

 private:
  Result CreateVkImageView(VkImageAspectFlags aspect);
  Result AllocateAndBindMemoryToVkImage(VkImage image,
                                        VkDeviceMemory* memory,
                                        VkMemoryPropertyFlags flags,
                                        bool force_flags,
                                        uint32_t* memory_type_index);
  VkBufferImageCopy CreateBufferImageCopy(VkImageAspectFlags aspect,
                                          uint32_t mip_level);

  VkImageViewType GetImageViewType() const;
  Result CreateMultisampleTransfer();
  void RecordMultisampleCopy(CommandBuffer* command_buffer,
                             bool upload,
                             VkImageLayout layout,
                             VkAccessFlags access,
                             VkPipelineStageFlags stage);

  /// An extra `VkBuffer` is used to facilitate the transfer of data from the
  /// host into the `VkImage` on the device.
  VkBuffer host_accessible_buffer_ = VK_NULL_HANDLE;
  VkDeviceMemory host_accessible_memory_ = VK_NULL_HANDLE;

  VkImageCreateInfo image_info_;
  VkImageAspectFlags aspect_;

  VkImage image_ = VK_NULL_HANDLE;
  VkImageView view_ = VK_NULL_HANDLE;
  VkDeviceMemory memory_ = VK_NULL_HANDLE;

  VkImageLayout layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
  VkPipelineStageFlags stage_ = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

  uint32_t mip_levels_;
  uint32_t base_mip_level_;
  uint32_t used_mip_levels_;
  uint32_t samples_;
  uint32_t layers_;
  VkImageViewType view_type_;

  bool multisample_transfer_ = false;
  VkImageView storage_view_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout ms_set_layout_ = VK_NULL_HANDLE;
  VkPipelineLayout ms_pipeline_layout_ = VK_NULL_HANDLE;
  VkDescriptorPool ms_pool_ = VK_NULL_HANDLE;
  VkDescriptorSet ms_set_ = VK_NULL_HANDLE;
  VkPipeline ms_upload_ = VK_NULL_HANDLE;
  VkPipeline ms_download_ = VK_NULL_HANDLE;
};

}  // namespace vulkan
}  // namespace amber

#endif  // SRC_VULKAN_TRANSFER_IMAGE_H_
