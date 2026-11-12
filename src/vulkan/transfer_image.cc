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

#include "src/vulkan/transfer_image.h"

#include <cstring>
#include <limits>
#include <vector>

#include "src/vulkan/command_buffer.h"
#include "src/vulkan/device.h"
#include "src/vulkan/multisample_copy_shaders.h"

namespace amber {
namespace vulkan {
namespace {

const VkImageCreateInfo kDefaultImageInfo = {
    VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, /* sType */
    nullptr,                             /* pNext */
    0,                                   /* flags */
    VK_IMAGE_TYPE_2D,                    /* imageType */
    VK_FORMAT_R8G8B8A8_UNORM,            /* format */
    {250, 250, 1},                       /* extent */
    1,                                   /* mipLevels */
    1,                                   /* arrayLayers */
    VK_SAMPLE_COUNT_1_BIT,               /* samples */
    VK_IMAGE_TILING_OPTIMAL,             /* tiling */
    0,                                   /* usage */
    VK_SHARING_MODE_EXCLUSIVE,           /* sharingMode */
    0,                                   /* queueFamilyIndexCount */
    nullptr,                             /* pQueueFamilyIndices */
    VK_IMAGE_LAYOUT_UNDEFINED,           /* initialLayout */
};

VkSampleCountFlagBits GetVkSampleCount(uint32_t samples) {
  switch (samples) {
    case 1u:
      return VK_SAMPLE_COUNT_1_BIT;
    case 2u:
      return VK_SAMPLE_COUNT_2_BIT;
    case 4u:
      return VK_SAMPLE_COUNT_4_BIT;
    case 8u:
      return VK_SAMPLE_COUNT_8_BIT;
    case 16u:
      return VK_SAMPLE_COUNT_16_BIT;
    case 32u:
      return VK_SAMPLE_COUNT_32_BIT;
    case 64u:
      return VK_SAMPLE_COUNT_64_BIT;
  }

  return VK_SAMPLE_COUNT_FLAG_BITS_MAX_ENUM;
}

}  // namespace

TransferImage::TransferImage(Device* device,
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
                             uint32_t layers,
                             VkImageViewType view_type)
    : Resource(
          device,
          x * y * z * layers * (samples > 1 ? samples : 1) *
              (format.SizeInBytes() +
               // D24_UNORM_S8_UINT requires 32bit component for depth when
               // performing buffer copies. Reserve extra room to handle that.
               (format.GetFormatType() == FormatType::kD24_UNORM_S8_UINT ? 1
                                                                         : 0))),
      image_info_(kDefaultImageInfo),
      aspect_(aspect),
      mip_levels_(mip_levels),
      base_mip_level_(base_mip_level),
      used_mip_levels_(used_mip_levels),
      samples_(samples),
      layers_(layers),
      view_type_(view_type) {
  image_info_.format = device_->GetVkFormat(format);
  image_info_.imageType = image_type;
  image_info_.extent = {x, y, z};
  image_info_.mipLevels = mip_levels;
  image_info_.arrayLayers = layers;
  if (view_type == VK_IMAGE_VIEW_TYPE_CUBE ||
      view_type == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY) {
    image_info_.flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
  }
  image_info_.samples = GetVkSampleCount(samples);
  image_info_.usage = image_usage_flags;
}

TransferImage::~TransferImage() {
  auto* ptrs = device_->GetPtrs();
  VkDevice vk_device = device_->GetVkDevice();
  if (ms_upload_ != VK_NULL_HANDLE) {
    ptrs->vkDestroyPipeline(vk_device, ms_upload_, nullptr);
  }
  if (ms_download_ != VK_NULL_HANDLE) {
    ptrs->vkDestroyPipeline(vk_device, ms_download_, nullptr);
  }
  if (ms_pool_ != VK_NULL_HANDLE) {
    ptrs->vkDestroyDescriptorPool(vk_device, ms_pool_, nullptr);
  }
  if (ms_pipeline_layout_ != VK_NULL_HANDLE) {
    ptrs->vkDestroyPipelineLayout(vk_device, ms_pipeline_layout_, nullptr);
  }
  if (ms_set_layout_ != VK_NULL_HANDLE) {
    ptrs->vkDestroyDescriptorSetLayout(vk_device, ms_set_layout_, nullptr);
  }
  if (storage_view_ != VK_NULL_HANDLE) {
    ptrs->vkDestroyImageView(vk_device, storage_view_, nullptr);
  }

  if (view_ != VK_NULL_HANDLE) {
    device_->GetPtrs()->vkDestroyImageView(device_->GetVkDevice(), view_,
                                           nullptr);
  }

  if (image_ != VK_NULL_HANDLE) {
    device_->GetPtrs()->vkDestroyImage(device_->GetVkDevice(), image_, nullptr);
  }

  if (memory_ != VK_NULL_HANDLE) {
    device_->GetPtrs()->vkFreeMemory(device_->GetVkDevice(), memory_, nullptr);
  }

  if (host_accessible_memory_ != VK_NULL_HANDLE) {
    UnMapMemory(host_accessible_memory_);
    device_->GetPtrs()->vkFreeMemory(device_->GetVkDevice(),
                                     host_accessible_memory_, nullptr);
  }

  if (host_accessible_buffer_ != VK_NULL_HANDLE) {
    device_->GetPtrs()->vkDestroyBuffer(device_->GetVkDevice(),
                                        host_accessible_buffer_, nullptr);
  }
}

Result TransferImage::Initialize() {
  if (image_ != VK_NULL_HANDLE) {
    return Result("Vulkan::TransferImage was already initialized");
  }

  if (device_->GetPtrs()->vkCreateImage(device_->GetVkDevice(), &image_info_,
                                        nullptr, &image_) != VK_SUCCESS) {
    return Result("Vulkan::Calling vkCreateImage Fail");
  }

  uint32_t memory_type_index = 0;
  Result r = AllocateAndBindMemoryToVkImage(image_, &memory_,
                                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                            false, &memory_type_index);
  if (!r.IsSuccess()) {
    return r;
  }

  if (aspect_ & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT) &&
      !(image_info_.usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)) {
    // Combined depth/stencil image used as a descriptor. Only one aspect can be
    // used for the image view.
    r = CreateVkImageView(VK_IMAGE_ASPECT_DEPTH_BIT);
  } else {
    r = CreateVkImageView(aspect_);
  }

  if (!r.IsSuccess()) {
    return r;
  }

  // For images, we always make a secondary buffer. When the tiling of an image
  // is optimal, read/write data from CPU does not show correct values. We need
  // a secondary buffer to convert the GPU-optimal data to CPU-readable data
  // and vice versa.
  r = CreateVkBuffer(
      &host_accessible_buffer_,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
          (multisample_transfer_ ? VK_BUFFER_USAGE_STORAGE_BUFFER_BIT : 0));
  if (!r.IsSuccess()) {
    return r;
  }

  memory_type_index = 0;
  r = AllocateAndBindMemoryToVkBuffer(host_accessible_buffer_,
                                      &host_accessible_memory_,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                      true, &memory_type_index);
  if (!r.IsSuccess()) {
    return r;
  }

  r = MapMemory(host_accessible_memory_);
  if (!r.IsSuccess()) {
    return r;
  }

  if (multisample_transfer_) {
    return CreateMultisampleTransfer();
  }
  return {};
}

void TransferImage::EnableMultisampleTransfer() {
  if (samples_ <= 1) {
    return;
  }
  multisample_transfer_ = true;
  image_info_.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
  image_info_.flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT |
                       VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
}

Result TransferImage::CreateMultisampleTransfer() {
  auto* ptrs = device_->GetPtrs();
  VkDevice vk_device = device_->GetVkDevice();

  VkFormat word_format = VK_FORMAT_UNDEFINED;
  const uint32_t* upload_code = nullptr;
  size_t upload_size = 0;
  const uint32_t* download_code = nullptr;
  size_t download_size = 0;
  const uint32_t texel_bytes = GetSizeInBytes() / (image_info_.extent.width *
                                                   image_info_.extent.height *
                                                   layers_ * samples_);
  switch (texel_bytes) {
    case 4:
      word_format = VK_FORMAT_R32_UINT;
      upload_code = kUpload1;
      upload_size = sizeof(kUpload1);
      download_code = kDownload1;
      download_size = sizeof(kDownload1);
      break;
    case 8:
      word_format = VK_FORMAT_R32G32_UINT;
      upload_code = kUpload2;
      upload_size = sizeof(kUpload2);
      download_code = kDownload2;
      download_size = sizeof(kDownload2);
      break;
    case 16:
      word_format = VK_FORMAT_R32G32B32A32_UINT;
      upload_code = kUpload4;
      upload_size = sizeof(kUpload4);
      download_code = kDownload4;
      download_size = sizeof(kDownload4);
      break;
    default:
      return Result(
          "Vulkan: only multisampled images with 4, 8 or 16 bytes per texel "
          "can take data from the host, not " +
          std::to_string(texel_bytes));
  }

  VkImageViewCreateInfo view_info = VkImageViewCreateInfo();
  view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  view_info.image = image_;
  view_info.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
  view_info.format = word_format;
  view_info.components = {VK_COMPONENT_SWIZZLE_IDENTITY,
                          VK_COMPONENT_SWIZZLE_IDENTITY,
                          VK_COMPONENT_SWIZZLE_IDENTITY,
                          VK_COMPONENT_SWIZZLE_IDENTITY};
  view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers_};
  VkImageViewUsageCreateInfo view_usage = VkImageViewUsageCreateInfo();
  view_usage.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO;
  view_usage.usage = VK_IMAGE_USAGE_STORAGE_BIT;
  view_info.pNext = &view_usage;
  if (ptrs->vkCreateImageView(vk_device, &view_info, nullptr,
                              &storage_view_) != VK_SUCCESS) {
    return Result(
        "Vulkan: cannot create the storage view of a multisampled image");
  }

  VkDescriptorSetLayoutBinding bindings[2] = {};
  bindings[0].binding = 0;
  bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  bindings[0].descriptorCount = 1;
  bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  bindings[1].binding = 1;
  bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bindings[1].descriptorCount = 1;
  bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  VkDescriptorSetLayoutCreateInfo set_info = VkDescriptorSetLayoutCreateInfo();
  set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  set_info.bindingCount = 2;
  set_info.pBindings = bindings;
  if (ptrs->vkCreateDescriptorSetLayout(vk_device, &set_info, nullptr,
                                        &ms_set_layout_) != VK_SUCCESS) {
    return Result("Vulkan::Calling vkCreateDescriptorSetLayout Fail");
  }

  VkPushConstantRange push_range = {VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                    3 * sizeof(uint32_t)};
  VkPipelineLayoutCreateInfo layout_info = VkPipelineLayoutCreateInfo();
  layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  layout_info.setLayoutCount = 1;
  layout_info.pSetLayouts = &ms_set_layout_;
  layout_info.pushConstantRangeCount = 1;
  layout_info.pPushConstantRanges = &push_range;
  if (ptrs->vkCreatePipelineLayout(vk_device, &layout_info, nullptr,
                                   &ms_pipeline_layout_) != VK_SUCCESS) {
    return Result("Vulkan::Calling vkCreatePipelineLayout Fail");
  }

  VkDescriptorPoolSize pool_sizes[2] = {
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}};
  VkDescriptorPoolCreateInfo pool_info = VkDescriptorPoolCreateInfo();
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.maxSets = 1;
  pool_info.poolSizeCount = 2;
  pool_info.pPoolSizes = pool_sizes;
  if (ptrs->vkCreateDescriptorPool(vk_device, &pool_info, nullptr, &ms_pool_) !=
      VK_SUCCESS) {
    return Result("Vulkan::Calling vkCreateDescriptorPool Fail");
  }

  VkDescriptorSetAllocateInfo alloc_info = VkDescriptorSetAllocateInfo();
  alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  alloc_info.descriptorPool = ms_pool_;
  alloc_info.descriptorSetCount = 1;
  alloc_info.pSetLayouts = &ms_set_layout_;
  if (ptrs->vkAllocateDescriptorSets(vk_device, &alloc_info, &ms_set_) !=
      VK_SUCCESS) {
    return Result("Vulkan::Calling vkAllocateDescriptorSets Fail");
  }

  VkDescriptorImageInfo image_info = {VK_NULL_HANDLE, storage_view_,
                                      VK_IMAGE_LAYOUT_GENERAL};
  VkDescriptorBufferInfo buffer_info = {host_accessible_buffer_, 0,
                                        VK_WHOLE_SIZE};
  VkWriteDescriptorSet writes[2] = {};
  writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes[0].dstSet = ms_set_;
  writes[0].dstBinding = 0;
  writes[0].descriptorCount = 1;
  writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  writes[0].pImageInfo = &image_info;
  writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes[1].dstSet = ms_set_;
  writes[1].dstBinding = 1;
  writes[1].descriptorCount = 1;
  writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[1].pBufferInfo = &buffer_info;
  ptrs->vkUpdateDescriptorSets(vk_device, 2, writes, 0, nullptr);

  const struct {
    const uint32_t* code;
    size_t size;
    VkPipeline* pipeline;
  } shaders[] = {{upload_code, upload_size, &ms_upload_},
                 {download_code, download_size, &ms_download_}};
  for (const auto& shader : shaders) {
    VkShaderModuleCreateInfo module_info = VkShaderModuleCreateInfo();
    module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_info.codeSize = shader.size;
    module_info.pCode = shader.code;
    VkShaderModule module = VK_NULL_HANDLE;
    if (ptrs->vkCreateShaderModule(vk_device, &module_info, nullptr,
                                   &module) != VK_SUCCESS) {
      return Result("Vulkan::Calling vkCreateShaderModule Fail");
    }
    VkComputePipelineCreateInfo pipeline_info = VkComputePipelineCreateInfo();
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = module;
    pipeline_info.stage.pName = "main";
    pipeline_info.layout = ms_pipeline_layout_;
    VkResult created = ptrs->vkCreateComputePipelines(
        vk_device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr,
        shader.pipeline);
    ptrs->vkDestroyShaderModule(vk_device, module, nullptr);
    if (created != VK_SUCCESS) {
      return Result(
          "Vulkan: cannot create the pipeline that copies a multisampled "
          "image");
    }
  }
  return {};
}

void TransferImage::RecordMultisampleCopy(CommandBuffer* command_buffer,
                                          bool upload,
                                          VkImageLayout layout,
                                          VkAccessFlags access,
                                          VkPipelineStageFlags stage) {
  auto* ptrs = device_->GetPtrs();
  VkCommandBuffer cmd = command_buffer->GetVkCommandBuffer();

  VkImageMemoryBarrier to_general = VkImageMemoryBarrier();
  to_general.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  to_general.srcAccessMask = access;
  to_general.dstAccessMask =
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  to_general.oldLayout = layout;
  to_general.newLayout = VK_IMAGE_LAYOUT_GENERAL;
  to_general.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_general.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_general.image = image_;
  to_general.subresourceRange = {aspect_, 0, VK_REMAINING_MIP_LEVELS, 0,
                                 VK_REMAINING_ARRAY_LAYERS};
  VkMemoryBarrier host_to_shader = VkMemoryBarrier();
  host_to_shader.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  host_to_shader.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
  host_to_shader.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  ptrs->vkCmdPipelineBarrier(
      cmd, stage | VK_PIPELINE_STAGE_HOST_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &host_to_shader, 0, nullptr,
      1, &to_general);

  ptrs->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          upload ? ms_upload_ : ms_download_);
  ptrs->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                ms_pipeline_layout_, 0, 1, &ms_set_, 0,
                                nullptr);
  const uint32_t extent[3] = {image_info_.extent.width,
                              image_info_.extent.height, samples_};
  ptrs->vkCmdPushConstants(cmd, ms_pipeline_layout_,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(extent),
                           extent);
  ptrs->vkCmdDispatch(cmd, image_info_.extent.width, image_info_.extent.height,
                      layers_);

  VkImageMemoryBarrier back = to_general;
  back.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  back.dstAccessMask = access;
  back.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
  back.newLayout = layout;
  VkMemoryBarrier shader_to_host = VkMemoryBarrier();
  shader_to_host.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  shader_to_host.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  shader_to_host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  ptrs->vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             stage | VK_PIPELINE_STAGE_HOST_BIT, 0, 1,
                             &shader_to_host, 0, nullptr, 1, &back);
}

VkImageViewType TransferImage::GetImageViewType() const {
  if (view_type_ != VK_IMAGE_VIEW_TYPE_MAX_ENUM) {
    return view_type_;
  }
  switch (image_info_.imageType) {
    case VK_IMAGE_TYPE_1D:
      return VK_IMAGE_VIEW_TYPE_1D;
    case VK_IMAGE_TYPE_2D:
      return VK_IMAGE_VIEW_TYPE_2D;
    case VK_IMAGE_TYPE_3D:
      return VK_IMAGE_VIEW_TYPE_3D;
    default:
      break;
  }

  // Default to 2D image view.
  return VK_IMAGE_VIEW_TYPE_2D;
}

Result TransferImage::CreateVkImageView(VkImageAspectFlags aspect) {
  VkImageViewCreateInfo image_view_info = VkImageViewCreateInfo();
  image_view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  image_view_info.image = image_;
  image_view_info.viewType = GetImageViewType();
  image_view_info.format = image_info_.format;
  image_view_info.components = {
      VK_COMPONENT_SWIZZLE_R,
      VK_COMPONENT_SWIZZLE_G,
      VK_COMPONENT_SWIZZLE_B,
      VK_COMPONENT_SWIZZLE_A,
  };
  image_view_info.subresourceRange = {
      aspect,           /* aspectMask */
      base_mip_level_,  /* baseMipLevel */
      used_mip_levels_, /* levelCount */
      0,                /* baseArrayLayer */
      layers_,          /* layerCount */
  };

  if (device_->GetPtrs()->vkCreateImageView(device_->GetVkDevice(),
                                            &image_view_info, nullptr,
                                            &view_) != VK_SUCCESS) {
    return Result("Vulkan::Calling vkCreateImageView Fail");
  }

  return {};
}

VkBufferImageCopy TransferImage::CreateBufferImageCopy(
    VkImageAspectFlags aspect,
    uint32_t mip_level) {
  VkBufferImageCopy copy_region = VkBufferImageCopy();
  if (aspect == VK_IMAGE_ASPECT_STENCIL_BIT) {
    // Store stencil data at the end of the buffer after depth data.
    copy_region.bufferOffset =
        GetSizeInBytes() -
        image_info_.extent.width * image_info_.extent.height * layers_;
  } else {
    copy_region.bufferOffset = 0;
  }
  // Row length of 0 results in tight packing of rows, so the row stride
  // is the number of texels times the texel stride.
  copy_region.bufferRowLength = 0;
  copy_region.bufferImageHeight = 0;
  copy_region.imageSubresource = {
      aspect,    /* aspectMask */
      mip_level, /* mipLevel */
      0,         /* baseArrayLayer */
      layers_,   /* layerCount */
  };
  copy_region.imageOffset = {0, 0, 0};
  copy_region.imageExtent = {image_info_.extent.width >> mip_level,
                             image_info_.extent.height >> mip_level,
                             image_info_.extent.depth};
  return copy_region;
}

void TransferImage::CopyToHost(CommandBuffer* command_buffer) {
  const VkImageAspectFlagBits aspects[] = {VK_IMAGE_ASPECT_COLOR_BIT,
                                           VK_IMAGE_ASPECT_DEPTH_BIT,
                                           VK_IMAGE_ASPECT_STENCIL_BIT};
  // Copy operations don't support multisample images.
  if (samples_ > 1) {
    if (multisample_transfer_) {
      RecordMultisampleCopy(command_buffer, false,
                            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                            VK_ACCESS_TRANSFER_READ_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT);
    }
    return;
  }

  std::vector<VkBufferImageCopy> copy_regions;
  uint32_t last_mip_level = used_mip_levels_ == VK_REMAINING_MIP_LEVELS
                                ? mip_levels_
                                : base_mip_level_ + used_mip_levels_;
  for (uint32_t i = base_mip_level_; i < last_mip_level; i++) {
    for (auto aspect : aspects) {
      if (aspect_ & aspect) {
        copy_regions.push_back(CreateBufferImageCopy(aspect, i));
      }
    }
  }

  device_->GetPtrs()->vkCmdCopyImageToBuffer(
      command_buffer->GetVkCommandBuffer(), image_,
      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, host_accessible_buffer_,
      static_cast<uint32_t>(copy_regions.size()), copy_regions.data());

  MemoryBarrier(command_buffer);
}

void TransferImage::CopyToDevice(CommandBuffer* command_buffer) {
  // Copy operations don't support multisample images.
  if (samples_ > 1) {
    if (multisample_transfer_) {
      RecordMultisampleCopy(command_buffer, true,
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            VK_ACCESS_TRANSFER_WRITE_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT);
    }
    return;
  }

  const VkImageAspectFlagBits aspects[] = {VK_IMAGE_ASPECT_COLOR_BIT,
                                           VK_IMAGE_ASPECT_DEPTH_BIT,
                                           VK_IMAGE_ASPECT_STENCIL_BIT};
  std::vector<VkBufferImageCopy> copy_regions;
  uint32_t last_mip_level = used_mip_levels_ == VK_REMAINING_MIP_LEVELS
                                ? mip_levels_
                                : base_mip_level_ + used_mip_levels_;
  for (uint32_t i = base_mip_level_; i < last_mip_level; i++) {
    for (auto aspect : aspects) {
      if (aspect_ & aspect) {
        copy_regions.push_back(CreateBufferImageCopy(aspect, i));
      }
    }
  }

  device_->GetPtrs()->vkCmdCopyBufferToImage(
      command_buffer->GetVkCommandBuffer(), host_accessible_buffer_, image_,
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      static_cast<uint32_t>(copy_regions.size()), copy_regions.data());

  MemoryBarrier(command_buffer);
}

void TransferImage::ImageBarrier(CommandBuffer* command_buffer,
                                 VkImageLayout to_layout,
                                 VkPipelineStageFlags to_stage) {
  if (to_layout == layout_ && to_stage == stage_) {
    return;
  }

  VkImageMemoryBarrier barrier = VkImageMemoryBarrier();
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout = layout_;
  barrier.newLayout = to_layout;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image_;
  barrier.subresourceRange = {
      aspect_,                 /* aspectMask */
      0,                       /* baseMipLevel */
      VK_REMAINING_MIP_LEVELS,   /* levelCount */
      0,                         /* baseArrayLayer */
      VK_REMAINING_ARRAY_LAYERS, /* layerCount */
  };

  switch (layout_) {
    case VK_IMAGE_LAYOUT_PREINITIALIZED:
      barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
      break;
    case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
      barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
      break;
    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
      barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
      break;
    case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
      barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      break;
    case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
      barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
      break;
    default:
      barrier.srcAccessMask = 0;
      break;
  }

  switch (to_layout) {
    case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
      barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                              VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
      break;
    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
      barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                              VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
      break;
    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL:
      barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
      break;
    case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
      barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      break;
    case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
      barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
      break;
    case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
      barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      break;
    default:
      barrier.dstAccessMask = 0;
      break;
  }

  device_->GetPtrs()->vkCmdPipelineBarrier(command_buffer->GetVkCommandBuffer(),
                                           stage_, to_stage, 0, 0, NULL, 0,
                                           NULL, 1, &barrier);

  layout_ = to_layout;
  stage_ = to_stage;
}

Result TransferImage::AllocateAndBindMemoryToVkImage(
    VkImage image,
    VkDeviceMemory* memory,
    VkMemoryPropertyFlags flags,
    bool force_flags,
    uint32_t* memory_type_index) {
  if (memory_type_index == nullptr) {
    return Result(
        "Vulkan: TransferImage::AllocateAndBindMemoryToVkImage "
        "memory_type_index is "
        "nullptr");
  }

  *memory_type_index = 0;

  if (image == VK_NULL_HANDLE) {
    return Result("Vulkan::Given VkImage is VK_NULL_HANDLE");
  }
  if (memory == nullptr) {
    return Result("Vulkan::Given VkDeviceMemory pointer is nullptr");
  }

  VkMemoryRequirements requirement;
  device_->GetPtrs()->vkGetImageMemoryRequirements(device_->GetVkDevice(),
                                                   image, &requirement);

  *memory_type_index =
      ChooseMemory(requirement.memoryTypeBits, flags, force_flags);
  if (*memory_type_index == std::numeric_limits<uint32_t>::max()) {
    return Result("Vulkan::Find Proper Memory Fail");
  }

  Result r = AllocateMemory(memory, requirement.size, *memory_type_index);
  if (!r.IsSuccess()) {
    return r;
  }

  if (device_->GetPtrs()->vkBindImageMemory(device_->GetVkDevice(), image,
                                            *memory, 0) != VK_SUCCESS) {
    return Result("Vulkan::Calling vkBindImageMemory Fail");
  }

  return {};
}

}  // namespace vulkan
}  // namespace amber
