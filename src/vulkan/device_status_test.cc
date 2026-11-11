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

#include "src/vulkan/device_status.h"

#include <chrono>
#include <memory>

#include "gtest/gtest.h"
#include "src/vulkan/command_buffer.h"
#include "src/vulkan/command_pool.h"
#include "src/vulkan/device.h"

namespace amber {
namespace vulkan {
namespace {

class StatusDevice : public Device {
 public:
  StatusDevice()
      : Device(VkInstance(),
               VkPhysicalDevice(),
               0u,
               VkDevice(this),
               VkQueue(this),
               nullptr) {
    ptrs_.vkAllocateCommandBuffers = vkAllocateCommandBuffers;
    ptrs_.vkCreateFence = vkCreateFence;
    ptrs_.vkBeginCommandBuffer = vkBeginCommandBuffer;
    ptrs_.vkEndCommandBuffer = vkEndCommandBuffer;
    ptrs_.vkResetCommandBuffer = vkResetCommandBuffer;
    ptrs_.vkResetFences = vkResetFences;
    ptrs_.vkQueueSubmit = vkQueueSubmit;
    ptrs_.vkWaitForFences = vkWaitForFences;
    ptrs_.vkQueueWaitIdle = vkQueueWaitIdle;
    ptrs_.vkDeviceWaitIdle = vkDeviceWaitIdle;
  }
  ~StatusDevice() override {}

  const VulkanPtrs* GetPtrs() const override { return &ptrs_; }

  void LoseOnQueueWaitIdle() { queue_wait_idle_result_ = VK_ERROR_DEVICE_LOST; }
  void LoseOnDeviceWaitIdle() {
    device_wait_idle_result_ = VK_ERROR_DEVICE_LOST;
  }
  void LoseOnSubmit(uint32_t submit) { lost_submit_ = submit; }
  void LoseOnFenceWait(uint32_t submit) { lost_fence_wait_ = submit; }

  uint32_t GetQueueWaitIdleCount() const { return queue_wait_idle_count_; }
  uint32_t GetDeviceWaitIdleCount() const { return device_wait_idle_count_; }
  uint32_t GetSubmitCount() const { return submit_count_; }

 private:
  static StatusDevice* From(VkDevice device) {
    return reinterpret_cast<StatusDevice*>(device);
  }
  static StatusDevice* From(VkQueue queue) {
    return reinterpret_cast<StatusDevice*>(queue);
  }

  static VkResult vkAllocateCommandBuffers(VkDevice,
                                           const VkCommandBufferAllocateInfo*,
                                           VkCommandBuffer*) {
    return VK_SUCCESS;
  }
  static VkResult vkCreateFence(VkDevice,
                                const VkFenceCreateInfo*,
                                const VkAllocationCallbacks*,
                                VkFence*) {
    return VK_SUCCESS;
  }
  static VkResult vkBeginCommandBuffer(VkCommandBuffer,
                                       const VkCommandBufferBeginInfo*) {
    return VK_SUCCESS;
  }
  static VkResult vkEndCommandBuffer(VkCommandBuffer) { return VK_SUCCESS; }
  static VkResult vkResetCommandBuffer(VkCommandBuffer,
                                       VkCommandBufferResetFlags) {
    return VK_SUCCESS;
  }
  static VkResult vkResetFences(VkDevice, uint32_t, const VkFence*) {
    return VK_SUCCESS;
  }
  static VkResult vkQueueSubmit(VkQueue queue,
                                uint32_t,
                                const VkSubmitInfo*,
                                VkFence) {
    StatusDevice* device = From(queue);
    ++device->submit_count_;
    return device->submit_count_ == device->lost_submit_ ? VK_ERROR_DEVICE_LOST
                                                         : VK_SUCCESS;
  }
  static VkResult vkWaitForFences(VkDevice handle,
                                  uint32_t,
                                  const VkFence*,
                                  VkBool32,
                                  uint64_t) {
    StatusDevice* device = From(handle);
    return device->submit_count_ == device->lost_fence_wait_
               ? VK_ERROR_DEVICE_LOST
               : VK_SUCCESS;
  }
  static VkResult vkQueueWaitIdle(VkQueue queue) {
    StatusDevice* device = From(queue);
    ++device->queue_wait_idle_count_;
    return device->queue_wait_idle_result_;
  }
  static VkResult vkDeviceWaitIdle(VkDevice handle) {
    StatusDevice* device = From(handle);
    ++device->device_wait_idle_count_;
    return device->device_wait_idle_result_;
  }

  VulkanPtrs ptrs_;
  VkResult queue_wait_idle_result_ = VK_SUCCESS;
  VkResult device_wait_idle_result_ = VK_SUCCESS;
  uint32_t lost_submit_ = 0;
  uint32_t lost_fence_wait_ = 0;
  uint32_t queue_wait_idle_count_ = 0;
  uint32_t device_wait_idle_count_ = 0;
  uint32_t submit_count_ = 0;
};

class DeviceStatusTest : public testing::Test {
 public:
  DeviceStatusTest()
      : device_(std::make_unique<StatusDevice>()),
        pool_(std::make_unique<CommandPool>(device_.get())),
        probe_(std::make_unique<CommandBuffer>(device_.get(), pool_.get())) {
    probe_->Initialize();
  }

  StatusDevice* GetDevice() { return device_.get(); }

  Result Check(uint32_t probe_ms) {
    return CheckDeviceStatus(device_.get(), probe_.get(), 1000, probe_ms);
  }

 private:
  std::unique_ptr<StatusDevice> device_;
  std::unique_ptr<CommandPool> pool_;
  std::unique_ptr<CommandBuffer> probe_;
};

}  // namespace

TEST_F(DeviceStatusTest, WaitsIdleAndProbesOnce) {
  Result r = Check(0);
  ASSERT_TRUE(r.IsSuccess()) << r.Error();
  EXPECT_EQ(1U, GetDevice()->GetQueueWaitIdleCount());
  EXPECT_EQ(1U, GetDevice()->GetDeviceWaitIdleCount());
  EXPECT_EQ(1U, GetDevice()->GetSubmitCount());
}

TEST_F(DeviceStatusTest, ProbesUntilWindowElapses) {
  const auto start = std::chrono::steady_clock::now();
  Result r = Check(5);
  const auto elapsed = std::chrono::steady_clock::now() - start;
  ASSERT_TRUE(r.IsSuccess()) << r.Error();
  EXPECT_GE(elapsed, std::chrono::milliseconds(5));
  EXPECT_LE(2U, GetDevice()->GetSubmitCount());
}

TEST_F(DeviceStatusTest, ReportsQueueWaitIdleLoss) {
  GetDevice()->LoseOnQueueWaitIdle();
  Result r = Check(1000);
  ASSERT_FALSE(r.IsSuccess());
  EXPECT_EQ("Vulkan::Calling vkQueueWaitIdle Fail (DEVICE_LOST)", r.Error());
  EXPECT_EQ(0U, GetDevice()->GetDeviceWaitIdleCount());
  EXPECT_EQ(0U, GetDevice()->GetSubmitCount());
}

TEST_F(DeviceStatusTest, ReportsDeviceWaitIdleLoss) {
  GetDevice()->LoseOnDeviceWaitIdle();
  Result r = Check(1000);
  ASSERT_FALSE(r.IsSuccess());
  EXPECT_EQ("Vulkan::Calling vkDeviceWaitIdle Fail (DEVICE_LOST)", r.Error());
  EXPECT_EQ(0U, GetDevice()->GetSubmitCount());
}

TEST_F(DeviceStatusTest, ReportsLossFromLaterProbeSubmit) {
  GetDevice()->LoseOnSubmit(3);
  Result r = Check(1000);
  ASSERT_FALSE(r.IsSuccess());
  EXPECT_EQ("Vulkan::Calling vkQueueSubmit Fail (DEVICE_LOST)", r.Error());
  EXPECT_EQ(3U, GetDevice()->GetSubmitCount());
}

TEST_F(DeviceStatusTest, ReportsLossFromLaterProbeFence) {
  GetDevice()->LoseOnFenceWait(2);
  Result r = Check(1000);
  ASSERT_FALSE(r.IsSuccess());
  EXPECT_EQ("Vulkan::Calling vkWaitForFences Fail (DEVICE_LOST)", r.Error());
  EXPECT_EQ(2U, GetDevice()->GetSubmitCount());
}

}  // namespace vulkan
}  // namespace amber
