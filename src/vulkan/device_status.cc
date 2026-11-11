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

#include <algorithm>
#include <chrono>
#include <string>
#include <thread>

#include "src/vulkan/command_buffer.h"
#include "src/vulkan/device.h"

namespace amber {
namespace vulkan {
namespace {

const std::chrono::microseconds kProbeInterval(1000);

Result WaitIdle(Device* device) {
  VkResult r = device->GetPtrs()->vkQueueWaitIdle(device->GetVkQueue());
  if (r != VK_SUCCESS) {
    return Result("Vulkan::Calling vkQueueWaitIdle Fail (" + VkResultName(r) +
                  ")");
  }
  r = device->GetPtrs()->vkDeviceWaitIdle(device->GetVkDevice());
  if (r != VK_SUCCESS) {
    return Result("Vulkan::Calling vkDeviceWaitIdle Fail (" + VkResultName(r) +
                  ")");
  }
  return {};
}

Result Probe(CommandBuffer* probe, uint32_t fence_timeout_ms) {
  CommandBufferGuard guard(probe);
  if (!guard.IsRecording()) {
    return guard.GetResult();
  }
  return guard.Submit(fence_timeout_ms, false);
}

}  // namespace

Result CheckDeviceStatus(Device* device,
                         CommandBuffer* probe,
                         uint32_t fence_timeout_ms,
                         uint32_t probe_ms) {
  const auto start = std::chrono::steady_clock::now();
  const auto window = std::chrono::milliseconds(probe_ms);

  Result r = WaitIdle(device);
  if (!r.IsSuccess()) {
    return r;
  }

  for (;;) {
    r = Probe(probe, fence_timeout_ms);
    if (!r.IsSuccess()) {
      return r;
    }

    const auto elapsed = std::chrono::steady_clock::now() - start;
    if (elapsed >= window) {
      break;
    }
    std::this_thread::sleep_for(std::min<std::chrono::steady_clock::duration>(
        kProbeInterval, window - elapsed));
  }

  return {};
}

}  // namespace vulkan
}  // namespace amber
