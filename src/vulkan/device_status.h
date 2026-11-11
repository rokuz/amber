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

#ifndef SRC_VULKAN_DEVICE_STATUS_H_
#define SRC_VULKAN_DEVICE_STATUS_H_

#include <cstdint>

#include "amber/result.h"

namespace amber {
namespace vulkan {

class CommandBuffer;
class Device;

Result CheckDeviceStatus(Device* device,
                         CommandBuffer* probe,
                         uint32_t fence_timeout_ms,
                         uint32_t probe_ms);

}  // namespace vulkan
}  // namespace amber

#endif  // SRC_VULKAN_DEVICE_STATUS_H_
