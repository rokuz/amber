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

#ifndef SRC_VULKAN_GRAPHICS_PIPELINE_H_
#define SRC_VULKAN_GRAPHICS_PIPELINE_H_

#include <functional>
#include <memory>
#include <vector>

#include "amber/result.h"
#include "amber/value.h"
#include "amber/vulkan_header.h"
#include "src/format.h"
#include "src/pipeline.h"
#include "src/vulkan/frame_buffer.h"
#include "src/vulkan/index_buffer.h"
#include "src/vulkan/pipeline.h"
#include "src/vulkan/vertex_buffer.h"

namespace amber {

class ProbeCommand;

namespace vulkan {

class CommandPool;

/// Wrapper around a graphics pipeline.
class GraphicsPipeline : public Pipeline {
 public:
  GraphicsPipeline(
      Device* device,
      BlasesMap* blases,
      TlasesMap* tlases,
      const std::vector<amber::Pipeline::BufferInfo>& color_buffers,
      amber::Pipeline::BufferInfo depth_stencil_buffer,
      const std::vector<amber::Pipeline::BufferInfo>& resolve_targets,
      uint32_t fence_timeout_ms,
      bool pipeline_runtime_layer_enabled,
      const std::vector<VkPipelineShaderStageCreateInfo>&);
  ~GraphicsPipeline() override;

  Result Initialize(uint32_t width, uint32_t height, CommandPool* pool);

  Result SetIndexBuffer(Buffer* buffer);

  Result Clear();

  Result SetClearColor(float r, float g, float b, float a);
  Result SetClearStencil(uint32_t stencil);
  Result SetClearDepth(float depth);

  Result Draw(const DrawArraysCommand* command,
              VertexBuffer* vertex_buffer,
              bool is_timed_execution);
  Result DrawMeshTasks(const DrawMeshTasksCommand* command,
                       bool is_timed_execution);

  VkRenderPass GetVkRenderPass() const { return render_pass_; }
  FrameBuffer* GetFrameBuffer() const { return frame_.get(); }

  uint32_t GetWidth() const { return frame_width_; }
  uint32_t GetHeight() const { return frame_height_; }

  BlasesMap* GetBlases() override { return blases_; }
  TlasesMap* GetTlases() override { return tlases_; }

  void SetPatchControlPoints(uint32_t points) {
    patch_control_points_ = points;
  }

  /// Sets the buffers that capture transform feedback in every draw; must be
  /// called before Initialize().
  void SetTransformFeedbackBuffers(
      const std::vector<amber::Pipeline::BufferInfo>& buffers) {
    transform_feedback_buffers_.clear();
    for (const auto& info : buffers) {
      transform_feedback_buffers_.push_back(&info);
    }
  }

  /// Renders |view_count| views with multiview (0 for none); must be called
  /// before Initialize().
  void SetViewCount(uint32_t view_count) { view_count_ = view_count; }

  /// Sets the images the subpass reads as input attachments; must be called
  /// before Initialize().
  void SetInputAttachments(
      const std::vector<amber::Pipeline::BufferInfo>& attachments) {
    input_attachments_.clear();
    input_descriptors_.clear();
    for (const auto& info : attachments) {
      input_descriptors_.push_back(&info);
      bool seen = false;
      for (const auto* known : input_attachments_) {
        seen = seen || known->location == info.location;
      }
      if (!seen) {
        input_attachments_.push_back(&info);
      }
    }
  }

 private:
  Result CreateVkGraphicsPipeline(const PipelineData* pipeline_data,
                                  VkPrimitiveTopology topology,
                                  const VertexBuffer* vertex_buffer,
                                  const VkPipelineLayout& pipeline_layout,
                                  VkPipeline* pipeline);
  Result CreateRenderPass();
  Result RecordDraw(const PipelineData* pipeline_data,
                    VkPrimitiveTopology topology,
                    VertexBuffer* vertex_buffer,
                    bool is_timed_execution,
                    const std::function<Result()>& record);
  Result SendVertexBufferDataIfNeeded(VertexBuffer* vertex_buffer);
  Result AddInputAttachmentDescriptors();

  VkPipelineDepthStencilStateCreateInfo GetVkPipelineDepthStencilInfo(
      const PipelineData* pipeline_data);
  std::vector<VkPipelineColorBlendAttachmentState>
  GetVkPipelineColorBlendAttachmentState(const PipelineData* pipeline_data);

  VkRenderPass render_pass_ = VK_NULL_HANDLE;
  std::unique_ptr<FrameBuffer> frame_;

  // color buffers and resolve targets are owned by the amber::Pipeline.
  std::vector<const amber::Pipeline::BufferInfo*> color_buffers_;
  std::vector<const amber::Pipeline::BufferInfo*> resolve_targets_;
  std::vector<const amber::Pipeline::BufferInfo*> transform_feedback_buffers_;
  std::vector<const amber::Pipeline::BufferInfo*> input_attachments_;
  std::vector<const amber::Pipeline::BufferInfo*> input_descriptors_;
  uint32_t view_count_ = 0;
  amber::Pipeline::BufferInfo depth_stencil_buffer_;
  std::unique_ptr<IndexBuffer> index_buffer_;

  uint32_t frame_width_ = 0;
  uint32_t frame_height_ = 0;

  float clear_color_r_ = 0;
  float clear_color_g_ = 0;
  float clear_color_b_ = 0;
  float clear_color_a_ = 0;
  uint32_t clear_stencil_ = 0;
  float clear_depth_ = 1.0f;
  uint32_t patch_control_points_ = 3;
  BlasesMap* blases_;
  TlasesMap* tlases_;
};

}  // namespace vulkan
}  // namespace amber

#endif  // SRC_VULKAN_GRAPHICS_PIPELINE_H_
