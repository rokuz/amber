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

#include "amber/amber.h"

#include <stdio.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "amber/recipe.h"
#include "samples/config_helper.h"
#include "samples/ppm.h"
#include "samples/timestamp.h"
#include "src/build-versions.h"

#if AMBER_ENGINE_VULKAN
#include "amber/amber_vulkan.h"
#endif  // AMBER_ENGINE_VULKAN

#if AMBER_ENABLE_SPIRV_TOOLS
#include "spirv-tools/libspirv.hpp"
#endif

#if AMBER_ENABLE_LODEPNG
#include "samples/png.h"
#endif  // AMBER_ENABLE_LODEPNG

namespace {

const char* kGeneratedColorBuffer = "framebuffer";

struct Options {
  std::vector<std::string> input_filenames;

  std::vector<std::string> image_filenames;
  std::string buffer_filename;
  std::vector<std::string> fb_names;
  std::vector<amber::BufferInfo> buffer_to_dump;
  uint32_t engine_major = 1;
  uint32_t engine_minor = 1;
  int32_t fence_timeout = -1;
  int32_t selected_device = -1;
  std::string selected_device_name;
  bool parse_only = false;
  bool pipeline_create_only = false;
  bool disable_validation_layer = false;
  bool quiet = false;
  bool show_help = false;
  bool show_version_info = false;
  bool log_graphics_calls = false;
  bool log_graphics_calls_time = false;
  bool log_execute_calls = false;
  bool log_execution_timing = false;
  bool disable_spirv_validation = false;
  bool enable_pipeline_runtime_layer = false;
  std::string layer_library;
  std::string pipeline_stats_filename;
  bool pipeline_internal_representations = false;
  std::string shader_filename;
  amber::EngineType engine = amber::kEngineTypeVulkan;
  std::string spv_env;
};

const char kUsage[] = R"(Usage: amber [options] SCRIPT [SCRIPTS...]

 options:
  -p                        -- Parse input files only; Don't execute.
  -ps                       -- Parse input files, create pipelines; Don't execute.
  -q                        -- Disable summary output.
  -d                        -- Disable validation layers.
  -D <ID>                   -- ID of device to run with (Vulkan only).
  --device-name <substring> -- Run with the first device whose name contains |substring|,
                               ignoring ASCII case, and that supports the script's
                               requirements (Vulkan only). Lists the available devices if
                               no name matches. Cannot be combined with -D.
  -f <value>                -- Sets the fence timeout value to |value|
  -t <spirv_env>            -- The target SPIR-V environment e.g., spv1.3, vulkan1.1, vulkan1.2.
                               If a SPIR-V environment, assume the lowest version of Vulkan that
                               requires support of that version of SPIR-V.
                               If a Vulkan environment, use the highest version of SPIR-V required
                               to be supported by that version of Vulkan.
                               Use vulkan1.1spv1.4 for SPIR-V 1.4 with Vulkan 1.1.
                               Defaults to spv1.0.
  -i <filename>             -- Write rendering to <filename> as a PNG image if it ends with '.png',
                               or as a PPM image otherwise.
  -I <buffername>           -- Name of framebuffer to dump. Defaults to 'framebuffer'.
  -b <filename>             -- Write contents of a UBO or SSBO to <filename>;
                               {script} in it is replaced by the script's path,
                               so that each script writes a file of its own.
  -B [<pipeline name>:][<desc set>:]<binding> -- Identifier of buffer to write.
                               Default is [first pipeline:][0:]0.
  -w <filename>             -- Write shader assembly to |filename|
  -e <engine>               -- Specify graphics engine: vulkan, dawn. Default is vulkan.
  -v <engine version>       -- Engine version (eg, 1.1 for Vulkan). Default 1.0.
  -V, --version             -- Output version information for Amber and libraries, and the
                               properties of the selected device (honours -D and --device-name).
  --log-graphics-calls      -- Log graphics API calls (only for Vulkan so far).
  --log-graphics-calls-time -- Log timing of graphics API calls timing (Vulkan only).
  --log-execute-calls       -- Log each execute call before run.
  --log-execution-timing    -- Log timing results from each command with the 'TIMED_EXECUTION' flag.
  --disable-spirv-val       -- Disable SPIR-V validation.
  --enable-runtime-layer    -- Enable pipeline runtime layer.
  --layer-library <path>    -- Load a Vulkan layer from the library at |path| and insert it
                               between Amber and the Vulkan loader, without a layer manifest
                               (Vulkan only). For example the ML emulation layer
                               (VkLayer_Tensor) to run VK_ARM_tensors scripts.
  --pipeline-stats <filename> -- Write the statistics of every pipeline created to
                               |filename| as JSON (Vulkan only): the wall time of each
                               pipeline creation, and the driver's statistics of each
                               pipeline executable and shader stage when the device
                               supports VK_KHR_pipeline_executable_properties or
                               VK_AMD_shader_info.
  --pipeline-ir             -- With --pipeline-stats, also write the internal
                               representations of every pipeline executable.
  -h                        -- This help text.
)";

// Parses a decimal integer from the given string, and writes it to |retval|.
// Returns true if parsing succeeded and consumed the whole string.
static bool ParseOneInt(const char* str, int* retval) {
  char trailing = 0;
#if defined(_MSC_VER)
  return sscanf_s(str, "%d%c", retval, &trailing, 1) == 1;
#else
  return std::sscanf(str, "%d%c", retval, &trailing) == 1;
#endif
}

// Parses a decimal integer, then a period (.), then a decimal integer  from the
// given string, and writes it to |retval|.  Returns true if parsing succeeded
// and consumed the whole string.
static int ParseIntDotInt(const char* str, int* retval0, int* retval1) {
  char trailing = 0;
#if defined(_MSC_VER)
  return sscanf_s(str, "%d.%d%c", retval0, retval1, &trailing, 1) == 2;
#else
  return std::sscanf(str, "%d.%d%c", retval0, retval1, &trailing) == 2;
#endif
}

bool ParseArgs(const std::vector<std::string>& args, Options* opts) {
  for (size_t i = 1; i < args.size(); ++i) {
    const std::string& arg = args[i];
    if (arg == "-i") {
      ++i;
      if (i >= args.size()) {
        std::cerr << "Missing value for -i argument." << std::endl;
        return false;
      }
      opts->image_filenames.push_back(args[i]);

    } else if (arg == "-I") {
      ++i;
      if (i >= args.size()) {
        std::cerr << "Missing value for -I argument." << std::endl;
        return false;
      }
      opts->fb_names.push_back(args[i]);

    } else if (arg == "-b") {
      ++i;
      if (i >= args.size()) {
        std::cerr << "Missing value for -b argument." << std::endl;
        return false;
      }
      opts->buffer_filename = args[i];

    } else if (arg == "-B") {
      ++i;
      if (i >= args.size()) {
        std::cerr << "Missing value for -B argument." << std::endl;
        return false;
      }
      opts->buffer_to_dump.emplace_back();
      opts->buffer_to_dump.back().buffer_name = args[i];
    } else if (arg == "-w") {
      ++i;
      if (i >= args.size()) {
        std::cerr << "Missing value for -w argument." << std::endl;
        return false;
      }
      opts->shader_filename = args[i];
    } else if (arg == "-e") {
      ++i;
      if (i >= args.size()) {
        std::cerr << "Missing value for -e argument." << std::endl;
        return false;
      }
      const std::string& engine = args[i];
      if (engine == "vulkan") {
        opts->engine = amber::kEngineTypeVulkan;
      } else if (engine == "dawn") {
        opts->engine = amber::kEngineTypeDawn;
      } else {
        std::cerr
            << "Invalid value for -e argument. Must be one of: vulkan dawn"
            << std::endl;
        return false;
      }
    } else if (arg == "-D") {
      ++i;
      if (i >= args.size()) {
        std::cerr << "Missing ID for -D argument." << std::endl;
        return false;
      }

      int32_t val = 0;
      if (!ParseOneInt(args[i].c_str(), &val)) {
        std::cerr << "Invalid device ID: " << args[i] << std::endl;
        return false;
      }
      if (val < 0) {
        std::cerr << "Device ID must be non-negative" << std::endl;
        return false;
      }
      opts->selected_device = val;

    } else if (arg == "--device-name") {
      ++i;
      if (i >= args.size()) {
        std::cerr << "Missing value for --device-name argument." << std::endl;
        return false;
      }
      if (args[i].empty()) {
        std::cerr << "Device name must be non-empty" << std::endl;
        return false;
      }
      opts->selected_device_name = args[i];

    } else if (arg == "-f") {
      ++i;
      if (i >= args.size()) {
        std::cerr << "Missing value for -f argument." << std::endl;
        return false;
      }

      int32_t val = 0;
      if (!ParseOneInt(args[i].c_str(), &val)) {
        std::cerr << "Invalid fence timeout: " << args[i] << std::endl;
        return false;
      }
      if (val < 0) {
        std::cerr << "Fence timeout must be non-negative" << std::endl;
        return false;
      }
      opts->fence_timeout = val;

    } else if (arg == "-t") {
      ++i;
      if (i >= args.size()) {
        std::cerr << "Missing value for -t argument." << std::endl;
        return false;
      }
      opts->spv_env = args[i];
    } else if (arg == "-h" || arg == "--help") {
      opts->show_help = true;
    } else if (arg == "-v") {
      ++i;
      if (i >= args.size()) {
        std::cerr << "Missing value for -v argument." << std::endl;
        return false;
      }
      const std::string& ver = std::string(args[i]);

      int32_t major = 0;
      int32_t minor = 0;
      if (ParseIntDotInt(ver.c_str(), &major, &minor) ||
          ParseOneInt(ver.c_str(), &major)) {
        if (major < 0) {
          std::cerr << "Version major must be non-negative" << std::endl;
          return false;
        }
        if (minor < 0) {
          std::cerr << "Version minor must be non-negative" << std::endl;
          return false;
        }
        opts->engine_major = static_cast<uint32_t>(major);
        opts->engine_minor = static_cast<uint32_t>(minor);
      } else {
        std::cerr << "Invalid engine version number: " << ver << std::endl;
        return false;
      }
    } else if (arg == "-V" || arg == "--version") {
      opts->show_version_info = true;
    } else if (arg == "-p") {
      opts->parse_only = true;
    } else if (arg == "-ps") {
      opts->pipeline_create_only = true;
    } else if (arg == "-d") {
      opts->disable_validation_layer = true;
    } else if (arg == "-s") {
      // -s is deprecated but still recognized, it inverts the quiet flag.
      opts->quiet = false;
    } else if (arg == "-q") {
      opts->quiet = true;
    } else if (arg == "--log-graphics-calls") {
      opts->log_graphics_calls = true;
    } else if (arg == "--log-graphics-calls-time") {
      opts->log_graphics_calls_time = true;
    } else if (arg == "--log-execution-timing") {
      opts->log_execution_timing = true;
    } else if (arg == "--log-execute-calls") {
      opts->log_execute_calls = true;
    } else if (arg == "--disable-spirv-val") {
      opts->disable_spirv_validation = true;
    } else if (arg == "--enable-runtime-layer") {
      opts->enable_pipeline_runtime_layer = true;
    } else if (arg == "--layer-library") {
      ++i;
      if (i >= args.size() || args[i].empty()) {
        std::cerr << "Missing value for --layer-library argument." << std::endl;
        return false;
      }
      opts->layer_library = args[i];
    } else if (arg == "--pipeline-stats") {
      ++i;
      if (i >= args.size() || args[i].empty()) {
        std::cerr << "Missing value for --pipeline-stats argument."
                  << std::endl;
        return false;
      }
      opts->pipeline_stats_filename = args[i];
    } else if (arg == "--pipeline-ir") {
      opts->pipeline_internal_representations = true;
    } else if (arg.size() > 0 && arg[0] == '-') {
      std::cerr << "Unrecognized option " << arg << std::endl;
      return false;
    } else if (!arg.empty()) {
      opts->input_filenames.push_back(arg);
    }
  }

  if (opts->pipeline_internal_representations &&
      opts->pipeline_stats_filename.empty()) {
    std::cerr << "--pipeline-ir needs --pipeline-stats." << std::endl;
    return false;
  }

  if (opts->selected_device > -1 && !opts->selected_device_name.empty()) {
    std::cerr << "-D and --device-name cannot be combined." << std::endl;
    return false;
  }

  return true;
}

std::vector<char> ReadFile(const std::string& input_file) {
  FILE* file = nullptr;
#if defined(_MSC_VER)
  fopen_s(&file, input_file.c_str(), "rb");
#else
  file = fopen(input_file.c_str(), "rb");
#endif
  if (!file) {
    std::cerr << "Failed to open " << input_file << std::endl;
    return {};
  }

  fseek(file, 0, SEEK_END);
  uint64_t tell_file_size = static_cast<uint64_t>(ftell(file));
  if (tell_file_size <= 0) {
    std::cerr << "Input file of incorrect size: " << input_file << std::endl;
    fclose(file);
    return {};
  }
  fseek(file, 0, SEEK_SET);

  size_t file_size = static_cast<size_t>(tell_file_size);

  std::vector<char> data;
  data.resize(file_size);

  size_t bytes_read = fread(data.data(), sizeof(char), file_size, file);
  fclose(file);
  if (bytes_read != file_size) {
    std::cerr << "Failed to read " << input_file << std::endl;
    return {};
  }

  return data;
}

class SampleDelegate : public amber::Delegate {
 public:
  SampleDelegate() = default;
  ~SampleDelegate() override = default;

  void Log(const std::string& message) override {
    std::cout << message << std::endl;
  }

  bool LogGraphicsCalls() const override { return log_graphics_calls_; }
  void SetLogGraphicsCalls(bool log_graphics_calls) {
    log_graphics_calls_ = log_graphics_calls;
  }

  bool LogExecuteCalls() const override { return log_execute_calls_; }
  void SetLogExecuteCalls(bool log_execute_calls) {
    log_execute_calls_ = log_execute_calls;
  }

  bool LogGraphicsCallsTime() const override {
    return log_graphics_calls_time_;
  }
  void SetLogGraphicsCallsTime(bool log_graphics_calls_time) {
    log_graphics_calls_time_ = log_graphics_calls_time;
    if (log_graphics_calls_time) {
      // Make sure regular logging is also enabled
      log_graphics_calls_ = true;
    }
  }

  void ReportExecutionTiming(double time_in_ms) override {
    reported_execution_timing.push_back(time_in_ms);
  }

  std::vector<double> GetAndClearExecutionTiming() {
    auto returning = reported_execution_timing;
    reported_execution_timing.clear();
    return returning;
  }

  bool CapturePipelineStatistics() const override {
    return capture_pipeline_statistics_;
  }
  bool CapturePipelineInternalRepresentations() const override {
    return capture_pipeline_internal_representations_;
  }
  void SetCapturePipelineStatistics(bool capture,
                                    bool internal_representations) {
    capture_pipeline_statistics_ = capture;
    capture_pipeline_internal_representations_ = internal_representations;
  }

  void ReportPipelineStatistics(
      const amber::PipelineStatistics& statistics) override {
    reported_pipeline_statistics_.push_back(statistics);
  }

  std::vector<amber::PipelineStatistics> GetAndClearPipelineStatistics() {
    auto returning = reported_pipeline_statistics_;
    reported_pipeline_statistics_.clear();
    return returning;
  }

  uint64_t GetTimestampNs() const override {
    return timestamp::SampleGetTimestampNs();
  }

  void SetScriptPath(std::string path) { path_ = path; }

  amber::Result LoadBufferData(const std::string file_name,
                               amber::BufferDataFileType file_type,
                               amber::BufferInfo* buffer) const override {
    if (file_type == amber::BufferDataFileType::kPng) {
#if AMBER_ENABLE_LODEPNG
      return png::LoadPNG(path_ + file_name, &buffer->width, &buffer->height,
                          &buffer->values);
#else
      return amber::Result("PNG support is not enabled in compile options.");
#endif  // AMBER_ENABLE_LODEPNG
    } else {
      auto data = ReadFile(path_ + file_name);
      if (data.empty()) {
        return amber::Result("Failed to load buffer data " + file_name);
      }

      for (auto d : data) {
        amber::Value v;
        v.SetIntValue(static_cast<uint64_t>(d));
        buffer->values.push_back(v);
      }

      buffer->width = 1;
      buffer->height = 1;
    }

    return {};
  }

  amber::Result LoadFile(const std::string file_name,
                         std::vector<char>* buffer) const override {
    *buffer = ReadFile(path_ + file_name);
    if (buffer->empty()) {
      return amber::Result("Failed to load file " + file_name);
    }
    return {};
  }

 private:
  bool log_graphics_calls_ = false;
  bool log_graphics_calls_time_ = false;
  bool log_execute_calls_ = false;
  std::string path_ = "";
  std::vector<double> reported_execution_timing;
  bool capture_pipeline_statistics_ = false;
  bool capture_pipeline_internal_representations_ = false;
  std::vector<amber::PipelineStatistics> reported_pipeline_statistics_;
};

std::string JsonString(const std::string& text) {
  std::ostringstream out;
  out << '"';
  for (const char c : text) {
    switch (c) {
      case '"':
        out << "\\\"";
        break;
      case '\\':
        out << "\\\\";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<int>(c) << std::dec << std::setfill(' ');
        } else {
          out << c;
        }
        break;
    }
  }
  out << '"';
  return out.str();
}

std::string JsonHex(const std::string& data) {
  std::ostringstream out;
  out << '"' << std::hex << std::setfill('0');
  for (const char c : data) {
    out << std::setw(2) << static_cast<int>(static_cast<unsigned char>(c));
  }
  out << '"';
  return out.str();
}

std::string JsonStatisticValue(const amber::PipelineStatistic& statistic) {
  if (statistic.format == "bool32" &&
      (statistic.value == "true" || statistic.value == "false")) {
    return statistic.value;
  }
  const bool number = statistic.format == "int64" ||
                      statistic.format == "uint64" ||
                      statistic.format == "float64";
  if (number && !statistic.value.empty() &&
      statistic.value.find_first_not_of("0123456789+-.eE") ==
          std::string::npos) {
    return statistic.value;
  }
  return JsonString(statistic.value);
}

struct ScriptPipelineStatistics {
  std::string script;
  amber::PipelineStatistics statistics;
};

bool WritePipelineStatistics(
    const std::string& filename,
    bool pipeline_executable_properties,
    bool shader_info_amd,
    const std::vector<ScriptPipelineStatistics>& pipelines) {
  std::ofstream out(filename, std::ios::out | std::ios::trunc);
  if (!out.is_open()) {
    return false;
  }
  out << std::setprecision(17);
  out << "{\n  \"pipeline_executable_properties\": "
      << (pipeline_executable_properties ? "true" : "false")
      << ",\n  \"shader_info_amd\": " << (shader_info_amd ? "true" : "false")
      << ",\n  \"pipelines\": [";
  for (size_t p = 0; p < pipelines.size(); ++p) {
    const auto& pipeline = pipelines[p].statistics;
    out << (p == 0 ? "\n" : ",\n")
        << "    {\"script\": " << JsonString(pipelines[p].script)
        << ", \"pipeline\": " << JsonString(pipeline.pipeline)
        << ", \"type\": " << JsonString(pipeline.type)
        << ", \"create_time_ms\": " << pipeline.create_time_ms
        << ", \"executables\": [";
    for (size_t e = 0; e < pipeline.executables.size(); ++e) {
      const auto& executable = pipeline.executables[e];
      out << (e == 0 ? "\n" : ",\n")
          << "      {\"source\": " << JsonString(executable.source)
          << ", \"name\": " << JsonString(executable.name)
          << ", \"description\": " << JsonString(executable.description)
          << ", \"stages\": [";
      for (size_t s = 0; s < executable.stages.size(); ++s) {
        out << (s == 0 ? "" : ", ") << JsonString(executable.stages[s]);
      }
      out << "], \"subgroup_size\": " << executable.subgroup_size
          << ", \"statistics\": [";
      for (size_t s = 0; s < executable.statistics.size(); ++s) {
        const auto& statistic = executable.statistics[s];
        out << (s == 0 ? "\n" : ",\n")
            << "        {\"name\": " << JsonString(statistic.name)
            << ", \"description\": " << JsonString(statistic.description)
            << ", \"format\": " << JsonString(statistic.format)
            << ", \"value\": " << JsonStatisticValue(statistic) << "}";
      }
      out << (executable.statistics.empty() ? "]" : "\n      ]")
          << ", \"internal_representations\": [";
      for (size_t r = 0; r < executable.internal_representations.size(); ++r) {
        const auto& representation = executable.internal_representations[r];
        out << (r == 0 ? "\n" : ",\n")
            << "        {\"name\": " << JsonString(representation.name)
            << ", \"description\": " << JsonString(representation.description)
            << ", \"is_text\": " << (representation.is_text ? "true" : "false")
            << ", \"data\": "
            << (representation.is_text ? JsonString(representation.data)
                                       : JsonHex(representation.data))
            << "}";
      }
      out << (executable.internal_representations.empty() ? "]}"
                                                          : "\n      ]}");
    }
    out << (pipeline.executables.empty() ? "]}" : "\n    ]}");
  }
  out << (pipelines.empty() ? "]\n}\n" : "\n  ]\n}\n");
  return out.good();
}

std::string disassemble(const std::string& env,
                        const std::vector<uint32_t>& data) {
#if AMBER_ENABLE_SPIRV_TOOLS
  std::string spv_errors;

  spv_target_env target_env = SPV_ENV_UNIVERSAL_1_0;
  if (!env.empty()) {
    if (!spvParseTargetEnv(env.c_str(), &target_env)) {
      return "";
    }
  }

  auto msg_consumer = [&spv_errors](spv_message_level_t level, const char*,
                                    const spv_position_t& position,
                                    const char* message) {
    switch (level) {
      case SPV_MSG_FATAL:
      case SPV_MSG_INTERNAL_ERROR:
      case SPV_MSG_ERROR:
        spv_errors += "error: line " + std::to_string(position.index) + ": " +
                      message + "\n";
        break;
      case SPV_MSG_WARNING:
        spv_errors += "warning: line " + std::to_string(position.index) + ": " +
                      message + "\n";
        break;
      case SPV_MSG_INFO:
        spv_errors += "info: line " + std::to_string(position.index) + ": " +
                      message + "\n";
        break;
      case SPV_MSG_DEBUG:
        break;
    }
  };

  spvtools::SpirvTools tools(target_env);
  tools.SetMessageConsumer(msg_consumer);

  std::string result;
  tools.Disassemble(data, &result,
                    SPV_BINARY_TO_TEXT_OPTION_INDENT |
                        SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES);
  return result;
#else
  return "";
#endif  // AMBER_ENABLE_SPIRV_TOOLS
}

}  // namespace

#ifdef AMBER_ANDROID_MAIN
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma ide diagnostic ignored "OCUnusedGlobalDeclarationInspection"
int android_main(int argc, const char** argv) {
#pragma clang diagnostic pop
#else
int main(int argc, const char** argv) {
#endif
  std::vector<std::string> args(argv, argv + argc);
  Options options;
  SampleDelegate delegate;

  if (!ParseArgs(args, &options)) {
    std::cerr << "Failed to parse arguments." << std::endl;
    return 1;
  }

  if (options.show_version_info) {
    std::cout << "Amber        : " << AMBER_VERSION << std::endl;
#if AMBER_ENABLE_SPIRV_TOOLS
    std::cout << "SPIRV-Tools  : " << SPIRV_TOOLS_VERSION << std::endl;
    std::cout << "SPIRV-Headers: " << SPIRV_HEADERS_VERSION << std::endl;
#endif  // AMBER_ENABLE_SPIRV_TOOLS
#if AMBER_ENABLE_SHADERC
    std::cout << "GLSLang      : " << GLSLANG_VERSION << std::endl;
    std::cout << "Shaderc      : " << SHADERC_VERSION << std::endl;
#endif  // AMBER_ENABLE_SHADERC
  }

  if (options.show_help) {
    std::cout << kUsage << std::endl;
    return 0;
  }

  amber::Result result;
  std::vector<std::string> failures;
  struct RecipeData {
    std::string file;
    std::unique_ptr<amber::Recipe> recipe;
  };
  std::vector<RecipeData> recipe_data;
  for (const auto& file : options.input_filenames) {
    auto char_data = ReadFile(file);
    auto data = std::string(char_data.begin(), char_data.end());
    if (data.empty()) {
      std::cerr << file << " is empty." << std::endl;
      failures.push_back(file);
      continue;
    }

    // Parse file path and set it for delegate to use when loading buffer data.
    delegate.SetScriptPath(file.substr(0, file.find_last_of("/\\") + 1));

    amber::Amber am(&delegate);
    std::unique_ptr<amber::Recipe> recipe = std::make_unique<amber::Recipe>();

    result = am.Parse(data, recipe.get());
    if (!result.IsSuccess()) {
      std::cerr << file << ": " << result.Error() << std::endl;
      failures.push_back(file);
      continue;
    }

    if (options.fence_timeout > -1) {
      recipe->SetFenceTimeout(static_cast<uint32_t>(options.fence_timeout));
    }

    recipe->SetPipelineRuntimeLayerEnabled(
        options.enable_pipeline_runtime_layer);

    recipe_data.emplace_back();
    recipe_data.back().file = file;
    recipe_data.back().recipe = std::move(recipe);
  }

  if (options.parse_only) {
    return 0;
  }

  if (options.log_graphics_calls) {
    delegate.SetLogGraphicsCalls(true);
  }
  if (options.log_graphics_calls_time) {
    delegate.SetLogGraphicsCallsTime(true);
  }
  if (options.log_execute_calls) {
    delegate.SetLogExecuteCalls(true);
  }
  delegate.SetCapturePipelineStatistics(
      !options.pipeline_stats_filename.empty(),
      options.pipeline_internal_representations);

  amber::Options amber_options;
  amber_options.engine = options.engine;
  amber_options.spv_env = options.spv_env;
  amber_options.execution_type = options.pipeline_create_only
                                     ? amber::ExecutionType::kPipelineCreateOnly
                                     : amber::ExecutionType::kExecute;
  amber_options.disable_spirv_validation = options.disable_spirv_validation;

  std::set<std::string> required_features;
  std::set<std::string> required_device_extensions;
  std::set<std::string> required_instance_extensions;
  for (const auto& recipe_data_elem : recipe_data) {
    const auto features = recipe_data_elem.recipe->GetRequiredFeatures();
    required_features.insert(features.begin(), features.end());

    const auto device_extensions =
        recipe_data_elem.recipe->GetRequiredDeviceExtensions();
    required_device_extensions.insert(device_extensions.begin(),
                                      device_extensions.end());

    const auto inst_extensions =
        recipe_data_elem.recipe->GetRequiredInstanceExtensions();
    required_instance_extensions.insert(inst_extensions.begin(),
                                        inst_extensions.end());
  }

  sample::ConfigHelper config_helper;
  std::unique_ptr<amber::EngineConfig> config;

  amber::Result r = config_helper.CreateConfig(
      amber_options.engine, options.engine_major, options.engine_minor,
      options.selected_device, options.selected_device_name,
      std::vector<std::string>(required_features.begin(),
                               required_features.end()),
      std::vector<std::string>(required_instance_extensions.begin(),
                               required_instance_extensions.end()),
      std::vector<std::string>(required_device_extensions.begin(),
                               required_device_extensions.end()),
      options.disable_validation_layer, options.enable_pipeline_runtime_layer,
      options.layer_library, options.show_version_info,
      !options.pipeline_stats_filename.empty(), &config);

  if (!r.IsSuccess()) {
    std::cout << r.Error() << std::endl;
    return 1;
  }

  amber_options.config = config.get();

  bool pipeline_executable_properties = false;
  bool shader_info_amd = false;
#if AMBER_ENGINE_VULKAN
  if (amber_options.engine == amber::kEngineTypeVulkan) {
    const auto* vulkan_config =
        static_cast<const amber::VulkanEngineConfig*>(config.get());
    pipeline_executable_properties = vulkan_config->pipeline_executable_info;
    shader_info_amd = vulkan_config->shader_info_amd;
  }
#endif  // AMBER_ENGINE_VULKAN
  std::vector<ScriptPipelineStatistics> pipeline_statistics;

  if (!options.buffer_filename.empty()) {
    // Have a filename to dump, but no explicit buffer, set the default of 0:0.
    if (options.buffer_to_dump.empty()) {
      options.buffer_to_dump.emplace_back();
      options.buffer_to_dump.back().buffer_name = "0:0";
    }

    amber_options.extractions.insert(amber_options.extractions.end(),
                                     options.buffer_to_dump.begin(),
                                     options.buffer_to_dump.end());
  }

  if (options.image_filenames.size() - options.fb_names.size() > 1) {
    std::cerr << "Need to specify framebuffer names using -I for each output "
                 "image specified by -i."
              << std::endl;
    return 1;
  }

  // Use default frame buffer name when not specified.
  while (options.image_filenames.size() > options.fb_names.size()) {
    options.fb_names.push_back(kGeneratedColorBuffer);
  }

  for (const auto& fb_name : options.fb_names) {
    amber::BufferInfo buffer_info;
    buffer_info.buffer_name = fb_name;
    buffer_info.is_image_buffer = true;
    amber_options.extractions.push_back(buffer_info);
  }

  for (const auto& recipe_data_elem : recipe_data) {
    const auto* recipe = recipe_data_elem.recipe.get();
    const auto& file = recipe_data_elem.file;

    amber::Amber am(&delegate);
    result = am.Execute(recipe, &amber_options);
    if (!result.IsSuccess()) {
      std::cerr << file << ": " << result.Error() << "\n";
      failures.push_back(file);
      // Note, we continue after failure to allow dumping the buffers which may
      // give clues as to the failure.
    }

    for (auto& statistics : delegate.GetAndClearPipelineStatistics()) {
      pipeline_statistics.push_back({file, statistics});
    }

    auto execution_timing = delegate.GetAndClearExecutionTiming();
    if (result.IsSuccess() && options.log_execution_timing &&
        !execution_timing.empty()) {
      std::cout << "Execution timing (in script-order): " << file << "\n";
      std::cout << "    ";
      bool is_first_iter = true;
      for (auto& timing : execution_timing) {
        if (!is_first_iter) {
          std::cout << ", ";
        }
        is_first_iter = false;
        std::cout << timing;
      }
      std::cout << "\n";
      std::sort(execution_timing.begin(), execution_timing.end());
      auto report_median =
          (execution_timing[execution_timing.size() / 2] +
           execution_timing[(execution_timing.size() - 1) / 2]) /
          2;
      std::cout << "\n";
      std::cout << "Execution time median = " << report_median << " ms" << "\n";
    }

    // Dump the shader assembly
    if (!options.shader_filename.empty()) {
#if AMBER_ENABLE_SPIRV_TOOLS
      std::ofstream shader_file;
      shader_file.open(options.shader_filename, std::ios::out);
      if (!shader_file.is_open()) {
        std::cerr << "Cannot open file for shader dump: ";
        std::cerr << options.shader_filename << std::endl;
      } else {
        auto info = recipe->GetShaderInfo();
        for (const auto& sh : info) {
          shader_file << ";;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;"
                      << std::endl;
          shader_file << "; " << sh.shader_name << std::endl
                      << ";" << std::endl;
          shader_file << disassemble(options.spv_env, sh.shader_data)
                      << std::endl;
        }
        shader_file.close();
      }
#endif  // AMBER_ENABLE_SPIRV_TOOLS
    }

    for (size_t i = 0; i < options.image_filenames.size(); ++i) {
      std::vector<uint8_t> out_buf;
      auto image_filename = options.image_filenames[i];
      auto pos = image_filename.find_last_of('.');
      bool usePNG =
          pos != std::string::npos && image_filename.substr(pos + 1) == "png";
      for (const amber::BufferInfo& buffer_info : amber_options.extractions) {
        if (buffer_info.buffer_name == options.fb_names[i]) {
          if (buffer_info.values.size() !=
              (buffer_info.width * buffer_info.height)) {
            result = amber::Result(
                "Framebuffer (" + buffer_info.buffer_name + ") size (" +
                std::to_string(buffer_info.values.size()) +
                ") != " + "width * height (" +
                std::to_string(buffer_info.width * buffer_info.height) + ")");
            break;
          }

          if (buffer_info.values.empty()) {
            result = amber::Result("Framebuffer (" + buffer_info.buffer_name +
                                   ") empty or non-existent.");
            break;
          }

          if (usePNG) {
#if AMBER_ENABLE_LODEPNG
            result = png::ConvertToPNG(buffer_info.width, buffer_info.height,
                                       buffer_info.values, &out_buf);
#else   // AMBER_ENABLE_LODEPNG
            result = amber::Result("PNG support not enabled");
#endif  // AMBER_ENABLE_LODEPNG
          } else {
            ppm::ConvertToPPM(buffer_info.width, buffer_info.height,
                              buffer_info.values, &out_buf);
            result = {};
          }
          break;
        }
      }
      if (result.IsSuccess()) {
        std::ofstream image_file;
        image_file.open(image_filename, std::ios::out | std::ios::binary);
        if (!image_file.is_open()) {
          std::cerr << "Cannot open file for image dump: ";
          std::cerr << image_filename << std::endl;
          continue;
        }
        image_file << std::string(out_buf.begin(), out_buf.end());
        image_file.close();
      } else {
        std::cerr << result.Error() << std::endl;
      }
    }

    if (!options.buffer_filename.empty()) {
      std::string buffer_filename = options.buffer_filename;
      const std::string script_token = "{script}";
      const auto token = buffer_filename.find(script_token);
      if (token != std::string::npos) {
        buffer_filename.replace(token, script_token.size(), file);
      }
      std::ofstream buffer_file;
      buffer_file.open(buffer_filename, std::ios::out);
      if (!buffer_file.is_open()) {
        std::cerr << "Cannot open file for buffer dump: ";
        std::cerr << buffer_filename << std::endl;
      } else {
        for (const amber::BufferInfo& buffer_info : amber_options.extractions) {
          // Skip frame buffers.
          if (std::any_of(options.fb_names.begin(), options.fb_names.end(),
                          [&](std::string s) {
                            return s == buffer_info.buffer_name;
                          }) ||
              buffer_info.buffer_name == kGeneratedColorBuffer) {
            continue;
          }

          buffer_file << buffer_info.buffer_name << std::endl;
          const auto& values = buffer_info.values;
          for (size_t i = 0; i < values.size(); ++i) {
            buffer_file << " " << std::setfill('0') << std::setw(2) << std::hex
                        << values[i].AsUint32();
            if (i % 16 == 15) {
              buffer_file << std::endl;
            }
          }
          buffer_file << std::endl;
        }
        buffer_file.close();
      }
    }
  }

  if (!options.pipeline_stats_filename.empty() &&
      !WritePipelineStatistics(options.pipeline_stats_filename,
                               pipeline_executable_properties, shader_info_amd,
                               pipeline_statistics)) {
    std::cerr << "Cannot write the pipeline statistics to "
              << options.pipeline_stats_filename << std::endl;
    failures.push_back(options.pipeline_stats_filename);
  }

  if (!options.quiet) {
    if (!failures.empty()) {
      std::cout << "\nSummary of Failures:" << std::endl;

      for (const auto& failure : failures) {
        std::cout << "  " << failure << std::endl;
      }
    }

    std::cout << "\nSummary: "
              << (options.input_filenames.size() - failures.size()) << " pass, "
              << failures.size() << " fail" << std::endl;
  }

  return !failures.empty();
}
