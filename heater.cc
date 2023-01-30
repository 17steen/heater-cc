#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <CL/cl_platform.h>
#include <CL/opencl.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <ranges>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

namespace {
volatile std::atomic<bool> done = false;
}

auto do_gpu() -> bool {
  auto platform = cl::Platform{};
  if (cl::Platform::get(&platform) != CL_SUCCESS) {
    std::cerr << "No platform\n";
    return false;
  }

  std::clog << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

  auto devices = std::vector<cl::Device>{};
  if (platform.getDevices(CL_DEVICE_TYPE_GPU, &devices) != CL_SUCCESS ||
      devices.empty()) {
    std::cerr << "No available GPU'S\n";
    return false;
  }
  for (auto const &device : devices) {
    std::clog << device.getInfo<CL_DEVICE_NAME>() << std::endl;
  }

  auto context = cl::Context(devices.front());

  auto buffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(cl_uint));

  auto queue = cl::CommandQueue{context, devices.front()};

  auto amount = cl_uint{1'000'000};

  queue.enqueueWriteBuffer(buffer, true, 0, sizeof(cl_bool), &amount);
  
  
  auto const source = std::string{
    #include "kernel.cl.h"
  };

  auto program = cl::Program(context, {source});

  if (program.build() != CL_SUCCESS) {
    std::cerr << " Error building: "
              << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices.front())
              << std::endl;

    return false;
  }

  auto errc = int{};

  auto busy = cl::KernelFunctor<cl::Buffer>{program, "busy", &errc};
  if (errc) {
    std::clog << "kernel\n";
    return false;
  }

  auto global = cl::NDRange(1);

  while (not done) {
    auto res = busy(cl::EnqueueArgs{queue, global}, buffer, errc);
    if (errc) {
      std::clog << "run:" << errc << "\n";
      return false;
    }
    res.wait();
  }

  return true;
}

auto main() -> int {

  std::signal(SIGINT, [](int) { done = true; });

  std::puts("Heater ON");
  auto range =
      std::ranges::views::iota(0u, std::jthread::hardware_concurrency()) |
      std::ranges::views::transform([](auto) {
        return std::jthread([] {
          while (not done)
            ;
        });
      });
  auto result = std::vector<std::jthread>();
  std::ranges::move(range, std::back_inserter(result));

  auto gpu = std::jthread{[] {
    if (!do_gpu()) {
      std::puts("Failed to use GPU for heating. Only the CPU will be used.");
    }
  }};
}