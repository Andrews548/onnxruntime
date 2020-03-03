// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2020, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#include "armnn_execution_provider.h"
#include "core/framework/allocator.h"
#include "core/framework/op_kernel.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/compute_capability.h"
#include "contrib_ops/cpu_contrib_kernels.h"
#include "armnn_fwd.h"

namespace onnxruntime {

constexpr const char* ARMNN = "Armnn";
constexpr const char* ARMNN_CPU = "ArmnnCpu";

namespace armnn_ep {

// Forward declarations of op kernels
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kArmnnExecutionProvider, kOnnxDomain, 6, Relu);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kArmnnExecutionProvider, kOnnxDomain, 1, Conv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kArmnnExecutionProvider, kMSDomain, 1, float, FusedConv);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kArmnnExecutionProvider, kOnnxDomain, 7, 9, Gemm);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kArmnnExecutionProvider, kOnnxDomain, 7, 9, float, AveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kArmnnExecutionProvider, kOnnxDomain, 1, 7, float, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kArmnnExecutionProvider, kOnnxDomain, 8, 9, float, MaxPool);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kArmnnExecutionProvider, kOnnxDomain, 1, 8, float, GlobalAveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kArmnnExecutionProvider, kOnnxDomain, 1, 8, float, GlobalMaxPool);

// Opset 10
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kArmnnExecutionProvider, kOnnxDomain, 10, 10, float, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kArmnnExecutionProvider, kOnnxDomain, 10, 10, float, AveragePool);

static void RegisterARMNNKernels(KernelRegistry& kernel_registry) {

  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kArmnnExecutionProvider, kOnnxDomain, 6, Relu)>());
  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kArmnnExecutionProvider, kOnnxDomain, 1, Conv)>());
  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kArmnnExecutionProvider, kMSDomain, 1, float, FusedConv)>());

  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kArmnnExecutionProvider, kOnnxDomain, 7, 9, Gemm)>());

  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kArmnnExecutionProvider, kOnnxDomain, 7, 9, float, AveragePool)>());
  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kArmnnExecutionProvider, kOnnxDomain, 1, 7, float, MaxPool)>());
  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kArmnnExecutionProvider, kOnnxDomain, 8, 9, float, MaxPool)>());

  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kArmnnExecutionProvider, kOnnxDomain, 1, 8, float, GlobalAveragePool)>());
  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kArmnnExecutionProvider, kOnnxDomain, 1, 8, float, GlobalMaxPool)>());

  // Opset 10
  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kArmnnExecutionProvider, kOnnxDomain, 10, 10, float, MaxPool)>());
  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kArmnnExecutionProvider, kOnnxDomain, 10, 10, float, AveragePool)>());
}

std::shared_ptr<KernelRegistry> GetArmnnKernelRegistry() {
  std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>();
  RegisterARMNNKernels(*kernel_registry);

// #ifndef DISABLE_CONTRIB_OPS
//   ::onnxruntime::contrib::armnn_ep::RegisterARMNNContribKernels(*kernel_registry);
// #endif

  return kernel_registry;
}

}  // namespace armnn_ep

ARMNNExecutionProvider::ARMNNExecutionProvider(const ARMNNExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kArmnnExecutionProvider} {
  ORT_UNUSED_PARAMETER(info);

  auto default_allocator_factory = [](int) {
    auto memory_info = onnxruntime::make_unique<OrtMemoryInfo>(ARMNN, OrtAllocatorType::OrtDeviceAllocator);
    return onnxruntime::make_unique<CPUAllocator>(std::move(memory_info));
  };

  DeviceAllocatorRegistrationInfo default_memory_info{
      OrtMemTypeDefault,
      std::move(default_allocator_factory),
      std::numeric_limits<size_t>::max()};

  InsertAllocator(CreateAllocator(default_memory_info));

  auto cpu_allocator_factory = [](int) {
    auto memory_info = onnxruntime::make_unique<OrtMemoryInfo>(
        ARMNN_CPU, OrtAllocatorType::OrtDeviceAllocator, OrtDevice(), 0, OrtMemTypeCPUOutput);
    return onnxruntime::make_unique<CPUAllocator>(std::move(memory_info));
  };

  DeviceAllocatorRegistrationInfo cpu_memory_info{
      OrtMemTypeCPUOutput,
      std::move(cpu_allocator_factory),
      std::numeric_limits<size_t>::max()};

  InsertAllocator(CreateAllocator(cpu_memory_info));
}

ARMNNExecutionProvider::~ARMNNExecutionProvider() {
}

std::shared_ptr<KernelRegistry> ARMNNExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> kernel_registry = onnxruntime::armnn_ep::GetArmnnKernelRegistry();
  return kernel_registry;
}

std::vector<std::unique_ptr<ComputeCapability>>
ARMNNExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph,
                                    const std::vector<const KernelRegistry*>& kernel_registries) const {
  std::vector<std::unique_ptr<ComputeCapability>>
      result = IExecutionProvider::GetCapability(graph, kernel_registries);

  return result;
}

}  // namespace onnxruntime
