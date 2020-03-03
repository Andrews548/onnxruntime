// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2020, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/armnn/armnn_provider_factory.h"
#include <atomic>
#include "armnn_execution_provider.h"
#include "core/session/abi_session_options_impl.h"

namespace onnxruntime {

struct ARMNNProviderFactory : IExecutionProviderFactory {
  ARMNNProviderFactory(bool create_arena) : create_arena_(create_arena) {}
  ~ARMNNProviderFactory() override {}
  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  bool create_arena_;
};

std::unique_ptr<IExecutionProvider> ARMNNProviderFactory::CreateProvider() {
  ARMNNExecutionProviderInfo info;
  info.create_arena = create_arena_;
  return std::make_unique<ARMNNExecutionProvider>(info);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_ARMNN(int use_arena) {
  return std::make_shared<onnxruntime::ARMNNProviderFactory>(use_arena != 0);
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_ARMNN, _In_ OrtSessionOptions* options, int use_arena) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_ARMNN(use_arena));
  return nullptr;
}
