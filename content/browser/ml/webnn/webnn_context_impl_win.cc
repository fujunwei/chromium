// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/browser/ml/webnn/webnn_context_impl_win.h"

#include <dxgi1_4.h>
#include <dxgi1_6.h>

#include "base/memory/ptr_util.h"
#include "content/browser/ml/webnn/dml/execution_context.h"
#include "content/browser/ml/webnn/dml/graph_dml_impl.h"
#include "mojo/public/cpp/bindings/self_owned_receiver.h"

namespace content::webnn {

namespace {

using ml::webnn::mojom::CreateGraphOptionsPtr;
using ml::webnn::mojom::CreateGraphResult;
using ml::webnn::mojom::WebnnContext;

std::map<AdapterType, scoped_refptr<AdapterDML>> EnumerateAdapters() {
  std::map<AdapterType, scoped_refptr<AdapterDML>> adapter_map = {};
  ComPtr<IDXGIFactory6> dxgi_factory = nullptr;
  HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&dxgi_factory));
  if (FAILED(hr)) {
    DLOG(ERROR) << "Create DXGI factory failed: "
                << logging::SystemErrorCodeToString(hr);
    return adapter_map;
  }

  // Eumerate all available adapters, DXGI_ERROR_NOT_FOUND means there are no
  // more adapters to enumerate.
  ComPtr<IDXGIAdapter1> dxgi_adapter;
  uint32_t adapter_index = 0;
  while (dxgi_factory->EnumAdapters1(++adapter_index, &dxgi_adapter) !=
         DXGI_ERROR_NOT_FOUND) {
    ComPtr<IDXGIAdapter3> dxgi_adapter3 = nullptr;
    hr = dxgi_adapter.As(&dxgi_adapter3);
    if (FAILED(hr)) {
      DLOG(ERROR) << "Get adapter3 failed: "
                  << logging::SystemErrorCodeToString(hr);
      return adapter_map;
    }
    auto adapter = base::MakeRefCounted<AdapterDML>(std::move(dxgi_adapter3));
    if (FAILED(adapter->Initialize())) {
      DLOG(ERROR) << "Initialize adapter failed: "
                  << logging::SystemErrorCodeToString(hr);
      return adapter_map;
    }

    adapter_map[adapter->GetAdapterType()] = adapter;
  }

  return adapter_map;
}

scoped_refptr<AdapterDML> RequestAdapter(
    std::map<AdapterType, scoped_refptr<AdapterDML>>& adapter_map,
    PowerPreference power_preference) {
  AdapterType preferred_type;
  switch (power_preference) {
    case PowerPreference::kLowPower:
      preferred_type = AdapterType::kIntegratedGPU;
      break;
    case PowerPreference::kDefault:
    case PowerPreference::kHighPerformance:
      preferred_type = AdapterType::kDiscreteGPU;
      break;
  }
  auto iter = adapter_map.find(preferred_type);
  if (iter != adapter_map.end()) {
    return iter->second;
  }

  // Select device sequentially if there is no preferred type.
  iter = adapter_map.find(AdapterType::kDiscreteGPU);
  if (iter != adapter_map.end()) {
    return iter->second;
  }

  iter = adapter_map.find(AdapterType::kIntegratedGPU);
  if (iter != adapter_map.end()) {
    return iter->second;
  }

  iter = adapter_map.find(AdapterType::kCPU);
  if (iter != adapter_map.end()) {
    return iter->second;
  }

  return nullptr;
}

}  // namespace

std::map<AdapterType, scoped_refptr<AdapterDML>>
    WebnnContextImplWin::adapter_map_ = {};

// static
void WebnnContextImplWin::Create(mojo::PendingReceiver<WebnnContext> receiver) {
  mojo::MakeSelfOwnedReceiver<WebnnContext>(
      base::WrapUnique(new WebnnContextImplWin()), std::move(receiver));
}

WebnnContextImplWin::~WebnnContextImplWin() = default;

WebnnContextImplWin::WebnnContextImplWin() {
  if (adapter_map_.empty()) {
    adapter_map_ = EnumerateAdapters();
  }
}

void WebnnContextImplWin::CreateGraph(
    CreateGraphOptionsPtr options,
    WebnnContext::CreateGraphCallback callback) {
  auto adapter = RequestAdapter(adapter_map_, options->power_preference);
  if (!adapter) {
    std::move(callback).Run(CreateGraphResult::kUnknownError,
                            mojo::NullRemote());
    return;
  }

  scoped_refptr<ExecutionContext> execution_context =
      ExecutionContext::GetInstance(adapter);
  if (!execution_context) {
    DLOG(ERROR) << "Create execution context failed.";
    std::move(callback).Run(CreateGraphResult::kUnknownError,
                            mojo::NullRemote());
    return;
  }

  // The remote sent to the renderer.
  mojo::PendingRemote<ml::webnn::mojom::WebnnGraph> blink_remote;
  // The receiver bind to GraphDMLImpl.
  GraphDMLImpl::Create(blink_remote.InitWithNewPipeAndPassReceiver(),
                       execution_context);
  std::move(callback).Run(CreateGraphResult::kOk, std::move(blink_remote));
}

}  // namespace content::webnn
