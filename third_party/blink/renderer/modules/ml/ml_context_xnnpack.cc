// Copyright 2022 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/ml_context_xnnpack.h"

#include "base/system/sys_info.h"
#include "third_party/blink/renderer/platform/wtf/thread_safe_ref_counted.h"
#include "third_party/blink/renderer/platform/wtf/threading_primitives.h"

namespace blink {

namespace {

class SharedXnnpackContext : public ThreadSafeRefCounted<SharedXnnpackContext> {
 public:
  static scoped_refptr<SharedXnnpackContext> GetInstance(size_t num_threads) {
    if (instance_ == nullptr) {
      scoped_refptr<SharedXnnpackContext> refptr =
          base::MakeRefCounted<SharedXnnpackContext>(num_threads);
      instance_ = refptr.get();
      return refptr;
    } else {
      return base::WrapRefCounted(instance_);
    }
  }

  explicit SharedXnnpackContext(size_t num_threads)
      : initialized_(false), num_threads_(num_threads) {}

  SharedXnnpackContext(const SharedXnnpackContext&) = delete;
  SharedXnnpackContext& operator=(const SharedXnnpackContext&) = delete;

  bool Initialize() {
    WTF::MutexLocker locker(mutex_);
    if (initialized_) {
      return true;
    }
#if BUILDFLAG(IS_WIN)
    if (xnn_initialize(NULL) != xnn_status_success) {
      return false;
    }
#endif

    uint32_t num_cores = base::SysInfo::NumberOfProcessors() / 2;
    size_t num_threads = num_threads_ == 0          ? num_cores
                         : num_threads_ > num_cores ? num_cores
                                                    : num_threads_;
    pthreadpool_ = pthreadpool_create(num_threads);
    if (pthreadpool_ == nullptr) {
      return false;
    }
    initialized_ = true;
    return true;
  }

  pthreadpool_t Pthreadpool() { return pthreadpool_; }

 private:
  friend class ThreadSafeRefCounted<SharedXnnpackContext>;

  ~SharedXnnpackContext() {
    WTF::MutexLocker locker(mutex_);
#if BUILDFLAG(IS_WIN)
    xnn_deinitialize();
#endif
    if (pthreadpool_ != nullptr) {
      pthreadpool_destroy(pthreadpool_);
    }
    instance_ = nullptr;
  }

  static SharedXnnpackContext* instance_;

  WTF::Mutex mutex_;
  bool initialized_;
  size_t num_threads_;
  pthreadpool_t pthreadpool_;
};

SharedXnnpackContext* SharedXnnpackContext::instance_ = nullptr;

}  // namespace

MLContextXnnpack::MLContextXnnpack(const unsigned int num_threads, ML* ml)
    : MLContext(V8MLDevicePreference(V8MLDevicePreference::Enum::kCpu),
                V8MLPowerPreference(V8MLPowerPreference::Enum::kAuto),
                V8MLModelFormat(V8MLModelFormat::Enum::kTflite),
                num_threads,
                ml),
      impl_(SharedXnnpackContext::GetInstance(num_threads)) {}

MLContextXnnpack::~MLContextXnnpack() = default;

bool MLContextXnnpack::Initialize() {
  return impl_->Initialize();
}

pthreadpool_t MLContextXnnpack::Pthreadpool() const {
  return impl_->Pthreadpool();
}

}  // namespace blink
