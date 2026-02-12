#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "ClusterPropSoA.h"
#include "specs.h"

using Dim = alpaka::DimInt<1>;
using Idx = std::size_t;

// ===== Alpaka Acc type selection =====
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
  using Acc   = alpaka::AccGpuCudaRt<Dim, Idx>;
  using Queue = alpaka::QueueCudaRtNonBlocking;
#elif defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
  using Acc   = alpaka::AccCpuSerial<Dim, Idx>;
  using Queue = alpaka::QueueCpuBlocking;
#else
  #error "No Alpaka backend enabled (need CUDA or CPU serial)."
#endif

namespace ot {

struct UnpackerDriver {
  UnpackerDriver();

  ClusterPropSoA run(
      std::vector<unsigned char> const& linearRaw,
      std::vector<std::size_t>   const& sizes,
      std::vector<std::size_t>   const& offsets,
      std::vector<int>           const& detIdxModuleType, // 0=undef,1=2S,2=PS
      std::vector<uint32_t>      const& innerDetId,
      std::vector<uint32_t>      const& outerDetId
  ) const;

private:
  using Device = alpaka::Dev<Acc>;

  template <typename T>
  using DevBuf = decltype(alpaka::allocBuf<T, Idx>(std::declval<Device>(),
                                                   alpaka::Vec<Dim, Idx>::all(0)));

  Device dev_;
  mutable Queue queue_;

  // capacities (must be before buffers, used in ctor initializer list)
  mutable std::size_t rawCap_;
  mutable std::size_t slinkCap_;
  mutable std::size_t mapCap_;
  mutable std::size_t outCap_;

  // preallocated device buffers
  mutable DevBuf<unsigned char> rawDev_;
  mutable DevBuf<std::size_t>   sizesDev_;
  mutable DevBuf<std::size_t>   offsDev_;
  mutable DevBuf<int>           modDev_;
  mutable DevBuf<uint32_t>      innerDev_;
  mutable DevBuf<uint32_t>      outerDev_;

  mutable DevBuf<uint32_t> outDet_;
  mutable DevBuf<uint16_t> outX_;
  mutable DevBuf<uint16_t> outY_;
  mutable DevBuf<uint8_t>  outZ_;
  mutable DevBuf<uint8_t>  outW_;
  mutable DevBuf<uint8_t>  outSeed_;
  mutable DevBuf<uint8_t>  outMip_;
  mutable DevBuf<uint8_t>  outMType_;
  mutable DevBuf<uint32_t> counter_;
};

} // namespace ot

