#pragma once
#include <cstdint>
#include <vector>
#include <cstddef>
#include "ClusterPropSoA.h"
#include "specs.h"
#include <alpaka/alpaka.hpp>

using Dim = alpaka::DimInt<1>;
using Idx = std::size_t;

// ===== Alpaka Acc type selection for Maxwell =====
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLE) && ALPAKA_ACC_GPU_CUDA_ENABLE
  // Maxwell-optimized CUDA backend
  using Acc = alpaka::AccGpuCudaRt<Dim, Idx>;
  using Queue = alpaka::QueueCudaRtNonBlocking;
#else
  // CPU fallback
  using Acc = alpaka::AccCpuSerial<Dim, Idx>;
  using Queue = alpaka::QueueCpuBlocking;
#endif

namespace ot {

// host-side driver
struct UnpackerDriver {
  ClusterPropSoA run(
      std::vector<unsigned char> const& linearRaw,
      std::vector<std::size_t>   const& sizes,
      std::vector<std::size_t>   const& offsets,
      std::vector<int>           const& detIdxModuleType, // 0=undef,1=2S,2=PS
      std::vector<uint32_t>      const& innerDetId,
      std::vector<uint32_t>      const& outerDetId
  ) const;
};
 
} // namespace ot