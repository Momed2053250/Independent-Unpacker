#pragma once
#include <cstdint>
#include <vector>
#include <cstddef>
#include "ClusterPropSoA.h"
#include "specs.h"
#include <alpaka/alpaka.hpp>

using Dim = alpaka::DimInt<1>;  // prcessing linear data
using Idx = std::size_t; // index type for addressing 
// TODO: add all ACC types the ones needed and arrange in a working way with the CMAKE options
// ===== Alpaka Acc type selection =====
#if defined(ALPAKA_CUDA_ENABLED)  // use non blocking queue in GPU
  using Acc = alpaka::AccGpuCudaRt<Dim, Idx>;
  using Queue = alpaka::QueueCudaRtNonBlocking;
#elif defined(ALPAKA_CPU_SERIAL_ENABLED)   // uses blocking queue only for CPU serial
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