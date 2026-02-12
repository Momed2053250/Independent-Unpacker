#include "Unpacker.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <type_traits>
#include <vector>

ALPAKA_FN_HOST_ACC inline int createMask(int nBits) { return (1 << nBits) - 1; }

ALPAKA_FN_HOST_ACC inline uint32_t readLine(const unsigned char* dataPtr, int byteIdx) {
  return (static_cast<uint32_t>(dataPtr[byteIdx])     << 24) |
         (static_cast<uint32_t>(dataPtr[byteIdx + 1]) << 16) |
         (static_cast<uint32_t>(dataPtr[byteIdx + 2]) << 8)  |
          static_cast<uint32_t>(dataPtr[byteIdx + 3]);
}

ALPAKA_FN_HOST_ACC inline int getLineIndex(int channelIdx, unsigned int iline) {
  return channelIdx + N_BYTES_PER_WORD + static_cast<int>(iline * N_BYTES_PER_WORD);
}

ALPAKA_FN_HOST_ACC inline void readPayload(
    uint32_t* clusterWords,
    const uint32_t* lines,
    int numClusters,
    int& nAvailableBits,
    int& iLine,
    int& bitsToRead,
    int& nFullClusters,
    const int clusterBits,
    const int clusterWordMask,
    const bool isPixelCluster,
    int nFullClustersStrips = 0
) {
  for (int icluster = 0; icluster < numClusters; ++icluster) {
    if (nAvailableBits >= clusterBits) {
      int shift = N_BITS_PER_WORD - bitsToRead - (nFullClusters + 1) * clusterBits;
      if (icluster == 0 && isPixelCluster) shift -= (nFullClustersStrips) * SS_CLUSTER_BITS;
      nFullClustersStrips = 0;
      clusterWords[icluster] = (lines[iLine] >> shift) & clusterWordMask;
      nAvailableBits -= clusterBits;
      nFullClusters++;
      if (nAvailableBits == 0) {
        ++iLine;
        nAvailableBits = N_BITS_PER_WORD;
        nFullClusters = 0;
        bitsToRead = 0;
      }
    } else {
      const int nMask = createMask(nAvailableBits);
      const uint16_t wordLeft = static_cast<uint16_t>(lines[iLine] & nMask);
      bitsToRead = clusterBits - nAvailableBits;
      const int nextMask = createMask(bitsToRead);
      const uint16_t wordRight =
          static_cast<uint16_t>((lines[iLine + 1] >> (N_BITS_PER_WORD - bitsToRead)) & nextMask);
      clusterWords[icluster] = (static_cast<uint32_t>(wordLeft) << bitsToRead) | wordRight;
      nAvailableBits = N_BITS_PER_WORD - bitsToRead;
      ++iLine;
      nFullClusters = 0;
    }
  }
}

struct UnpackKernel {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(
      TAcc const& acc,
      unsigned char* raw,
      std::size_t* sizes,
      std::size_t* offsets,
      int* detIdxModuleType,
      uint32_t* innerDetIdForFlatIdx,
      uint32_t* outerDetIdForFlatIdx,
      uint32_t* outDet,
      uint16_t* outX,
      uint16_t* outY,
      uint8_t* outZ,
      uint8_t* outWidth,
      uint8_t* outIsSeed,
      uint8_t* outMip,
      uint8_t* outModType,
      uint32_t* globalCounter,
      uint32_t outCapacity
  ) const {

    const uint32_t gtid = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u];
    const uint32_t gdim = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0u];

    static constexpr int MaxOffsetWords   = (OFFSET_BITS * CICs_PER_SLINK) / N_BITS_PER_WORD;
    static constexpr int MaxStripClusters = N_CLUSTER_MASK + 1;
    static constexpr int MaxPixelClusters = N_CLUSTER_MASK + 1;
    static constexpr int MaxPayloadLines =
      ((MaxStripClusters * SS_CLUSTER_BITS + MaxPixelClusters * PX_CLUSTER_BITS) / N_BITS_PER_WORD) + 1;

    uint32_t offsetWords[MaxOffsetWords];
    uint32_t lines[MaxPayloadLines];
    uint32_t stripClusterWords[MaxStripClusters];
    uint32_t pixelClusterWords[MaxPixelClusters];

    const uint32_t NSlinks = (MAX_DTC_ID - MIN_DTC_ID + 1) * SLINKS_PER_DTC;

    for (uint32_t frdId = gtid; frdId < NSlinks; frdId += gdim) {
      if (sizes[frdId] == 0u) continue;

      const unsigned char* dataPtr = raw + offsets[frdId];

      const size_t nOffsetsLines = MaxOffsetWords;
      const size_t initByte = HEADER_N_LINES * N_BYTES_PER_WORD;
      for (size_t k = 0; k < nOffsetsLines; ++k) {
        const int byteIdx = static_cast<int>(initByte + k * N_BYTES_PER_WORD);
        offsetWords[k] = readLine(dataPtr, byteIdx);
      }

      for (unsigned int iChannel = 0; iChannel < CICs_PER_SLINK; ++iChannel) {
        const unsigned flatIdx = frdId * CICs_PER_SLINK + iChannel;

        const int moduleType = detIdxModuleType[flatIdx];
        if (moduleType == 0) continue;

        const bool is2SModule = (moduleType == 1);

        const size_t offsetTableStart = (HEADER_N_LINES + MODULES_PER_SLINK) * N_BYTES_PER_WORD;
        const int wordIdx = static_cast<int>(iChannel / 2);

        const uint16_t channelOffset16 = (iChannel % 2 == 0)
          ? static_cast<uint16_t>(offsetWords[wordIdx] & 0xFFFFu)
          : static_cast<uint16_t>(offsetWords[wordIdx] >> 16);

        const int idx = static_cast<int>(offsetTableStart + channelOffset16 * N_BYTES_PER_WORD);

        const uint32_t chHeaderWord = readLine(dataPtr, idx);
        const unsigned int numStripClusters =
          (chHeaderWord >> (N_BITS_PER_WORD - L1ID_BITS - CIC_ERROR_BITS - N_STRIP_CLUSTER_BITS)) & N_CLUSTER_MASK;
        const unsigned int numPixelClusters = chHeaderWord & N_CLUSTER_MASK;

        unsigned int nLines = 0;
        if (numStripClusters + numPixelClusters > 0) {
          const unsigned int neededBits =
            numStripClusters * SS_CLUSTER_BITS + numPixelClusters * PX_CLUSTER_BITS;
          nLines = static_cast<unsigned int>(neededBits / N_BITS_PER_WORD) + 1;
        }
        if (nLines > MaxPayloadLines) nLines = MaxPayloadLines;

        for (unsigned int k = 0; k < nLines; ++k) {
          const int byteIdx = getLineIndex(idx, k);
          lines[k] = readLine(dataPtr, byteIdx);
        }

        int nAvailableBits = N_BITS_PER_WORD;
        int iLine = 0;
        int bitsToRead = 0;
        int nFullClustersStrip = 0;
        int nFullClustersPix = 0;

        const unsigned int useStrip = (numStripClusters <= MaxStripClusters) ? numStripClusters : MaxStripClusters;
        const unsigned int usePixel = (numPixelClusters <= MaxPixelClusters) ? numPixelClusters : MaxPixelClusters;

        if (useStrip > 0) {
          readPayload(stripClusterWords, lines, static_cast<int>(useStrip),
                      nAvailableBits, iLine, bitsToRead, nFullClustersStrip,
                      SS_CLUSTER_BITS, SS_CLUSTER_WORD_MASK, false);
        }
        if (!is2SModule && usePixel > 0) {
          readPayload(pixelClusterWords, lines, static_cast<int>(usePixel),
                      nAvailableBits, iLine, bitsToRead, nFullClustersPix,
                      PX_CLUSTER_BITS, PX_CLUSTER_WORD_MASK, true, nFullClustersStrip);
        }

        const uint32_t writeCount = is2SModule ? useStrip : (useStrip + usePixel);
        if (writeCount == 0) continue;

        const uint32_t base = alpaka::atomicAdd(acc, globalCounter, writeCount);

        const uint32_t innerDet = innerDetIdForFlatIdx[flatIdx];
        const uint32_t outerDet = outerDetIdForFlatIdx[flatIdx];
        const uint8_t  parity   = static_cast<uint8_t>(iChannel & 0x1);

        if (is2SModule) {
          for (unsigned int ic = 0; ic < useStrip; ++ic) {
            const uint32_t outIdx = base + ic;
            if (outIdx >= outCapacity) continue; // CMSSW-like guard

            const uint32_t word = stripClusterWords[ic];
            const uint32_t chip = (word >> (SS_CLUSTER_BITS - CHIP_ID_BITS)) & CHIP_ID_MAX_VALUE;
            const uint32_t addr = (word >> (SS_CLUSTER_BITS - CHIP_ID_BITS - SCLUSTER_ADDRESS_ONLY_BITS_2S)) & SCLUSTER_ADDRESS_MASK;
            const bool     seed = (word >> (SS_CLUSTER_BITS - CHIP_ID_BITS - SCLUSTER_ADDRESS_BITS_2S)) & IS_SEED_SENSOR_MASK;
            uint32_t       w    = word & WIDTH_MAX_VALUE;
            if (w == 0) w = 8;

            outDet[outIdx]     = seed ? innerDet : outerDet;
            outX[outIdx]       = static_cast<uint16_t>(STRIPS_PER_CBC * chip + addr);
            outY[outIdx]       = static_cast<uint16_t>(parity);
            outZ[outIdx]       = 0u;
            outWidth[outIdx]   = static_cast<uint8_t>(w);
            outIsSeed[outIdx]  = static_cast<uint8_t>(seed);
            outMip[outIdx]     = 0u;
            outModType[outIdx] = 1u;
          }
        } else {
          for (unsigned int ic = 0; ic < useStrip; ++ic) {
            const uint32_t outIdx = base + ic;
            if (outIdx >= outCapacity) continue;

            const uint32_t word = stripClusterWords[ic];
            const uint32_t chip = (word >> (SS_CLUSTER_BITS - CHIP_ID_BITS)) & CHIP_ID_MAX_VALUE;
            const uint32_t addr = (word >> (SS_CLUSTER_BITS - CHIP_ID_BITS - SCLUSTER_ADDRESS_BITS_PS)) & SCLUSTER_ADDRESS_PS_MAX_VALUE;
            uint32_t       w    = (word >> (SS_CLUSTER_BITS - CHIP_ID_BITS - SCLUSTER_ADDRESS_BITS_PS - WIDTH_BITS)) & WIDTH_MAX_VALUE;
            const uint32_t mip  = word & MIP_BITS_MASK;
            if (w == 0) w = 8;

            outDet[outIdx]     = outerDet;
            outX[outIdx]       = static_cast<uint16_t>(STRIPS_PER_SSA * chip + addr);
            outY[outIdx]       = static_cast<uint16_t>(parity);
            outZ[outIdx]       = 0u;
            outWidth[outIdx]   = static_cast<uint8_t>(w);
            outIsSeed[outIdx]  = 0u;
            outMip[outIdx]     = static_cast<uint8_t>(mip);
            outModType[outIdx] = 2u;
          }

          for (unsigned int ic = 0; ic < usePixel; ++ic) {
            const uint32_t outIdx = base + useStrip + ic;
            if (outIdx >= outCapacity) continue;

            const uint32_t word = pixelClusterWords[ic];
            const uint32_t chip = (word >> (PX_CLUSTER_BITS - CHIP_ID_BITS)) & CHIP_ID_MAX_VALUE;
            const uint32_t addr = (word >> (PX_CLUSTER_BITS - CHIP_ID_BITS - SCLUSTER_ADDRESS_BITS_PS)) & SCLUSTER_ADDRESS_PS_MAX_VALUE;
            uint32_t       w    = (word >> (PX_CLUSTER_BITS - CHIP_ID_BITS - SCLUSTER_ADDRESS_BITS_PS - WIDTH_BITS)) & WIDTH_MAX_VALUE;
            const uint32_t z    = word & PS_Z_BITS_MASK;
            if (w == 0) w = 8;

            outDet[outIdx]     = innerDet;
            outX[outIdx]       = static_cast<uint16_t>(STRIPS_PER_SSA * chip + addr);
            outY[outIdx]       = static_cast<uint16_t>(parity == 0 ? z : (z + 16));
            outZ[outIdx]       = static_cast<uint8_t>(z);
            outWidth[outIdx]   = static_cast<uint8_t>(w);
            outIsSeed[outIdx]  = 1u;
            outMip[outIdx]     = 0u;
            outModType[outIdx] = 2u;
          }
        }
      }
    }
  }
};

namespace ot {

static inline alpaka::Vec<Dim, Idx> v1(Idx n) { return alpaka::Vec<Dim, Idx>::all(n); }

UnpackerDriver::UnpackerDriver()
  : dev_(alpaka::getDevByIdx(alpaka::Platform<Acc>{}, 0u))
  , queue_(dev_)
  , rawCap_(10'000'000)   // 10MB
  , slinkCap_(1'000)
  , mapCap_(31'104)
  , outCap_(1'000'000)
  , rawDev_(alpaka::allocBuf<unsigned char, Idx>(dev_, v1(rawCap_)))
  , sizesDev_(alpaka::allocBuf<std::size_t, Idx>(dev_, v1(slinkCap_)))
  , offsDev_(alpaka::allocBuf<std::size_t, Idx>(dev_, v1(slinkCap_)))
  , modDev_(alpaka::allocBuf<int, Idx>(dev_, v1(mapCap_)))
  , innerDev_(alpaka::allocBuf<uint32_t, Idx>(dev_, v1(mapCap_)))
  , outerDev_(alpaka::allocBuf<uint32_t, Idx>(dev_, v1(mapCap_)))
  , outDet_(alpaka::allocBuf<uint32_t, Idx>(dev_, v1(outCap_)))
  , outX_(alpaka::allocBuf<uint16_t, Idx>(dev_, v1(outCap_)))
  , outY_(alpaka::allocBuf<uint16_t, Idx>(dev_, v1(outCap_)))
  , outZ_(alpaka::allocBuf<uint8_t, Idx>(dev_, v1(outCap_)))
  , outW_(alpaka::allocBuf<uint8_t, Idx>(dev_, v1(outCap_)))
  , outSeed_(alpaka::allocBuf<uint8_t, Idx>(dev_, v1(outCap_)))
  , outMip_(alpaka::allocBuf<uint8_t, Idx>(dev_, v1(outCap_)))
  , outMType_(alpaka::allocBuf<uint8_t, Idx>(dev_, v1(outCap_)))
  , counter_(alpaka::allocBuf<uint32_t, Idx>(dev_, v1(1)))
{}

ClusterPropSoA UnpackerDriver::run(
    std::vector<unsigned char> const& linearRaw,
    std::vector<std::size_t>   const& sizes,
    std::vector<std::size_t>   const& offsets,
    std::vector<int>           const& detIdxModuleType,
    std::vector<uint32_t>      const& innerDetId,
    std::vector<uint32_t>      const& outerDetId
) const {

  // grow buffers if needed (kept minimal; avoids memcpy out-of-bounds)
  if (linearRaw.size() > rawCap_) {
    rawCap_ = linearRaw.size();
    rawDev_ = alpaka::allocBuf<unsigned char, Idx>(dev_, v1(rawCap_));
  }
  if (sizes.size() > slinkCap_) {
    slinkCap_ = sizes.size();
    sizesDev_ = alpaka::allocBuf<std::size_t, Idx>(dev_, v1(slinkCap_));
    offsDev_  = alpaka::allocBuf<std::size_t, Idx>(dev_, v1(slinkCap_));
  }
  if (detIdxModuleType.size() > mapCap_) {
    mapCap_  = detIdxModuleType.size();
    modDev_   = alpaka::allocBuf<int, Idx>(dev_, v1(mapCap_));
    innerDev_ = alpaka::allocBuf<uint32_t, Idx>(dev_, v1(mapCap_));
    outerDev_ = alpaka::allocBuf<uint32_t, Idx>(dev_, v1(mapCap_));
  }

  // Copy data to device
  {
    auto host = alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0);

    auto copyVec = [&](auto const& vec, auto& devbuf) {
      using VecT = std::decay_t<decltype(vec)>;
      using Elem = typename VecT::value_type;
      if (vec.empty()) return;
      auto hostbuf = alpaka::allocBuf<Elem, Idx>(host, v1(vec.size()));
      std::memcpy(alpaka::getPtrNative(hostbuf), vec.data(), vec.size() * sizeof(Elem));
      alpaka::memcpy(queue_, devbuf, hostbuf, vec.size());
    };

    copyVec(linearRaw, rawDev_);
    copyVec(sizes, sizesDev_);
    copyVec(offsets, offsDev_);
    copyVec(detIdxModuleType, modDev_);
    copyVec(innerDetId, innerDev_);
    copyVec(outerDetId, outerDev_);

    auto hostCnt = alpaka::allocBuf<uint32_t, Idx>(host, v1(1));
    *alpaka::getPtrNative(hostCnt) = 0u;
    alpaka::memcpy(queue_, counter_, hostCnt);
    alpaka::wait(queue_);
  }

  const uint32_t NSlinks = (MAX_DTC_ID - MIN_DTC_ID + 1) * SLINKS_PER_DTC;

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
  const uint32_t threadsPerBlock = 128;
  const uint32_t blocks = (NSlinks + threadsPerBlock - 1) / threadsPerBlock;
#else
  const uint32_t threadsPerBlock = 1;
  const uint32_t blocks = 1;
#endif

  auto workDiv = alpaka::WorkDivMembers<Dim, Idx>(v1(blocks), v1(threadsPerBlock), v1(1));

  alpaka::wait(queue_);

  // kernel-only timing
  auto t0 = std::chrono::steady_clock::now();

  UnpackKernel kernel;
  alpaka::exec<Acc>(
      queue_, workDiv, kernel,
      alpaka::getPtrNative(rawDev_),
      alpaka::getPtrNative(sizesDev_),
      alpaka::getPtrNative(offsDev_),
      alpaka::getPtrNative(modDev_),
      alpaka::getPtrNative(innerDev_),
      alpaka::getPtrNative(outerDev_),
      alpaka::getPtrNative(outDet_),
      alpaka::getPtrNative(outX_),
      alpaka::getPtrNative(outY_),
      alpaka::getPtrNative(outZ_),
      alpaka::getPtrNative(outW_),
      alpaka::getPtrNative(outSeed_),
      alpaka::getPtrNative(outMip_),
      alpaka::getPtrNative(outMType_),
      alpaka::getPtrNative(counter_),
      static_cast<uint32_t>(outCap_)
  );

  alpaka::wait(queue_);

  auto t1 = std::chrono::steady_clock::now();
  std::cout << "[Standalone] KERNEL ONLY time: "
            << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
            << " us\n";

  // read back counter
  uint32_t nOut = 0;
  {
    auto host = alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0);
    auto hostCnt = alpaka::allocBuf<uint32_t, Idx>(host, v1(1));
    alpaka::memcpy(queue_, hostCnt, counter_);
    alpaka::wait(queue_);
    nOut = *alpaka::getPtrNative(hostCnt);
  }
  nOut = std::min<uint32_t>(nOut, static_cast<uint32_t>(outCap_));

  ClusterPropSoA out;
  out.reserve(nOut);

  auto copyBack = [&](auto const& devbuf, auto* tmp) {
    if (nOut == 0) return;
    auto host = alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0);
    using Elem = std::remove_pointer_t<decltype(tmp)>;
    auto hostBuf = alpaka::allocBuf<Elem, Idx>(host, v1(nOut));
    alpaka::memcpy(queue_, hostBuf, devbuf, nOut);
    alpaka::wait(queue_);
    std::memcpy(tmp, alpaka::getPtrNative(hostBuf), nOut * sizeof(Elem));
  };

  std::vector<uint32_t> vDet(nOut);
  std::vector<uint16_t> vX(nOut), vY(nOut);
  std::vector<uint8_t>  vZ(nOut), vW(nOut), vSeed(nOut), vMip(nOut), vMType(nOut);

  copyBack(outDet_, vDet.data());
  copyBack(outX_,   vX.data());
  copyBack(outY_,   vY.data());
  copyBack(outZ_,   vZ.data());
  copyBack(outW_,   vW.data());
  copyBack(outSeed_,vSeed.data());
  copyBack(outMip_, vMip.data());
  copyBack(outMType_, vMType.data());

  for (uint32_t i = 0; i < nOut; ++i) {
    out.push_back(ClusterProp{ vDet[i], vX[i], vY[i], vZ[i], vW[i], vSeed[i], vMip[i], vMType[i] });
  }
  return out;
}

} // namespace ot

#ifndef NO_MAIN_IN_UNPACKER
int main() { return 0; }
#endif