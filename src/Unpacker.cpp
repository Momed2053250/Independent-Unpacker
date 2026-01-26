#include "Unpacker.h"
#include <iostream>
#include <cstring>
#include <vector>
#include <cstdint>

using namespace std;

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
      const uint16_t wordRight = static_cast<uint16_t>((lines[iLine + 1] >> (N_BITS_PER_WORD - bitsToRead)) & nextMask);
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
      uint32_t* globalCounter
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
            const uint32_t word = stripClusterWords[ic];
            const uint32_t chip = (word >> (SS_CLUSTER_BITS - CHIP_ID_BITS)) & CHIP_ID_MAX_VALUE;
            const uint32_t addr = (word >> (SS_CLUSTER_BITS - CHIP_ID_BITS - SCLUSTER_ADDRESS_ONLY_BITS_2S)) & SCLUSTER_ADDRESS_MASK;
            const bool     seed = (word >> (SS_CLUSTER_BITS - CHIP_ID_BITS - SCLUSTER_ADDRESS_BITS_2S)) & IS_SEED_SENSOR_MASK;
            uint32_t       w    = word & WIDTH_MAX_VALUE;
            if (w == 0) w = 8;

            const uint32_t outIdx = base + ic;
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
            const uint32_t word = stripClusterWords[ic];
            const uint32_t chip = (word >> (SS_CLUSTER_BITS - CHIP_ID_BITS)) & CHIP_ID_MAX_VALUE;
            const uint32_t addr = (word >> (SS_CLUSTER_BITS - CHIP_ID_BITS - SCLUSTER_ADDRESS_BITS_PS)) & SCLUSTER_ADDRESS_PS_MAX_VALUE;
            uint32_t       w    = (word >> (SS_CLUSTER_BITS - CHIP_ID_BITS - SCLUSTER_ADDRESS_BITS_PS - WIDTH_BITS)) & WIDTH_MAX_VALUE;
            const uint32_t mip  = word & MIP_BITS_MASK;
            if (w == 0) w = 8;

            const uint32_t outIdx = base + ic;
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
            const uint32_t word = pixelClusterWords[ic];
            const uint32_t chip = (word >> (PX_CLUSTER_BITS - CHIP_ID_BITS)) & CHIP_ID_MAX_VALUE;
            const uint32_t addr = (word >> (PX_CLUSTER_BITS - CHIP_ID_BITS - SCLUSTER_ADDRESS_BITS_PS)) & SCLUSTER_ADDRESS_PS_MAX_VALUE;
            uint32_t       w    = (word >> (PX_CLUSTER_BITS - CHIP_ID_BITS - SCLUSTER_ADDRESS_BITS_PS - WIDTH_BITS)) & WIDTH_MAX_VALUE;
            const uint32_t z    = word & PS_Z_BITS_MASK;
            if (w == 0) w = 8;

            const uint32_t outIdx = base + useStrip + ic;
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

namespace {
template <typename T> using Buf = decltype(alpaka::allocBuf<T, Idx>(std::declval<alpaka::DevCpu>(), alpaka::Vec<alpaka::DimInt<1>, Idx>::all(0)));
}

namespace ot {

ClusterPropSoA UnpackerDriver::run(
    std::vector<unsigned char> const& linearRaw,
    std::vector<std::size_t>   const& sizes,
    std::vector<std::size_t>   const& offsets,
    std::vector<int>           const& detIdxModuleType,
    std::vector<uint32_t>      const& innerDetId,
    std::vector<uint32_t>      const& outerDetId
) const {
  auto dev = alpaka::getDevByIdx(alpaka::Platform<Acc>{}, 0u);
  Queue q{dev};

  const uint32_t NSlinks = (MAX_DTC_ID - MIN_DTC_ID + 1) * SLINKS_PER_DTC;
  const std::size_t maxClusters =
      (N_CLUSTER_MASK + 1) * CICs_PER_SLINK * (MAX_DTC_ID - MIN_DTC_ID + 1) * SLINKS_PER_DTC;

  auto rawDev    = alpaka::allocBuf<unsigned char, Idx>(dev, alpaka::Vec<alpaka::DimInt<1>, Idx>::all(linearRaw.size()));
  auto sizesDev  = alpaka::allocBuf<std::size_t, Idx>(dev, alpaka::Vec<alpaka::DimInt<1>, Idx>::all(sizes.size()));
  auto offsDev   = alpaka::allocBuf<std::size_t, Idx>(dev, alpaka::Vec<alpaka::DimInt<1>, Idx>::all(offsets.size()));
  auto modDev    = alpaka::allocBuf<int, Idx>(dev, alpaka::Vec<alpaka::DimInt<1>, Idx>::all(detIdxModuleType.size()));
  auto innerDev  = alpaka::allocBuf<uint32_t, Idx>(dev, alpaka::Vec<alpaka::DimInt<1>, Idx>::all(innerDetId.size()));
  auto outerDev  = alpaka::allocBuf<uint32_t, Idx>(dev, alpaka::Vec<alpaka::DimInt<1>, Idx>::all(outerDetId.size()));
  auto outDet    = alpaka::allocBuf<uint32_t, Idx>(dev, alpaka::Vec<alpaka::DimInt<1>, Idx>::all(maxClusters));
  auto outX      = alpaka::allocBuf<uint16_t, Idx>(dev, alpaka::Vec<alpaka::DimInt<1>, Idx>::all(maxClusters));
  auto outY      = alpaka::allocBuf<uint16_t, Idx>(dev, alpaka::Vec<alpaka::DimInt<1>, Idx>::all(maxClusters));
  auto outZ      = alpaka::allocBuf<uint8_t,  Idx>(dev, alpaka::Vec<alpaka::DimInt<1>, Idx>::all(maxClusters));
  auto outW      = alpaka::allocBuf<uint8_t,  Idx>(dev, alpaka::Vec<alpaka::DimInt<1>, Idx>::all(maxClusters));
  auto outSeed   = alpaka::allocBuf<uint8_t,  Idx>(dev, alpaka::Vec<alpaka::DimInt<1>, Idx>::all(maxClusters));
  auto outMip    = alpaka::allocBuf<uint8_t,  Idx>(dev, alpaka::Vec<alpaka::DimInt<1>, Idx>::all(maxClusters));
  auto outMType  = alpaka::allocBuf<uint8_t,  Idx>(dev, alpaka::Vec<alpaka::DimInt<1>, Idx>::all(maxClusters));
  auto counter   = alpaka::allocBuf<uint32_t, Idx>(dev, alpaka::Vec<alpaka::DimInt<1>, Idx>::all(1));
  
  {
    auto host = alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0);
    auto mkview = [&](auto const& vec, auto& devbuf){
      using Elem = std::remove_cv_t<std::remove_reference_t<decltype(vec[0])>>;
      auto extent = alpaka::getExtents(devbuf)[0];
      auto hostbuf = alpaka::allocBuf<Elem, Idx>(host, alpaka::Vec<alpaka::DimInt<1>, Idx>::all(extent));
      auto* ptr = alpaka::getPtrNative(hostbuf);
      std::memcpy(static_cast<void*>(ptr), vec.data(), vec.size() * sizeof(Elem));
      alpaka::memcpy(q, devbuf, hostbuf);
      alpaka::wait(q);
    };

    mkview(linearRaw, rawDev);
    mkview(sizes,    sizesDev);
    mkview(offsets,  offsDev);
    mkview(detIdxModuleType, modDev);
    mkview(innerDetId, innerDev);
    mkview(outerDetId, outerDev);

    auto hostCnt = alpaka::allocBuf<uint32_t, Idx>(host, alpaka::Vec<alpaka::DimInt<1>, Idx>::all(1));
    *alpaka::getPtrNative(hostCnt) = 0u;
    alpaka::memcpy(q, counter, hostCnt);
    alpaka::wait(q);
  }
  
  #ifdef ALPAKA_CUDA_ENABLED
  const uint32_t threadsPerBlock = 1024;
  const uint32_t blocks = (NSlinks + threadsPerBlock - 1) / threadsPerBlock;
  #else
  const uint32_t threadsPerBlock = 1;
  const uint32_t blocks = 1;
  #endif
  
  auto workDiv = alpaka::WorkDivMembers<alpaka::DimInt<1>, Idx>(
      alpaka::Vec<alpaka::DimInt<1>, Idx>::all(blocks),
      alpaka::Vec<alpaka::DimInt<1>, Idx>::all(threadsPerBlock),
      alpaka::Vec<alpaka::DimInt<1>, Idx>::all(1));

  UnpackKernel kernel;
  alpaka::exec<Acc>(
    q, workDiv, kernel,
    alpaka::getPtrNative(rawDev),
    alpaka::getPtrNative(sizesDev),
    alpaka::getPtrNative(offsDev),
    alpaka::getPtrNative(modDev),
    alpaka::getPtrNative(innerDev),
    alpaka::getPtrNative(outerDev),
    alpaka::getPtrNative(outDet),
    alpaka::getPtrNative(outX),
    alpaka::getPtrNative(outY),
    alpaka::getPtrNative(outZ),
    alpaka::getPtrNative(outW),
    alpaka::getPtrNative(outSeed),
    alpaka::getPtrNative(outMip),
    alpaka::getPtrNative(outMType),
    alpaka::getPtrNative(counter)
  );

  alpaka::wait(q);

  uint32_t nOut = 0;
  {
    auto host = alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0);
    auto hostCnt = alpaka::allocBuf<uint32_t, Idx>(host, alpaka::Vec<alpaka::DimInt<1>, Idx>::all(1));
    alpaka::memcpy(q, hostCnt, counter);
    alpaka::wait(q);
    nOut = *alpaka::getPtrNative(hostCnt);
  }

  ClusterPropSoA out;
  out.reserve(nOut);

  auto copyBack = [&](auto const& devbuf, auto* tmp){
    auto host = alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0);
    auto hostBuf = alpaka::allocBuf<std::remove_reference_t<decltype(*tmp)>, Idx>(host, alpaka::Vec<alpaka::DimInt<1>, Idx>::all(nOut));
    alpaka::memcpy(q, hostBuf, devbuf, nOut);
    alpaka::wait(q);
    std::memcpy(tmp, alpaka::getPtrNative(hostBuf), nOut * sizeof(*tmp));
  };

  std::vector<uint32_t> vDet(nOut);
  std::vector<uint16_t> vX(nOut), vY(nOut);
  std::vector<uint8_t>  vZ(nOut), vW(nOut), vSeed(nOut), vMip(nOut), vMType(nOut);

  copyBack(outDet, vDet.data());
  copyBack(outX,   vX.data());
  copyBack(outY,   vY.data());
  copyBack(outZ,   vZ.data());
  copyBack(outW,   vW.data());
  copyBack(outSeed,vSeed.data());
  copyBack(outMip, vMip.data());
  copyBack(outMType,vMType.data());

  for (uint32_t i = 0; i < nOut; ++i) {
    out.push_back(ClusterProp{
      vDet[i], vX[i], vY[i], vZ[i], vW[i], vSeed[i], vMip[i], vMType[i]
    });
  }
  return out;
}

}

#ifndef NO_MAIN_IN_UNPACKER
int main() {
  return 0;
}
#endif