#include "Unpacker.h"
#include <iostream>
#include <cstring>
#include <vector>
#include <cstdint>

//
using namespace std;

// ======== device helpers ========
ALPAKA_FN_HOST_ACC inline int createMask(int nBits) { return (1 << nBits) - 1; }

ALPAKA_FN_HOST_ACC inline uint32_t readLine(const unsigned char* dataPtr, int byteIdx) {
  return (static_cast<uint32_t>(dataPtr[byteIdx])     << 24) |
         (static_cast<uint32_t>(dataPtr[byteIdx + 1]) << 16) |
         (static_cast<uint32_t>(dataPtr[byteIdx + 2]) << 8)  |
          static_cast<uint32_t>(dataPtr[byteIdx + 3]);
}

ALPAKA_FN_HOST_ACC inline int getLineIndex(int byteBase, unsigned int iline) {
  return byteBase + static_cast<int>(iline * N_BYTES_PER_WORD);
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

// ======== kernel ========
struct UnpackKernel {
  template <
    typename TAcc,
    typename RawView, typename SizeView, typename OffView,
    typename ModTypeView, typename InnerView, typename OuterView,
    typename OutDetView, typename OutXView, typename OutYView, typename OutZView,
    typename OutWView, typename OutSeedView, typename OutMipView, typename OutModTypeView,
    typename CounterPtr
  >
  ALPAKA_FN_ACC void operator()(
      TAcc const& acc,
      RawView raw, SizeView sizes, OffView offsets,
      ModTypeView detIdxModuleType,
      InnerView innerDetIdForFlatIdx,
      OuterView outerDetIdForFlatIdx,
      OutDetView outDet, OutXView outX, OutYView outY, OutZView outZ,
      OutWView outWidth, OutSeedView outIsSeed, OutMipView outMip, OutModTypeView outModType,
      CounterPtr globalCounter
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

    for (uint32_t sl = gtid; sl < NSlinks; sl += gdim) {
      if (sizes[sl] == 0u) continue;

      const unsigned char* dataPtr = raw + offsets[sl];

      // read offset table (immediately after header + module table)
      const size_t nOffsetsLines = MaxOffsetWords;
      const size_t initByte = HEADER_N_LINES * N_BYTES_PER_WORD;
      for (size_t k = 0; k < nOffsetsLines; ++k) {
        const int byteIdx = static_cast<int>(initByte + k * N_BYTES_PER_WORD);
        offsetWords[k] = readLine(dataPtr, byteIdx);
      }

      for (unsigned iChannel = 0; iChannel < static_cast<unsigned>(CICs_PER_SLINK); ++iChannel) {
        const unsigned flatIdx = sl * CICs_PER_SLINK + iChannel;
        const int moduleType = detIdxModuleType[flatIdx]; // 0:undef, 1:2S, 2:PS
        if (moduleType == 0) continue;
        const bool is2SModule = (moduleType == 1);

        const size_t offsetTableStart = (HEADER_N_LINES + MODULES_PER_SLINK) * N_BYTES_PER_WORD;
        const int wordIdx = static_cast<int>(iChannel / 2);
        const uint16_t channelOffset16 = (iChannel % 2 == 0)
          ? static_cast<uint16_t>(offsetWords[wordIdx] & 0xFFFFu) // if true
          : static_cast<uint16_t>(offsetWords[wordIdx] >> 16); // if false 
        const int byteBase = static_cast<int>(offsetTableStart + channelOffset16 * N_BYTES_PER_WORD);

        // channel header
        const uint32_t chHeaderWord = readLine(dataPtr, byteBase);
        const unsigned numStripClusters =
          (chHeaderWord >> (N_BITS_PER_WORD - L1ID_BITS - CIC_ERROR_BITS - N_STRIP_CLUSTER_BITS)) & N_CLUSTER_MASK;
        const unsigned numPixelClusters = chHeaderWord & N_CLUSTER_MASK;

        // payload lines
        unsigned int nLines = 0;
        if (numStripClusters + numPixelClusters > 0) {
          const unsigned int neededBits =
            numStripClusters * SS_CLUSTER_BITS + numPixelClusters * PX_CLUSTER_BITS;
          nLines = static_cast<unsigned int>(neededBits / N_BITS_PER_WORD) + 1;
        }

        if (nLines > MaxPayloadLines) nLines = MaxPayloadLines;
        for (unsigned k = 0; k < nLines; ++k) {
          const int byteIdx = getLineIndex(byteBase, k);
          lines[k] = readLine(dataPtr, byteIdx);
        }

        // unpack
        int nAvailableBits = N_BITS_PER_WORD;
        int iLine = 0;
        int bitsToRead = 0;
        int nFullClustersStrip = 0;
        int nFullClustersPix = 0;

        const unsigned useStrip = (numStripClusters <= static_cast<unsigned>(MaxStripClusters)) ? numStripClusters : MaxStripClusters;
        const unsigned usePixel = (numPixelClusters <= static_cast<unsigned>(MaxPixelClusters)) ? numPixelClusters : MaxPixelClusters;

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
        if (writeCount == 0u) continue;
        const uint32_t base = alpaka::atomicAdd(acc, globalCounter, writeCount);
        const uint32_t innerDet = innerDetIdForFlatIdx[flatIdx];
        const uint32_t outerDet = outerDetIdForFlatIdx[flatIdx];
        const uint8_t parity = static_cast<uint8_t>(iChannel & 0x1);

        // 2S strips
        if (is2SModule) {
          for (unsigned ic = 0; ic < useStrip; ++ic) {
            const uint32_t word = stripClusterWords[ic];
            const uint32_t chip = (word >> (SS_CLUSTER_BITS - CHIP_ID_BITS)) & CHIP_ID_MAX_VALUE;
            const uint32_t addr = (word >> (SS_CLUSTER_BITS - CHIP_ID_BITS - SCLUSTER_ADDRESS_ONLY_BITS_2S)) & SCLUSTER_ADDRESS_MASK;
            const bool     seed = (word >> (SS_CLUSTER_BITS - CHIP_ID_BITS - SCLUSTER_ADDRESS_BITS_2S)) & IS_SEED_SENSOR_MASK;
            uint32_t       w    = word & WIDTH_MAX_VALUE;  // WIDTH_BITS = 3
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
          // PS strips (outer) — decode with SS bit-widths and MIP field
          for (unsigned ic = 0; ic < useStrip; ++ic) {
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

          // PS pixels (inner)
          for (unsigned ic = 0; ic < usePixel; ++ic) {
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

// ======== host driver ========
// This defines a translation-unit–local template alias named Buf that yields the exact return type
// of an Alpaka buffer allocation call for a given element type T. Placing it inside an anonymous namespace 
// gives the alias internal linkage so it is visible only inside this .cpp file.

// !! alpaka::allocBuf<TElem>(device, extents) --> Device given here as DevCpu !! 
namespace {

template <typename T> using Buf = decltype(alpaka::allocBuf<T, Idx>(std::declval<alpaka::DevCpu>(), alpaka::Vec<alpaka::DimInt<1>, Idx>::all(0)));

} // anon

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

  // Buffer allocations: 
  /*
  Raw Data 
  Sizes
  Offsets
  Module Type 
  Inner DetId
  Outer DetId
  */
  auto rawDev    = alpaka::allocBuf<unsigned char, Idx>(dev, alpaka::Vec<alpaka::DimInt<1>, Idx>::all(linearRaw.size()));
  auto sizesDev  = alpaka::allocBuf<std::size_t, Idx>(dev, alpaka::Vec<alpaka::DimInt<1>, Idx>::all(sizes.size()));
  auto offsDev   = alpaka::allocBuf<std::size_t, Idx>(dev, alpaka::Vec<alpaka::DimInt<1>, Idx>::all(offsets.size()));
  auto modDev    = alpaka::allocBuf<int, Idx>(dev, alpaka::Vec<alpaka::DimInt<1>, Idx>::all(detIdxModuleType.size()));
  auto innerDev  = alpaka::allocBuf<uint32_t, Idx>(dev, alpaka::Vec<alpaka::DimInt<1>, Idx>::all(innerDetId.size()));
  auto outerDev  = alpaka::allocBuf<uint32_t, Idx>(dev, alpaka::Vec<alpaka::DimInt<1>, Idx>::all(outerDetId.size()));
  /*
  */
  auto outDet    = alpaka::allocBuf<uint32_t, Idx>(dev, alpaka::Vec<alpaka::DimInt<1>, Idx>::all(maxClusters));
  auto outX      = alpaka::allocBuf<uint16_t, Idx>(dev, alpaka::Vec<alpaka::DimInt<1>, Idx>::all(maxClusters));
  auto outY      = alpaka::allocBuf<uint16_t, Idx>(dev, alpaka::Vec<alpaka::DimInt<1>, Idx>::all(maxClusters));
  auto outZ      = alpaka::allocBuf<uint8_t,  Idx>(dev, alpaka::Vec<alpaka::DimInt<1>, Idx>::all(maxClusters));
  auto outW      = alpaka::allocBuf<uint8_t,  Idx>(dev, alpaka::Vec<alpaka::DimInt<1>, Idx>::all(maxClusters));
  auto outSeed   = alpaka::allocBuf<uint8_t,  Idx>(dev, alpaka::Vec<alpaka::DimInt<1>, Idx>::all(maxClusters));
  auto outMip    = alpaka::allocBuf<uint8_t,  Idx>(dev, alpaka::Vec<alpaka::DimInt<1>, Idx>::all(maxClusters));
  auto outMType  = alpaka::allocBuf<uint8_t,  Idx>(dev, alpaka::Vec<alpaka::DimInt<1>, Idx>::all(maxClusters));
  
  // counter buff 
  auto counter   = alpaka::allocBuf<uint32_t, Idx>(dev, alpaka::Vec<alpaka::DimInt<1>, Idx>::all(1));
  {
    auto host = alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0);
    auto mkview = [&](auto const& vec, auto& devbuf){
      using Elem = std::remove_cv_t<std::remove_reference_t<decltype(vec[0])>>;
      // alpaka extent represents the length you can iterate over or use when launching work.
      auto extent = alpaka::getExtents(devbuf)[0];
      auto hostbuf = alpaka::allocBuf<Elem, Idx>(host, alpaka::Vec<alpaka::DimInt<1>, Idx>::all(extent));
      auto* ptr = alpaka::getPtrNative(hostbuf); // ptr is Elem*
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

  const uint32_t threadsPerBlock = 128;
  const uint32_t blocks = (NSlinks + threadsPerBlock - 1) / threadsPerBlock;
  auto workDiv = alpaka::WorkDivMembers<alpaka::DimInt<1>, Idx>(
      alpaka::Vec<alpaka::DimInt<1>, Idx>::all(blocks),
      alpaka::Vec<alpaka::DimInt<1>, Idx>::all(threadsPerBlock),
      alpaka::Vec<alpaka::DimInt<1>, Idx>::all(1));

  UnpackKernel kernel;
    alpaka::exec<Acc>(
    q, workDiv, kernel,
    rawDev.view(), sizesDev.view(), offsDev.view(),
    modDev.view(), innerDev.view(), outerDev.view(),
    outDet.view(), outX.view(), outY.view(), outZ.view(),
    outW.view(), outSeed.view(), outMip.view(), outMType.view(),
    alpaka::getPtrNative(counter) // counter is a single scalar, ptr is fine
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

} // namespace ot

// ======== minimal test harness ========
int main() {
  // keep synthetic test minimal
  const std::size_t numSlinks = 1;
  std::vector<unsigned char> linear;
  std::vector<std::size_t> sizes(numSlinks, 0), offsets(numSlinks, 0);

  // header (4 words) + offset table (MODULES_PER_SLINK words) + chHeader + 1 payload line
  const int fakeWords = HEADER_N_LINES + MODULES_PER_SLINK + 2;
  linear.resize(fakeWords * N_BYTES_PER_WORD, 0);
  sizes[0] = linear.size();
  offsets[0] = 0;

  // offset table: channel 0 -> offset 0
  {
    auto p = reinterpret_cast<uint32_t*>(linear.data() + HEADER_N_LINES * N_BYTES_PER_WORD);
    p[0] = 0x00000000u; // ch0=0, ch1=0
  }

  // channel header at start of payload (1 strip, 0 pixel)w
  const size_t offsetTableStart = (HEADER_N_LINES + MODULES_PER_SLINK) * N_BYTES_PER_WORD;
  {
    const uint32_t chHeader = 0x00000001u;
    linear[offsetTableStart + 0] = static_cast<unsigned char>((chHeader >> 24) & 0xFF);
    linear[offsetTableStart + 1] = static_cast<unsigned char>((chHeader >> 16) & 0xFF);
    linear[offsetTableStart + 2] = static_cast<unsigned char>((chHeader >> 8) & 0xFF);
    linear[offsetTableStart + 3] = static_cast<unsigned char>(chHeader & 0xFF);
  }

  // maps for 1 slink × CICs_PER_SLINK
  const std::size_t M = numSlinks * CICs_PER_SLINK;
  std::vector<int>      modType(M, 0);
  std::vector<uint32_t> inner(M, 0), outer(M, 0);

  // channel 0 as 2S
  modType[0] = 1;
  inner[0] = 11;
  outer[0] = 22;

  ot::UnpackerDriver drv;
  auto out = drv.run(linear, sizes, offsets, modType, inner, outer);

  std::cout << "Decoded clusters: " << out.size() << "\n";
  for (std::size_t i = 0; i < out.size(); ++i) {
    auto const& c = out.clusters[i];
    std::cout << i << ": det=" << c.detId << " x=" << c.x << " y=" << c.y
              << " z=" << int(c.z) << " w=" << int(c.width)
              << " seed=" << int(c.isSeed) << " mip=" << int(c.mip)
              << " type=" << int(c.moduleType) << "\n";
  }
  return 0;
}



