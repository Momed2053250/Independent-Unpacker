#pragma once
#include <cstdint>
#include <cstddef>

namespace Phase2DAQ {
  static constexpr int DTC_DAQ_HEADER     = 0xFFFFFFFF;
  static constexpr int N_BITS_PER_WORD    = 32;
  static constexpr int N_BYTES_PER_WORD   = 4;

  static constexpr int L1ID_MAX_VALUE     = 0x1FF;
  static constexpr int L1ID_BITS          = 9;
  static constexpr int CIC_ERROR_BITS     = 9;
  static constexpr int MOD_TYPE_BITS      = 2;
  static constexpr int N_PIXEL_CLUSTER_BITS = 7;
  static constexpr int N_STRIP_CLUSTER_BITS = 7;

  static constexpr int CHIP_ID_MAX_VALUE  = 0x7;
  static constexpr int CHIP_ID_BITS       = 3;

  static constexpr int SCLUSTER_ADDRESS_2S_MAX_VALUE = 0x7F;
  static constexpr int SCLUSTER_ADDRESS_BITS_2S      = 8;
  static constexpr int SCLUSTER_ADDRESS_ONLY_BITS_2S = 7;

  static constexpr int SCLUSTER_ADDRESS_PS_MAX_VALUE = 0x7F;
  static constexpr int SCLUSTER_ADDRESS_BITS_PS      = 7;

  static constexpr int SCLUSTER_ADDRESS_MASK = 0x7F;
  static constexpr int IS_SEED_SENSOR_MASK   = 0x01;

  static constexpr int WIDTH_MAX_VALUE       = 0x7;
  static constexpr int WIDTH_BITS            = 3;

  static constexpr int MIP_BITS              = 1;
  static constexpr int MIP_BITS_MASK         = 0x1;
  static constexpr int PS_Z_BITS_MASK        = 0xF;

  static constexpr int SS_CLUSTER_BITS       = 14;
  static constexpr int PX_CLUSTER_BITS       = 17;
  static constexpr int Z_MAX_VALUE           = 0;

  static constexpr int CMSSW_TRACKER_ID      = 0;

  static constexpr int HEADER_N_LINES        = 4;
  static constexpr int OFFSET_BITS           = 16;

  static constexpr int CIC_ERROR_MASK        = 0x1FF;
  static constexpr int N_CLUSTER_MASK        = 0x7F;
  static constexpr int SS_CLUSTER_WORD_MASK  = 0x3FFF;
  static constexpr int PX_CLUSTER_WORD_MASK  = 0x1FFFF;
}

namespace Phase2Spec {
  static constexpr int SLINKS_PER_DTC          = 4;
  static constexpr int STRIPS_PER_CBC          = 127;
  static constexpr int CHANNELS_PER_CBC        = 254;
  static constexpr int STRIPS_PER_SSA          = 120;
  static constexpr int CHANNELS_PER_SSA        = 240;
  static constexpr int MODULES_PER_SLINK       = 18;
  static constexpr int CICs_PER_SLINK          = 36;
  static constexpr int MAX_DTC_ID              = 216;
  static constexpr int MIN_DTC_ID              = 1;
  static constexpr int MIN_SLINK_ID            = 0;
  static constexpr int MAX_SLINK_ID            = 3;
  static constexpr int TRACKER_HEADER          = 0;
}

using namespace Phase2DAQ;
using namespace Phase2Spec;