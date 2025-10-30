#pragma once
#include <cstdint>
#include <vector>

struct ClusterProp {
  uint32_t detId;
  uint16_t x;
  uint16_t y;
  uint8_t  z;
  uint8_t  width;
  uint8_t  isSeed;
  uint8_t  mip;
  uint8_t  moduleType; // 1=2S, 2=PS
};

struct ClusterPropSoA {
  std::vector<ClusterProp> clusters;
  void reserve(std::size_t n) { clusters.reserve(n); }
  std::size_t size() const { return clusters.size(); }
  void push_back(ClusterProp const& c) { clusters.push_back(c); }
};
