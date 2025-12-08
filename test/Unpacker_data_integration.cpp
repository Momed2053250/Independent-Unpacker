#include <catch2/catch_test_macros.hpp>
#include <TFile.h>
#include <TTree.h>
#include <vector>
#include <iostream>

#include "Unpacker.h"
#include "specs.h"

TEST_CASE("Unpacker data-driven integration test (no comparison)", "[unpacker][integration]") {
    using namespace ot;

    // Open the ROOT file
    TFile file("PackedData.root", "READ");
    REQUIRE(file.IsOpen());

    // Access Events tree
    TTree* events = (TTree*) file.Get("Events");
    REQUIRE(events != nullptr);

    // FEDRawDataCollection payload branches:
    std::vector<unsigned char>* rawData = nullptr;
    std::vector<unsigned int>* rawSizes = nullptr;

    events->SetBranchAddress("FEDRawDataCollection_Packer__PACKONLY.obj.data_.data_", &rawData);
    events->SetBranchAddress("FEDRawDataCollection_Packer__PACKONLY.obj.data_.@size", &rawSizes);

    // Read first event only for now
    events->GetEntry(0);

    REQUIRE(rawData != nullptr);
    REQUIRE(rawSizes != nullptr);
    REQUIRE(rawSizes->size() > 0);

    const std::size_t numSlinks = rawSizes->size();

    // Construct sizes and offsets as required by UnpackerDriver
    std::vector<std::size_t> sizes(numSlinks), offsets(numSlinks);

    std::size_t sum = 0;
    for (std::size_t i = 0; i < numSlinks; i++) {
        sizes[i] = (*rawSizes)[i];
        offsets[i] = sum;
        sum += sizes[i];
    }

    // Linear raw buffer
    std::vector<unsigned char> linear(sum);
    memcpy(linear.data(), rawData->data(), sum);

    // Stub detector lookup tables
    // We will fill these later when cabling/geometry is available.
    std::vector<int>      moduleType(numSlinks * CICs_PER_SLINK, 0);
    std::vector<uint32_t> innerDetId(numSlinks * CICs_PER_SLINK, 0);
    std::vector<uint32_t> outerDetId(numSlinks * CICs_PER_SLINK, 0);

    UnpackerDriver drv;

    // Just check we can run without crash
    auto result = drv.run(linear, sizes, offsets, moduleType, innerDetId, outerDetId);

    // Basic sanity checks:
    REQUIRE(result.size() >= 0);   // Unpacker returned a cluster container
    // We cannot compare values yet, because no reference mapping is assigned.
}
