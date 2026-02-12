#include "../src/Unpacker.h"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <chrono>

struct FEDMetadata {
    int event;
    unsigned int fedId;
    size_t sizeBytes;
    size_t offsetBytes;
};
struct CablingEntry {
    unsigned int flatIdx;
    int dtcId;
    unsigned int slinkId;
    unsigned int channelId;
    int moduleType;
    uint32_t innerDetId;
    uint32_t outerDetId;
};
struct ExpectedCluster {
    int event;
    uint32_t detId;
    float x;
    float y;
    int width;
};
struct UnpackedCluster {
    uint32_t detId;
    uint16_t x;
    uint16_t y;
    uint8_t width;
};

std::vector<FEDMetadata> readMetadataCSV(const std::string& filename) {
    std::vector<FEDMetadata> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open: " + filename);
    }
    std::string line;
    std::getline(file, line);
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        FEDMetadata meta;
        char comma;
        ss >> meta.event >> comma 
           >> meta.fedId >> comma 
           >> meta.sizeBytes >> comma 
           >> meta.offsetBytes;
        data.push_back(meta);
    }

    std::cout << "Loaded " << data.size() << " FED metadata entries\n";
    return data;
}

std::vector<CablingEntry> readCablingCSV(const std::string& filename) {
    std::vector<CablingEntry> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open: " + filename);
    }
    std::string line;
    std::getline(file, line);
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        CablingEntry entry;
        char comma;
        ss >> entry.flatIdx >> comma
           >> entry.dtcId >> comma
           >> entry.slinkId >> comma
           >> entry.channelId >> comma
           >> entry.moduleType >> comma
           >> entry.innerDetId >> comma
           >> entry.outerDetId;
        data.push_back(entry);
    }

    std::cout << "Loaded " << data.size() << " cabling entries\n";
    return data;
}

std::vector<ExpectedCluster> readExpectedClustersCSV(const std::string& filename) {
    std::vector<ExpectedCluster> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open: " + filename);
    }
    std::string line;
    std::getline(file, line);
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        ExpectedCluster cluster;
        char comma;
        ss >> cluster.event >> comma
           >> cluster.detId >> comma
           >> cluster.x >> comma
           >> cluster.y >> comma
           >> cluster.width;
        data.push_back(cluster);
    }

    std::cout << "Loaded " << data.size() << " expected clusters\n";
    return data;
}

std::vector<unsigned char> readBinaryFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open: " + filename);
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<unsigned char> buffer(size);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        throw std::runtime_error("Failed to read: " + filename);
    }

    std::cout << "Loaded " << buffer.size() << " bytes of raw FED data\n";
    return buffer;
}

std::vector<UnpackedCluster> mergeConsecutiveClusters(const std::vector<UnpackedCluster>& unpacked) {
    if (unpacked.empty()) return unpacked;

    std::vector<UnpackedCluster> merged;

    for (size_t i = 0; i < unpacked.size(); ++i) {
        UnpackedCluster current = unpacked[i];

        while (i + 1 < unpacked.size() && 
               unpacked[i + 1].detId == current.detId &&
               unpacked[i + 1].y == current.y &&
               unpacked[i + 1].x == current.x + current.width) {
            current.width += unpacked[i + 1].width;
            ++i;
        }

        merged.push_back(current);
    }

    return merged;
}

bool clustersMatch(const ExpectedCluster& expected, const UnpackedCluster& unpacked, 
                   float xyTolerance = 0.0f) {
    if (expected.detId != unpacked.detId) return false;

    float dx = std::abs(expected.x - static_cast<float>(unpacked.x));
    float dy = std::abs(expected.y - static_cast<float>(unpacked.y));

    return (dx <= xyTolerance && dy <= xyTolerance);
}

struct EventResults {
    int eventNum;
    size_t expectedCount;
    size_t unpackedCount;
    size_t matched;
    double matchRate;
};

struct EventProcessResult {
    EventResults stats;
    std::vector<std::pair<size_t, size_t>> matches;
    std::vector<ExpectedCluster> expectedClusters;
    std::vector<UnpackedCluster> unpackedClusters;
};

EventProcessResult processEvent(
    int eventNum,
    const std::vector<FEDMetadata>& metadata,
    const std::vector<unsigned char>& rawData,
    const std::vector<int>& detIdxModuleType,
    const std::vector<uint32_t>& innerDetId,
    const std::vector<uint32_t>& outerDetId,
    const std::vector<ExpectedCluster>& expected,
    ot::UnpackerDriver& driver
) {
    const uint32_t NSlinks = (MAX_DTC_ID - MIN_DTC_ID + 1) * SLINKS_PER_DTC;
    std::vector<std::size_t> sizes(NSlinks, 0);
    std::vector<std::size_t> offsets(NSlinks, 0);
    std::vector<unsigned char> linearRaw;
    size_t currentOffset = 0;

    for (const auto& meta : metadata) {
        if (meta.event != eventNum) continue;

        const unsigned slinkIdx = meta.fedId - CMSSW_TRACKER_ID;
        if (slinkIdx >= NSlinks) continue;

        // Always set size; offsets must correspond to *linearRaw* layout.
        sizes[slinkIdx] = meta.sizeBytes;

        if (meta.sizeBytes == 0) {
            offsets[slinkIdx] = 0;
            continue;
        }

        const size_t start = meta.offsetBytes;
        const size_t end = start + meta.sizeBytes;

        if (end <= rawData.size()) {
            // Offset is where this fragment will start in linearRaw
            offsets[slinkIdx] = currentOffset;

            linearRaw.insert(linearRaw.end(),
                             rawData.begin() + start,
                             rawData.begin() + end);
            currentOffset += meta.sizeBytes;
        } else {
            // invalid slice -> ignore this fragment
            sizes[slinkIdx] = 0;
            offsets[slinkIdx] = 0;
        }
    }

    auto result = driver.run(linearRaw, sizes, offsets, detIdxModuleType, innerDetId, outerDetId);

    std::vector<UnpackedCluster> unpacked;
    for (const auto& c : result.clusters) {
        unpacked.push_back({c.detId, c.x, c.y, c.width});
    }

    std::sort(unpacked.begin(), unpacked.end(), [](const UnpackedCluster& a, const UnpackedCluster& b) {
        if (a.detId != b.detId) return a.detId < b.detId;
        if (a.y != b.y) return a.y < b.y;
        return a.x < b.x;
    });

    auto mergedUnpacked = mergeConsecutiveClusters(unpacked);

    std::vector<ExpectedCluster> eventExpected;
    for (const auto& cluster : expected) {
        if (cluster.event == eventNum) {
            eventExpected.push_back(cluster);
        }
    }

    std::set<size_t> matchedExpected;
    std::set<size_t> matchedUnpacked;
    const float tolerance = 0.0f; 

    std::vector<std::pair<size_t, size_t>> matches;

    for (size_t i = 0; i < mergedUnpacked.size(); ++i) {
        for (size_t j = 0; j < eventExpected.size(); ++j) {
            if (matchedExpected.count(j)) continue;

            if (clustersMatch(eventExpected[j], mergedUnpacked[i], tolerance)) {
                matchedExpected.insert(j);
                matchedUnpacked.insert(i);
                matches.push_back({j, i});
                break;
            }
        }
    }

    EventResults results;
    results.eventNum = eventNum;
    results.expectedCount = eventExpected.size();
    results.unpackedCount = mergedUnpacked.size();
    results.matched = matchedExpected.size();
    results.matchRate = eventExpected.empty() ? 0.0 : (100.0 * matchedExpected.size() / eventExpected.size());

    return {results, matches, eventExpected, mergedUnpacked};
}

int main() {
    auto startTime = std::chrono::high_resolution_clock::now();

    const std::string dataDir = "/home/momedmoh/data/newdata/output/Chronotestset/";

    std::vector<FEDMetadata> metadata;
    std::vector<CablingEntry> cabling;
    std::vector<ExpectedCluster> expected;
    std::vector<unsigned char> rawData;

    try {
        metadata = readMetadataCSV(dataDir + "phase2_metadata.csv");
        cabling = readCablingCSV(dataDir + "cabling_map.csv");
        expected = readExpectedClustersCSV(dataDir + "expected_clusters.csv");
        rawData = readBinaryFile(dataDir + "fed_raw_data.bin");
    } catch (const std::exception& e) {
        std::cerr << "ERROR loading files: " << e.what() << "\n";
        return 1;
    }

    if (metadata.empty() || cabling.empty() || expected.empty() || rawData.empty()) {
        std::cerr << "ERROR: One or more input files are empty\n";
        return 1;
    }

    const size_t M = (MAX_DTC_ID - MIN_DTC_ID + 1) * SLINKS_PER_DTC * CICs_PER_SLINK;
    std::vector<int> detIdxModuleType(M, 0);
    std::vector<uint32_t> innerDetId(M, 0);
    std::vector<uint32_t> outerDetId(M, 0);

    for (const auto& entry : cabling) {
        if (entry.flatIdx < M) {
            detIdxModuleType[entry.flatIdx] = entry.moduleType;
            innerDetId[entry.flatIdx] = entry.innerDetId;
            outerDetId[entry.flatIdx] = entry.outerDetId;
        }
    }

    std::set<int> uniqueEvents;
    for (const auto& meta : metadata) {
        uniqueEvents.insert(meta.event);
        if (uniqueEvents.size() >= 1) break;
    }

    std::cout << "Found " << uniqueEvents.size() << " events in metadata\n\n";

    std::cout << "========================================\n";
    std::cout << "GPU CHECK\n";
    std::cout << "========================================\n";
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    std::cout << "CUDA backend: ENABLED\n";
    auto testDev = alpaka::getDevByIdx(alpaka::Platform<Acc>{}, 0u);
    std::cout << "Device name: " << alpaka::getName(testDev) << "\n";
#else
    std::cout << "CPU backend only\n";
#endif
    std::cout << "========================================\n\n";

    std::cout << "========================================\n";
    std::cout << "Processing events (like produce loop)\n";
    std::cout << "========================================\n\n";

    ot::UnpackerDriver driver;
    std::vector<EventResults> allResults;

    std::ofstream csvFile("validation_results.csv");
    csvFile << "event,detId,expected_x,expected_y,expected_width,unpacked_x,unpacked_y,unpacked_width\n";

    for (int eventNum : uniqueEvents) {
        std::cout << "Processing event " << eventNum << "...\n";
        auto result = processEvent(eventNum, metadata, rawData, detIdxModuleType, 
                                   innerDetId, outerDetId, expected, driver);
        allResults.push_back(result.stats);

        for (const auto& match : result.matches) {
            const auto& exp = result.expectedClusters[match.first];
            const auto& unp = result.unpackedClusters[match.second];
            csvFile << eventNum << ","
                    << exp.detId << ","
                    << exp.x << ","
                    << exp.y << ","
                    << exp.width << ","
                    << unp.x << ","
                    << unp.y << ","
                    << static_cast<int>(unp.width) << "\n";
        }

        std::cout << "  Expected: " << result.stats.expectedCount 
                  << " | Unpacked: " << result.stats.unpackedCount
                  << " | Matched: " << result.stats.matched
                  << " | Rate: " << std::fixed << std::setprecision(1) << result.stats.matchRate << "%\n";
    }

    csvFile.close();
    std::cout << "\nValidation results saved to: validation_results.csv\n";

    std::cout << "\n========================================\n";
    std::cout << "Summary across all events\n";
    std::cout << "========================================\n\n";

    size_t totalExpected = 0;
    size_t totalUnpacked = 0;
    size_t totalMatched = 0;

    for (const auto& r : allResults) {
        totalExpected += r.expectedCount;
        totalUnpacked += r.unpackedCount;
        totalMatched += r.matched;
    }

    double overallRate = totalExpected == 0 ? 0.0 : (100.0 * totalMatched / totalExpected);

    std::cout << "Total expected clusters:  " << totalExpected << "\n";
    std::cout << "Total unpacked clusters:  " << totalUnpacked << "\n";
    std::cout << "Total matched:            " << totalMatched << "\n";
    std::cout << "Overall match rate:       " << std::fixed << std::setprecision(1) << overallRate << "%\n\n";

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endTime - startTime;

    std::cout << "========================================\n";
    std::cout << "Execution time: " << std::fixed << std::setprecision(2) << elapsed.count() << " seconds\n";
    std::cout << "========================================\n\n";

    std::cout << "========================================\n";
    if (overallRate >= 90.0) {
        std::cout << "RESULT: PASS (Match rate >= 90%)\n";
        std::cout << "========================================\n\n";
        return 0;
    } else {
        std::cout << "RESULT: FAIL (Match rate < 90%)\n";
        std::cout << "========================================\n\n";
        return 1;
    }
}

