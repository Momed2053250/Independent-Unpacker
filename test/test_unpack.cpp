#include "../src/Unpacker.h"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <set>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <unordered_map>

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

struct ExpectedSoA {
    int event;
    uint32_t detId;
    uint16_t x;
    uint16_t y;
    uint8_t  z;
    uint8_t  width;
    uint8_t  isSeed;
    uint8_t  mip;
    uint8_t  modType;
};

static inline std::string ensureTrailingSlash(std::string s) {
    if (!s.empty() && s.back() != '/') s.push_back('/');
    return s;
}

std::vector<FEDMetadata> readMetadataCSV(const std::string& filename) {
    std::vector<FEDMetadata> data;
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Cannot open: " + filename);
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
    if (!file.is_open()) throw std::runtime_error("Cannot open: " + filename);
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

static inline int toIntOrZero(std::string const& s) {
    if (s.empty()) return 0;
    return std::stoi(s);
}

std::vector<ExpectedSoA> readExpectedSoACSV(const std::string& filename) {
    std::vector<ExpectedSoA> data;
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Cannot open: " + filename);

    std::string line;
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::vector<std::string> f;
        f.reserve(9);
        std::string cur;
        std::stringstream ss(line);
        while (std::getline(ss, cur, ',')) f.push_back(cur);

        if (f.size() < 9) {
            throw std::runtime_error("Bad line (need 9 fields) in expected SoA CSV: " + line);
        }

        ExpectedSoA c{};
        c.event   = toIntOrZero(f[0]);
        c.detId   = static_cast<uint32_t>(std::stoul(f[1]));
        c.x       = static_cast<uint16_t>(toIntOrZero(f[2]));
        c.y       = static_cast<uint16_t>(toIntOrZero(f[3]));
        c.z       = static_cast<uint8_t>(toIntOrZero(f[4]));
        c.width   = static_cast<uint8_t>(toIntOrZero(f[5]));
        c.isSeed  = static_cast<uint8_t>(toIntOrZero(f[6]));
        c.mip     = static_cast<uint8_t>(toIntOrZero(f[7]));
        c.modType = static_cast<uint8_t>(toIntOrZero(f[8]));

        data.push_back(c);
    }

    std::cout << "Loaded " << data.size() << " expected SoA clusters\n";
    return data;
}

std::vector<unsigned char> readBinaryFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) throw std::runtime_error("Cannot open: " + filename);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<unsigned char> buffer(size);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size))
        throw std::runtime_error("Failed to read: " + filename);
    std::cout << "Loaded " << buffer.size() << " bytes of raw FED data\n";
    return buffer;
}

struct Key {
    uint32_t detId;
    uint16_t x;
    uint16_t y;
    uint8_t  z;
    uint8_t  width;
    uint8_t  isSeed;
    uint8_t  mip;
    uint8_t  modType;

    bool operator==(Key const& o) const {
        return detId==o.detId && x==o.x && y==o.y && z==o.z &&
               width==o.width && isSeed==o.isSeed && mip==o.mip && modType==o.modType;
    }
};

struct KeyHash {
    std::size_t operator()(Key const& k) const noexcept {
        std::size_t h = 1469598103934665603ull;
        auto mix = [&](std::size_t v) {
            h ^= v + 0x9e3779b97f4a7c15ull + (h<<6) + (h>>2);
        };
        mix(k.detId);
        mix(k.x);
        mix(k.y);
        mix(k.z);
        mix(k.width);
        mix(k.isSeed);
        mix(k.mip);
        mix(k.modType);
        return h;
    }
};

struct EventResults {
    int eventNum;
    size_t expectedCount;
    size_t unpackedCount;
    size_t matched;
    double matchRate;
};

static inline void dumpMismatchOnce(
    bool &alreadyDumped,
    int eventNum,
    std::vector<Key> const& missingFromUnpacked,
    std::vector<Key> const& extraInUnpacked
) {
    if (alreadyDumped) return;
    alreadyDumped = true;

    std::ofstream ue("unmatched_expected.csv");
    ue << "event,detId,x,y,z,width,isSeed,mip,modType\n";
    for (auto const& k : missingFromUnpacked) {
        ue << eventNum << "," << k.detId << "," << k.x << "," << k.y << "," << int(k.z) << ","
           << int(k.width) << "," << int(k.isSeed) << "," << int(k.mip) << "," << int(k.modType) << "\n";
    }
    ue.close();

    std::ofstream uu("unmatched_unpacked.csv");
    uu << "event,detId,x,y,z,width,isSeed,mip,modType\n";
    for (auto const& k : extraInUnpacked) {
        uu << eventNum << "," << k.detId << "," << k.x << "," << k.y << "," << int(k.z) << ","
           << int(k.width) << "," << int(k.isSeed) << "," << int(k.mip) << "," << int(k.modType) << "\n";
    }
    uu.close();
}

EventResults processEventFullSoA(
    int eventNum,
    const std::vector<FEDMetadata>& metadata,
    const std::vector<unsigned char>& rawData,
    const std::vector<int>& detIdxModuleType,
    const std::vector<uint32_t>& innerDetId,
    const std::vector<uint32_t>& outerDetId,
    const std::vector<ExpectedSoA>& expectedAll,
    ot::UnpackerDriver& driver,
    bool &dumpedFirstMismatch
) {
    const uint32_t NSlinks = (MAX_DTC_ID - MIN_DTC_ID + 1) * SLINKS_PER_DTC;
    std::vector<std::size_t> sizes(NSlinks, 0);
    std::vector<std::size_t> offsets(NSlinks, 0);
    std::vector<unsigned char> linearRaw;
    linearRaw.reserve(5'000'000);

    size_t currentOffset = 0;

    for (const auto& meta : metadata) {
        if (meta.event != eventNum) continue;

        const unsigned slinkIdx = meta.fedId - CMSSW_TRACKER_ID;
        if (slinkIdx >= NSlinks) continue;

        sizes[slinkIdx] = meta.sizeBytes;

        if (meta.sizeBytes == 0) {
            offsets[slinkIdx] = 0;
            continue;
        }

        const size_t start = meta.offsetBytes;
        const size_t end = start + meta.sizeBytes;

        if (end <= rawData.size()) {
            offsets[slinkIdx] = currentOffset;
            linearRaw.insert(linearRaw.end(), rawData.begin() + start, rawData.begin() + end);
            currentOffset += meta.sizeBytes;
        } else {
            sizes[slinkIdx] = 0;
            offsets[slinkIdx] = 0;
        }
    }

    for (uint32_t i = 0; i < NSlinks; ++i) {
        if (sizes[i] == 0) continue;
        if (offsets[i] + sizes[i] > linearRaw.size()) {
            throw std::runtime_error("Invalid offsets/sizes for linearRaw");
        }
    }

    auto result = driver.run(linearRaw, sizes, offsets, detIdxModuleType, innerDetId, outerDetId);

    // ---- ADDED: dump standalone SoA output (ALL columns) ----
    // Writes/append per event, header only once.
    {
        static bool wroteHeader = false;
        std::ofstream soacsv("../plot/standalone_clusters_soa.csv", wroteHeader ? std::ios::app : std::ios::out);
        if (!wroteHeader) {
            soacsv << "event,detId,x,y,z,width,isSeed,mip,modType\n";
            wroteHeader = true;
        }
        for (auto const& c : result.clusters) {
            soacsv << eventNum << ","
                   << c.detId << ","
                   << c.x << ","
                   << c.y << ","
                   << int(c.z) << ","
                   << int(c.width) << ","
                   << int(c.isSeed) << ","
                   << int(c.mip) << ","
                   << int(c.moduleType) << "\n";
        }
    }
    // ---- END ADDED BLOCK ----

    std::unordered_map<Key, int, KeyHash> expCount;
    size_t expectedCount = 0;
    for (auto const& e : expectedAll) {
        if (e.event != eventNum) continue;
        Key k{e.detId, e.x, e.y, e.z, e.width, e.isSeed, e.mip, e.modType};
        expCount[k] += 1;
        expectedCount++;
    }

    size_t matched = 0;
    size_t unpackedCount = 0;
    std::vector<Key> missingFromUnpacked;
    std::vector<Key> extraInUnpacked;

    for (auto const& c : result.clusters) {
        Key k{c.detId, c.x, c.y, c.z, c.width, c.isSeed, c.mip, c.moduleType};
        unpackedCount++;
        auto it = expCount.find(k);
        if (it != expCount.end() && it->second > 0) {
            it->second -= 1;
            matched++;
        } else {
            extraInUnpacked.push_back(k);
        }
    }

    for (auto const& kv : expCount) {
        for (int n = 0; n < kv.second; ++n) missingFromUnpacked.push_back(kv.first);
    }

    if (!missingFromUnpacked.empty() || !extraInUnpacked.empty()) {
        dumpMismatchOnce(dumpedFirstMismatch, eventNum, missingFromUnpacked, extraInUnpacked);
    }

    EventResults r;
    r.eventNum = eventNum;
    r.expectedCount = expectedCount;
    r.unpackedCount = unpackedCount;
    r.matched = matched;
    r.matchRate = (expectedCount == 0) ? 0.0 : (100.0 * double(matched) / double(expectedCount));
    return r;
}

int main() {
    auto startTime = std::chrono::high_resolution_clock::now();

    std::string dataDir = ensureTrailingSlash("/home/momedmoh/data/newdata/output/Chronotestset/");

    std::vector<FEDMetadata> metadata;
    std::vector<CablingEntry> cabling;
    std::vector<ExpectedSoA> expectedSoA;
    std::vector<unsigned char> rawData;

    try {
        metadata    = readMetadataCSV(dataDir + "phase2_metadata.csv");
        cabling     = readCablingCSV(dataDir + "cabling_map.csv");
        expectedSoA = readExpectedSoACSV(dataDir + "expected_clusters_soa.csv");
        rawData     = readBinaryFile(dataDir + "fed_raw_data.bin");
    } catch (const std::exception& e) {
        std::cerr << "ERROR loading files: " << e.what() << "\n";
        return 1;
    }

    if (metadata.empty() || cabling.empty() || expectedSoA.empty() || rawData.empty()) {
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
    for (const auto& meta : metadata) uniqueEvents.insert(meta.event);

    if (uniqueEvents.size() > 100) {
        auto it = uniqueEvents.begin();
        std::advance(it, 100);
        uniqueEvents.erase(it, uniqueEvents.end());
    }

    std::cout << "Processing " << uniqueEvents.size() << " events (limited to 100) in metadata\n\n";

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
    std::cout << "Processing events (FULL SoA validation)\n";
    std::cout << "========================================\n\n";

    ot::UnpackerDriver driver;
    std::vector<EventResults> all;

    bool dumpedFirstMismatch = false;

    for (int eventNum : uniqueEvents) {
        std::cout << "Processing event " << eventNum << "...\n";
        auto r = processEventFullSoA(eventNum, metadata, rawData, detIdxModuleType,
                                     innerDetId, outerDetId, expectedSoA, driver, dumpedFirstMismatch);
        all.push_back(r);

        std::cout << "  Expected: " << r.expectedCount
                  << " | Unpacked: " << r.unpackedCount
                  << " | Matched: " << r.matched
                  << " | Rate: " << std::fixed << std::setprecision(1) << r.matchRate << "%\n";
    }

    size_t totalE=0, totalU=0, totalM=0;
    for (auto const& r : all) { totalE += r.expectedCount; totalU += r.unpackedCount; totalM += r.matched; }

    std::cout << "\n========================================\n";
    std::cout << "Summary\n";
    std::cout << "========================================\n\n";
    std::cout << "Total expected clusters:  " << totalE << "\n";
    std::cout << "Total unpacked clusters:  " << totalU << "\n";
    std::cout << "Total matched:            " << totalM << "\n";
    std::cout << "Overall match rate:       " << std::fixed << std::setprecision(3)
              << (totalE ? 100.0 * double(totalM) / double(totalE) : 0.0) << "%\n\n";

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endTime - startTime;
    std::cout << "========================================\n";
    std::cout << "Execution time: " << std::fixed << std::setprecision(2) << elapsed.count() << " seconds\n";
    std::cout << "========================================\n\n";

    if (totalE == totalM && totalE == totalU) {
        std::cout << "RESULT: PASS (100% full SoA match)\n";
        return 0;
    } else {
        std::cout << "RESULT: FAIL (mismatch; first mismatch dumped to unmatched_expected.csv + unmatched_unpacked.csv)\n";
        return 1;
    }
}
