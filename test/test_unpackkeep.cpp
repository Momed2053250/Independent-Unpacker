#include "Unpacker.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <sstream>
#include <cstdint>

using namespace std;

struct CablingEntry {
    uint32_t detId;
    uint32_t gbtId;  // GBT_CMSSW_IdPerDTC (0-35, channel within DTC)
    uint32_t dtcId;  // DTC_CMSSW_Id
};

struct ModuleEntry {
    uint32_t detId;
    int moduleType;  // 1=2S, 2=PS
};

// Read cabling map
map<uint32_t, CablingEntry> readCablingMap(const char* filename) {
    map<uint32_t, CablingEntry> result;
    ifstream file(filename);
    string line;
    getline(file, line); // skip header
    
    while (getline(file, line)) {
        vector<string> fields;
        stringstream ss(line);
        string field;
        while (getline(ss, field, ',')) {
            field.erase(0, field.find_first_not_of(" \t"));
            field.erase(field.find_last_not_of(" \t") + 1);
            fields.push_back(field);
        }
        
        if (fields.size() < 3) continue;
        
        try {
            uint32_t detId = stoul(fields[0]);
            uint32_t gbtId = stoul(fields[1]);
            uint32_t dtcId = stoul(fields[2]);
            
            result[detId] = {detId, gbtId, dtcId};
        } catch (...) {
            continue;
        }
    }
    file.close();
    cout << "Loaded " << result.size() << " cabling entries\n";
    return result;
}

// Read module types
map<uint32_t, ModuleEntry> readModuleTypes(const char* filename) {
    map<uint32_t, ModuleEntry> result;
    ifstream file(filename);
    string line;
    getline(file, line); // skip header
    
    while (getline(file, line)) {
        vector<string> fields;
        stringstream ss(line);
        string field;
        while (getline(ss, field, ',')) {
            field.erase(0, field.find_first_not_of(" \t"));
            field.erase(field.find_last_not_of(" \t") + 1);
            fields.push_back(field);
        }
        
        if (fields.size() < 14) continue;
        
        try {
            uint32_t detId = stoul(fields[0]);
            string typeStr = fields[13];
            int type = (typeStr.find("2S") != string::npos) ? 1 : 2;
            result[detId] = {detId, type};
        } catch (...) {
            continue;
        }
    }
    file.close();
    cout << "Loaded " << result.size() << " module types\n";
    return result;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <cabling_map.csv> <modules_to_dtc.csv>\n";
        return 1;
    }
    
    cout << "\n=== Phase 2 OT Unpacker Test ===\n\n";
    
    // Load data
    auto cabling = readCablingMap(argv[1]);
    auto modules = readModuleTypes(argv[2]);
    
    // Setup unpacker inputs
    const uint32_t NSlinks = (MAX_DTC_ID - MIN_DTC_ID + 1) * SLINKS_PER_DTC;
    const uint32_t totalChannels = NSlinks * CICs_PER_SLINK;
    
    vector<int> detIdxModuleType(totalChannels, 0);
    vector<uint32_t> innerDetId(totalChannels, 0);
    vector<uint32_t> outerDetId(totalChannels, 0);
    
    // Map cabling to arrays
    for (auto& [detId, cableEntry] : cabling) {
        if (modules.find(detId) == modules.end()) continue;
        
        uint32_t dtcId = cableEntry.dtcId;
        uint32_t gbtId = cableEntry.gbtId;
        int moduleType = modules[detId].moduleType;
        
        // Convert DTC ID to slink index
        // Assuming DTC IDs are contiguous starting from MIN_DTC_ID
        uint32_t dtcIdx = dtcId - MIN_DTC_ID;
        uint32_t slinkIdx = dtcIdx * SLINKS_PER_DTC;  // 1 slink per DTC
        
        // Channel index within slink
        uint32_t flatIdx = slinkIdx * CICs_PER_SLINK + gbtId;
        
        if (flatIdx < totalChannels) {
            detIdxModuleType[flatIdx] = moduleType;
            innerDetId[flatIdx] = detId;
            outerDetId[flatIdx] = detId;
        }
    }
    
    cout << "Mapped " << cabling.size() << " modules to channels\n";
    
    // Read binary FED data
    vector<unsigned char> linearRaw;
    vector<size_t> sizes(NSlinks, 0);
    vector<size_t> offsets(NSlinks, 0);
    
    const char* binPaths[] = {
        "../data/fed_raw_data.bin",
        "fed_raw_data.bin",
        "../../test/fed_raw_data.bin"
    };
    
    ifstream binFile;
    for (const auto& path : binPaths) {
        binFile.open(path, ios::binary);
        if (binFile.is_open()) {
            cout << "Opened binary file: " << path << "\n";
            break;
        }
    }
    
    if (binFile.is_open()) {
        map<uint32_t, vector<unsigned char>> fedData;
        
        while (!binFile.eof()) {
            uint32_t fedId, size;
            if (!binFile.read(reinterpret_cast<char*>(&fedId), sizeof(uint32_t))) break;
            if (!binFile.read(reinterpret_cast<char*>(&size), sizeof(uint32_t))) break;
            
            vector<unsigned char> data(size);
            binFile.read(reinterpret_cast<char*>(data.data()), size);
            
            fedData[fedId] = data;
        }
        binFile.close();
        
        cout << "Read " << fedData.size() << " FEDs\n";
        
        // Map FED ID to slink and build linearRaw
        size_t currentOffset = 0;
        for (auto& [fedId, data] : fedData) {
            // FED ID should correspond to DTC CMSSW ID
            uint32_t dtcId = fedId;
            uint32_t dtcIdx = dtcId - MIN_DTC_ID;
            uint32_t slinkIdx = dtcIdx * SLINKS_PER_DTC;
            
            if (slinkIdx < NSlinks) {
                offsets[slinkIdx] = currentOffset;
                sizes[slinkIdx] = data.size();
                linearRaw.insert(linearRaw.end(), data.begin(), data.end());
                currentOffset += data.size();
            }
        }
    }
    
    if (linearRaw.empty()) {
        cerr << "Error: No raw data loaded\n";
        return 1;
    }
    
    cout << "Raw data size: " << linearRaw.size() << " bytes\n";
    cout << "Number of SLinks: " << NSlinks << "\n";
    cout << "Total channels: " << totalChannels << "\n";
    
    // Run unpacker
    cout << "\nRunning unpacker...\n";
    ot::UnpackerDriver driver;
    auto clusters = driver.run(linearRaw, sizes, offsets, detIdxModuleType, innerDetId, outerDetId);
    
    cout << "\n=== Results ===\n";
    cout << "Total clusters decoded: " << clusters.size() << "\n";
    
    if (!clusters.clusters.empty()) {
        int count2S = 0, countPS = 0;
        for (const auto& c : clusters.clusters) {
            if (c.moduleType == 1) count2S++;
            else if (c.moduleType == 2) countPS++;
        }
        cout << "2S clusters: " << count2S << "\n";
        cout << "PS clusters: " << countPS << "\n";
        
        cout << "\nFirst 20 clusters:\n";
        for (size_t i = 0; i < std::min(size_t(20), clusters.size()); i++) {
            auto& c = clusters.clusters[i];
            cout << i << ": det=" << c.detId << " x=" << c.x << " y=" << c.y 
                 << " z=" << (int)c.z << " w=" << (int)c.width 
                 << " seed=" << (int)c.isSeed << " mip=" << (int)c.mip 
                 << " type=" << (int)c.moduleType << "\n";
        }
    }
    
    cout << "\n=== Test Complete ===\n";
    return 0;
}