// Quick test: verify zlink/compress.hpp compiles and works
#include <zlink/compress.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <cstring>

int main() {
    // Test 1: Small data (below threshold, should be raw)
    {
        std::vector<std::byte> small(100, std::byte{0x42});
        auto result = zlink::compress(small);
        if (result.comp_flag != zlink::comp_flag_raw) {
            std::cerr << "FAIL: small data should be raw, got comp_flag=" << (int)result.comp_flag << "\n";
            return 1;
        }
        std::cout << "Test 1 PASS: small data (" << small.size() << " bytes) -> raw ("
                  << result.data.size() << " bytes)\n";
    }

    // Test 2: Large compressible data (should compress)
    {
        std::vector<std::byte> large(100000, std::byte{0x00});
        // Add some pattern to make it compressible
        for (size_t i = 0; i < large.size(); i++) {
            large[i] = std::byte(i % 4);
        }
        auto result = zlink::compress(large);
        std::cout << "Test 2: large compressible data (" << large.size() << " bytes) -> comp_flag="
                  << (result.comp_flag == zlink::comp_flag_lz4 ? "LZ4" : "raw")
                  << " (" << result.data.size() << " bytes, ratio="
                  << (double)result.data.size() / large.size() << ")\n";

        // Decompress and verify
        auto decompressed = zlink::decompress(
            result.data, result.comp_flag, result.original_size);
        if (decompressed.size() != large.size()) {
            std::cerr << "FAIL: decompressed size mismatch: " << decompressed.size()
                      << " != " << large.size() << "\n";
            return 1;
        }
        if (std::memcmp(decompressed.data(), large.data(), large.size()) != 0) {
            std::cerr << "FAIL: decompressed data doesn't match original\n";
            return 1;
        }
        std::cout << "Test 2 PASS: round-trip verified\n";
    }

    // Test 3: Float data (like GPU tensors)
    {
        std::vector<float> floats(100000);
        for (size_t i = 0; i < floats.size(); i++) {
            floats[i] = static_cast<float>(i) * 0.001f;
        }
        auto* bytes = reinterpret_cast<const std::byte*>(floats.data());
        auto result = zlink::compress(std::span<const std::byte>(bytes, floats.size() * sizeof(float)));
        std::cout << "Test 3: float tensor (" << floats.size() * sizeof(float) << " bytes) -> "
                  << (result.comp_flag == zlink::comp_flag_lz4 ? "LZ4" : "raw")
                  << " (" << result.data.size() << " bytes, ratio="
                  << (double)result.data.size() / (floats.size() * sizeof(float)) << ")\n";

        if (result.comp_flag == zlink::comp_flag_lz4) {
            auto decompressed = zlink::decompress(result.data, result.comp_flag, result.original_size);
            if (decompressed.size() != floats.size() * sizeof(float)) {
                std::cerr << "FAIL: decompressed size mismatch\n";
                return 1;
            }
            if (std::memcmp(decompressed.data(), bytes, floats.size() * sizeof(float)) != 0) {
                std::cerr << "FAIL: decompressed data doesn't match\n";
                return 1;
            }
            std::cout << "Test 3 PASS: float round-trip verified\n";
        }
    }

    // Test 4: Random/incompressible data (should fallback to raw)
    {
        std::vector<std::byte> random_data(100000);
        std::mt19937 gen(42);
        std::uniform_int_distribution<int> dist(0, 255);
        for (auto& b : random_data) b = std::byte(dist(gen));

        auto result = zlink::compress(random_data);
        std::cout << "Test 4: random data (" << random_data.size() << " bytes) -> "
                  << (result.comp_flag == zlink::comp_flag_lz4 ? "LZ4" : "raw")
                  << " (" << result.data.size() << " bytes)\n";
    }

    std::cout << "\nAll compression tests passed!\n";
    return 0;
}
