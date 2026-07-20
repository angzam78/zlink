// zlink/tests/test_rpc.cpp — Unit tests for core RPC components

#include <zlink/config.hpp>
#include <zlink/transport.hpp>
#include <zlink/ptr_map.hpp>

#include <iostream>
#include <cassert>
#include <cstring>

using namespace zlink;

// ── Test helpers ───────────────────────────────────────────────────────
static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) \
    do { \
        ++tests_run; \
        std::cout << "  TEST " << #name << " ... "; \
    } while(0)

#define PASS() \
    do { \
        ++tests_passed; \
        std::cout << "PASSED\n"; \
    } while(0)

#define FAIL(msg) \
    std::cout << "FAILED: " << msg << "\n"

#define ASSERT_EQ(a, b) \
    do { \
        if ((a) != (b)) { FAIL(#a " != " #b); return; } \
    } while(0)

#define ASSERT_TRUE(x) \
    do { \
        if (!(x)) { FAIL(#x " is false"); return; } \
    } while(0)

// ── Frame serialization tests ──────────────────────────────────────────
void test_frame_serialize_empty() {
    TEST(frame_serialize_empty);
    frame f;
    f.call_id = 42;
    f.type = frame_type::request;

    auto wire = f.serialize();
    ASSERT_EQ(wire.size(), frame_header_size);  // No payload

    frame out;
    auto consumed = frame::deserialize(wire, out);
    ASSERT_TRUE(consumed > 0);
    ASSERT_EQ(out.call_id, 42u);
    ASSERT_EQ(out.type, frame_type::request);
    ASSERT_EQ(out.payload.size(), 0u);

    PASS();
}

void test_frame_serialize_with_payload() {
    TEST(frame_serialize_with_payload);
    frame f;
    f.call_id = 123;
    f.type = frame_type::response;
    f.payload = {std::byte{0x01}, std::byte{0x02}, std::byte{0x03}, std::byte{0x04}};

    auto wire = f.serialize();
    ASSERT_EQ(wire.size(), frame_header_size + 4);

    frame out;
    auto consumed = frame::deserialize(wire, out);
    ASSERT_TRUE(consumed > 0);
    ASSERT_EQ(out.call_id, 123u);
    ASSERT_EQ(out.type, frame_type::response);
    ASSERT_EQ(out.payload.size(), 4u);
    ASSERT_EQ(out.payload[0], std::byte{0x01});
    ASSERT_EQ(out.payload[3], std::byte{0x04});

    PASS();
}

void test_frame_deserialize_incomplete() {
    TEST(frame_deserialize_incomplete);
    std::vector<std::byte> partial(3, std::byte{0x00});  // Too short for header

    frame out;
    auto consumed = frame::deserialize(partial, out);
    ASSERT_EQ(consumed, 0u);  // Should return 0 for incomplete data

    PASS();
}

// ── Pointer map tests ──────────────────────────────────────────────────
void test_ptr_map_basic() {
    TEST(ptr_map_basic);
    ptr_map pm;

    auto local = pm.map(0xDEADBEEF);
    ASSERT_TRUE(local != 0);

    auto remote = pm.to_remote(local);
    ASSERT_TRUE(remote.has_value());
    ASSERT_EQ(*remote, 0xDEADBEEF);

    auto local2 = pm.to_local(0xDEADBEEF);
    ASSERT_TRUE(local2.has_value());
    ASSERT_EQ(*local2, local);

    PASS();
}

void test_ptr_map_shadow_region() {
    TEST(ptr_map_shadow_region);
    ptr_map pm;
    pm.set_shadow_region(0x7F0000000000ULL, 0x100000000ULL);

    auto local1 = pm.map(0xAAAA);
    auto local2 = pm.map(0xBBBB);

    // Both should be in the shadow region
    ASSERT_TRUE(local1 >= 0x7F0000000000ULL);
    ASSERT_TRUE(local2 >= 0x7F0000000000ULL);
    ASSERT_TRUE(local1 != local2);  // Different mappings

    // Reverse lookup should work
    auto r1 = pm.to_remote(local1);
    ASSERT_TRUE(r1.has_value());
    ASSERT_EQ(*r1, 0xAAAA);

    PASS();
}

void test_ptr_map_unmap() {
    TEST(ptr_map_unmap);
    ptr_map pm;

    auto local = pm.map(0x1234);
    ASSERT_TRUE(pm.is_local(local));
    ASSERT_TRUE(pm.is_remote(0x1234));

    pm.unmap(local);
    ASSERT_TRUE(!pm.is_local(local));
    ASSERT_TRUE(!pm.is_remote(0x1234));

    auto remote = pm.to_remote(local);
    ASSERT_TRUE(!remote.has_value());

    PASS();
}

void test_ptr_map_size() {
    TEST(ptr_map_size);
    ptr_map pm;

    ASSERT_EQ(pm.size(), 0u);
    pm.map(1);
    pm.map(2);
    pm.map(3);
    ASSERT_EQ(pm.size(), 3u);

    pm.clear();
    ASSERT_EQ(pm.size(), 0u);

    PASS();
}

// ── Config tests ───────────────────────────────────────────────────────
void test_error_codes() {
    TEST(error_codes);
    ASSERT_TRUE(ok(error_code::ok));
    ASSERT_TRUE(!ok(error_code::transport));
    ASSERT_TRUE(failed(error_code::unknown_function));
    ASSERT_TRUE(!failed(error_code::ok));

    PASS();
}

// ── Main ───────────────────────────────────────────────────────────────
int main() {
    std::cout << "zlink unit tests\n"
              << "================\n\n";

    test_frame_serialize_empty();
    test_frame_serialize_with_payload();
    test_frame_deserialize_incomplete();
    test_ptr_map_basic();
    test_ptr_map_shadow_region();
    test_ptr_map_unmap();
    test_ptr_map_size();
    test_error_codes();

    std::cout << "\n" << tests_passed << "/" << tests_run << " tests passed.\n";
    return (tests_passed == tests_run) ? 0 : 1;
}
