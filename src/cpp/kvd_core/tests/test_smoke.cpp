#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include "kvd/api.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <limits.h>
#include <unistd.h>
#endif

TEST(KvdSmokeTest, CreateNullConfig) {
    EXPECT_EQ(kvd_create(nullptr), nullptr);
}

static std::filesystem::path find_repo_root() {
    std::error_code ec;
    std::filesystem::path p = std::filesystem::current_path(ec);
    if (ec) return {};
    for (int i = 0; i < 8; ++i) {
        auto candidate = p / "saved_models" / "lightgbm_model.txt";
        if (std::filesystem::exists(candidate, ec) && !ec) {
            return p;
        }
        if (!p.has_parent_path()) break;
        p = p.parent_path();
    }
    return {};
}

TEST(KvdSmokeTest, CreateWithModelPath) {
    auto root = find_repo_root();
    ASSERT_FALSE(root.empty());
    auto model_path = (root / "saved_models" / "lightgbm_model.txt").string();
    auto family_path =
        (root / "hdbscan_cluster_results" / "family_classifier.json").string();
    kvd_config cfg{};
    cfg.model_path = model_path.c_str();
    if (std::filesystem::exists(std::filesystem::path(family_path))) {
        cfg.family_classifier_json_path = family_path.c_str();
    }
    cfg.max_file_size = 64 * 1024;
    cfg.prediction_threshold = 0.98f;
    kvd_handle* h = kvd_create(&cfg);
    ASSERT_NE(h, nullptr);
    kvd_destroy(h);
}

static std::string current_module_path() {
#if defined(_WIN32)
    char buf[MAX_PATH + 1];
    DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    if (n == 0 || n >= MAX_PATH) return {};
    buf[n] = '\0';
    return std::string(buf);
#else
    char buf[PATH_MAX + 1];
    ssize_t n = readlink("/proc/self/exe", buf, PATH_MAX);
    if (n <= 0 || n >= PATH_MAX) return {};
    buf[n] = '\0';
    return std::string(buf);
#endif
}

TEST(KvdSmokeTest, ScanPathOnCurrentExe) {
    auto root = find_repo_root();
    ASSERT_FALSE(root.empty());
    auto model_path = (root / "saved_models" / "lightgbm_model.txt").string();
    auto family_path =
        (root / "hdbscan_cluster_results" / "family_classifier.json").string();

    kvd_config cfg{};
    cfg.model_path = model_path.c_str();
    if (std::filesystem::exists(std::filesystem::path(family_path))) {
        cfg.family_classifier_json_path = family_path.c_str();
    }
    cfg.max_file_size = 64 * 1024;
    cfg.prediction_threshold = 0.98f;

    kvd_handle* h = kvd_create(&cfg);
    ASSERT_NE(h, nullptr);

    auto target = current_module_path();
    ASSERT_FALSE(target.empty());

    char* out_json = nullptr;
    size_t out_len = 0;
    int rc = kvd_scan_path(h, target.c_str(), &out_json, &out_len);
    ASSERT_EQ(rc, 0);
    ASSERT_NE(out_json, nullptr);
    std::string s(out_json, out_len);
    kvd_free(out_json);
    kvd_destroy(h);

    EXPECT_NE(s.find("\"confidence\""), std::string::npos);
    EXPECT_NE(s.find("\"is_malware\""), std::string::npos);
}

TEST(KvdSmokeTest, DllExportsPresent) {
#if defined(_WIN32)
    HMODULE mod = LoadLibraryA("kvd.dll");
    ASSERT_NE(mod, nullptr);

    EXPECT_NE(GetProcAddress(mod, "kvd_create"), nullptr);
    EXPECT_NE(GetProcAddress(mod, "kvd_destroy"), nullptr);
    EXPECT_NE(GetProcAddress(mod, "kvd_scan_path"), nullptr);
    EXPECT_NE(GetProcAddress(mod, "kvd_scan_bytes"), nullptr);
    EXPECT_NE(GetProcAddress(mod, "kvd_free"), nullptr);
    EXPECT_NE(GetProcAddress(mod, "kvd_validate_models"), nullptr);

    FreeLibrary(mod);
#else
    GTEST_SKIP();
#endif
}

TEST(KvdSmokeTest, ScanPathMultithread) {
    auto root = find_repo_root();
    ASSERT_FALSE(root.empty());
    auto model_path = (root / "saved_models" / "lightgbm_model.txt").string();
    auto family_path =
        (root / "hdbscan_cluster_results" / "family_classifier.json").string();

    kvd_config cfg{};
    cfg.model_path = model_path.c_str();
    if (std::filesystem::exists(std::filesystem::path(family_path))) {
        cfg.family_classifier_json_path = family_path.c_str();
    }
    cfg.max_file_size = 64 * 1024;
    cfg.prediction_threshold = 0.98f;

    kvd_handle* h = kvd_create(&cfg);
    ASSERT_NE(h, nullptr);

    auto target = current_module_path();
    ASSERT_FALSE(target.empty());

    std::vector<std::thread> threads;
    std::vector<int> results(4, 1);
    for (int i = 0; i < 4; ++i) {
        int index = i;
        threads.emplace_back([&, index]() {
            char* out_json = nullptr;
            size_t out_len = 0;
            int rc = kvd_scan_path(h, target.c_str(), &out_json, &out_len);
            if (rc == 0 && out_json) {
                kvd_free(out_json);
            }
            results[index] = rc;
        });
    }
    for (auto& t : threads) {
        t.join();
    }
    kvd_destroy(h);

    for (int rc : results) {
        EXPECT_EQ(rc, 0);
    }
}

TEST(KvdSmokeTest, BoundaryConditions) {
    kvd_config cfg{};
    cfg.model_path = "non_existent_model_path.txt";
    cfg.max_file_size = 1;
    cfg.prediction_threshold = 2.0f;  // invalid threshold

    kvd_handle* h = kvd_create(&cfg);
    EXPECT_EQ(h, nullptr);

    char* out_error = nullptr;
    size_t out_len = 0;
    int rc = kvd_validate_models(&cfg, &out_error, &out_len);
    EXPECT_NE(rc, 0);
    if (out_error) {
        kvd_free(out_error);
    }
}
