#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <regex>
#include <string>
#include <tuple>

namespace {

size_t load_dim(const std::filesystem::path& filename) {
    std::ifstream ifs(filename, std::ios::binary);
    assert(ifs.is_open());

    int32_t d;
    ifs.read((char*)&d, sizeof(d));
    return static_cast<size_t>(d);
}

// Load n float vectors from "filename".
// If n < 0, load all vectors.
std::tuple<size_t, std::unique_ptr<float[]>> load_fvecs(
    const std::filesystem::path& filename, long n) {
    std::ifstream ifs(filename, std::ios::binary);
    assert(ifs.is_open());

    int d;
    ifs.read((char*)&d, sizeof(d));
    if (n < 0) {
        ifs.seekg(0, std::ios_base::end);
        n = ifs.tellg();
        assert(n % ((d + 1) * sizeof(float)) == 0);
        n /= (d + 1) * sizeof(float);
    }
    ifs.seekg(0, std::ios_base::beg);

    std::unique_ptr<float[]> x(new float[d * n]);

    constexpr size_t batch_size = 8192;
    std::unique_ptr<float[]> buf(new float[(d + 1) * batch_size]);
    size_t pos_x = 0;
    for (size_t i = 0; i < n; i += batch_size) {
        size_t nvecs = std::min(batch_size, n - i);
        ifs.read((char*)buf.get(), sizeof(float) * (d + 1) * nvecs);
        size_t pos_buf = 1;
        for (size_t j = 0; j < nvecs; ++j) {
            std::memmove(&x[pos_x], &buf[pos_buf], d * sizeof(float));
            pos_x += d;
            pos_buf += d + 1;
        }
    }

    return {n, std::move(x)};
}

// Load n int64_t vectors from "filename".
// If n < 0, load all vectors.
std::tuple<size_t, std::unique_ptr<int64_t[]>> load_ivecs(
    const std::filesystem::path& filename, long n) {
    std::ifstream ifs(filename, std::ios::binary);
    assert(ifs.is_open());

    int d;
    ifs.read((char*)&d, sizeof(d));
    if (n < 0) {
        ifs.seekg(0, std::ios_base::end);
        n = ifs.tellg();
        assert(n % ((d + 1) * sizeof(float)) == 0);
        n /= (d + 1) * sizeof(float);
    }
    ifs.seekg(0, std::ios_base::beg);

    std::unique_ptr<int64_t[]> x(new int64_t[d * n]);

    constexpr size_t batch_size = 8192;
    std::unique_ptr<int[]> buf(new int[(d + 1) * batch_size]);
    size_t pos_x = 0;
    for (size_t i = 0; i < n; i += batch_size) {
        size_t nvecs = std::min(batch_size, n - i);
        ifs.read((char*)(buf.get()), sizeof(int) * (d + 1) * nvecs);
        size_t pos_buf = 1;
        for (size_t j = 0; j < nvecs; ++j) {
            std::memmove(&x[pos_x], &buf[pos_buf], d * sizeof(float));
            for (size_t k = 0; k < d; ++k) {
                x[pos_x + k] = static_cast<int64_t>(buf[pos_buf + k]);
            }
            pos_x += d;
            pos_buf += d + 1;
        }
    }

    return {d, std::move(x)};
}

// Load n float vectors from "filename".
// If n < 0, load all vectors.
std::tuple<size_t, std::unique_ptr<float[]>> load_bvecs(
    const std::filesystem::path& filename, long n) {
    std::ifstream ifs(filename, std::ios::binary);
    assert(ifs.is_open());

    int d;
    ifs.read((char*)&d, sizeof(d));
    int bvec_size = sizeof(unsigned char) * d + sizeof(int);
    if (n < 0) {
        ifs.seekg(0, std::ios_base::end);
        n = ifs.tellg();
        assert(n % bvec_size == 0);
        n /= bvec_size;
    }
    ifs.seekg(0, std::ios_base::beg);

    std::unique_ptr<float[]> x(new float[d * n]);

    constexpr size_t batch_size = 8192;
    std::unique_ptr<unsigned char[]> buf(
        new unsigned char[bvec_size * batch_size]);
    size_t pos_x = 0;
    for (size_t i = 0; i < n; i += batch_size) {
        size_t nvecs = std::min(batch_size, n - i);
        ifs.read((char*)(buf.get()), bvec_size * nvecs);
        size_t pos_buf = sizeof(int);
        for (size_t j = 0; j < nvecs; ++j) {
            std::memmove(&x[pos_x], &buf[pos_buf], d * sizeof(unsigned char));
            for (size_t k = 0; k < d; ++k) {
                x[pos_x + k] = static_cast<float>(buf[pos_buf + k]);
            }
            pos_x += d;
            pos_buf += bvec_size;
        }
    }

    return {n, std::move(x)};
}

}  // namespace

std::tuple<size_t, std::unique_ptr<float[]>> load_fbvecs(
    const std::filesystem::path& filename, long n) {
    auto ext = filename.extension();

    if (ext != ".fvecs" && ext != ".bvecs") {
        assert(!"unsupported format");
    }

    if (ext == ".bvecs") {
        return load_bvecs(filename, n);
    }

    return load_fvecs(filename, n);
}

class DataLoader {
    long n;
    size_t d;
    std::filesystem::path train_path;
    std::filesystem::path base_path;
    std::filesystem::path query_path;
    std::filesystem::path gt_path;

   public:
    DataLoader(
        const std::string& dataset_name,
        const std::string& settings_path = "./benches/datasets/settings.json");

    inline size_t dim() const { return d; }

    std::tuple<size_t, std::unique_ptr<float[]>> load_train() const {
        return load_fbvecs(train_path, -1);
    }

    std::tuple<size_t, std::unique_ptr<float[]>> load_base() const {
        return load_fbvecs(base_path, n);
    }

    std::tuple<size_t, std::unique_ptr<float[]>> load_query() const {
        return load_fbvecs(query_path, -1);
    }

    std::tuple<size_t, std::unique_ptr<int64_t[]>> load_gt() const {
        return load_ivecs(gt_path, -1);
    }

    void dump();
};

DataLoader::DataLoader(const std::string& dataset_name,
                       const std::string& settings_path) {
    std::ifstream ifs(settings_path);
    auto settings = nlohmann::json::parse(ifs);
    auto dataset_settings = settings["datasets"][dataset_name];

    std::filesystem::path root_dir = settings["root_dir"];
    std::filesystem::path dataset_dir =
        root_dir / dataset_settings["dataset_dir"];
    train_path = dataset_dir / dataset_settings["train_path"];
    base_path = dataset_dir / dataset_settings["base_path"];
    query_path = dataset_dir / dataset_settings["query_path"];
    gt_path = dataset_dir / dataset_settings["gt_path"];

    n = -1;
    if (dataset_settings.find("nvecs") != dataset_settings.end()) {
        n = dataset_settings["nvecs"];
    }
    d = load_dim(base_path);
}

void DataLoader::dump() {
    std::cout << "{\n";
    std::cout << "\t"
              << "n : " << n << "\n";
    std::cout << "\t"
              << "d : " << d << "\n";
    std::cout << "\t"
              << "train_path : " << train_path << "\n";
    std::cout << "\t"
              << "base_path : " << base_path << "\n";
    std::cout << "\t"
              << "query_path : " << query_path << "\n";
    std::cout << "\t"
              << "gt_path : " << gt_path << "\n";
    std::cout << "}\n";
}