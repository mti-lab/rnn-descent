#include <faiss/IndexHNSW.h>

#include <argparse/argparse.hpp>
#include <benches/datasets/DataLoader.hpp>
#include <benches/utils/Timer.hpp>
#include <benches/utils/graph_properties.hpp>
#include <benches/utils/metrics.hpp>
#include <iostream>
#include <nlohmann/json.hpp>

std::tuple<std::unique_ptr<faiss::IndexHNSW>, double> construct_hnsw(
    const DataLoader& data_loader, const nlohmann::json& parameters) {
    int d = data_loader.dim();
    int M = parameters["M"];
    auto index = std::make_unique<faiss::IndexHNSWFlat>(d, M);
    index->hnsw.efConstruction = parameters["efConstruction"];
    index->verbose = true;

    // train

    // add
    double construction_time_sec;
    {
        auto [nb, xb] = data_loader.load_base();

        Timer timer;
        index->add(nb, xb.get());
        construction_time_sec = timer.elapsed_ms() * 1e-3;
        std::cout << "Time = " << construction_time_sec << " [s]" << std::endl;
    }

    return {std::move(index), construction_time_sec};
}

nlohmann::json measure_search_performance(faiss::IndexHNSW& index,
                                          const DataLoader& data_loader) {
    using idx_t = faiss::idx_t;

    size_t d = data_loader.dim();
    auto [nq, xq] = data_loader.load_query();
    auto [k, gt] = data_loader.load_gt();

    nlohmann::json results;

    for (int efSearch : {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}) {
        for (bool search_bounded_queue : {false, true}) {
            index.hnsw.efSearch = efSearch;
            index.hnsw.search_bounded_queue = search_bounded_queue;

            auto [qps, r_at_1] = compute_qps_recall(index, nq, xq, k, gt);

            nlohmann::json result;
            result["efSearch"] = efSearch;
            result["search_bounded_queue"] = search_bounded_queue;
            result["qps"] = qps;
            result["r@1"] = r_at_1;
            results.push_back(result);
        }
    }

    return results;
}

nlohmann::json hnsw_properties(const faiss::IndexHNSW& index) {
    const int n = index.ntotal;
    const auto& neighbors = index.hnsw.neighbors;
    const auto get_range = [&](int u) -> std::tuple<int, int> {
        size_t left, right;
        index.hnsw.neighbor_range(u, 0, &left, &right);
        return {(int)left, (int)right};
    };
    return graph_properties(n, neighbors, get_range);
}

int main(int argc, char** argv) {
    argparse::ArgumentParser program("bench_hnsw");
    program.add_argument("--M").default_value(32).scan<'i', int>();
    program.add_argument("--efConstruction")
        .default_value(500)
        .scan<'i', int>();
    program.add_argument("--dataset").required();
    program.add_argument("--fn_result").required();

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    std::string dataset_name = program.get<std::string>("--dataset");
    DataLoader data_loader(dataset_name);

    nlohmann::json parameters;
    parameters["M"] = program.get<int>("--M");
    parameters["efConstruction"] = program.get<int>("--efConstruction");

    auto [index, construction_time] = construct_hnsw(data_loader, parameters);
    auto results = measure_search_performance(*index, data_loader);

    nlohmann::json output;
    output["dataset"] = dataset_name;
    output["method"] = "HNSW";
    output["parameters"] = parameters;
    output["construction_time"] = construction_time;
    output["search_performances"] = results;
    output["properties"] = hnsw_properties(*index);

    std::string fn_result = program.get<std::string>("--fn_result");
    std::ofstream ofs(fn_result);
    ofs << output.dump(4) << std::endl;
    std::cout << "Saved the result to \"" << fn_result << "\"" << std::endl;
}