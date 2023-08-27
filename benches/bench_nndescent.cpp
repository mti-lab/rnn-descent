#include <faiss/IndexNNDescent.h>

#include <argparse/argparse.hpp>
#include <benches/datasets/DataLoader.hpp>
#include <benches/utils/Timer.hpp>
#include <benches/utils/graph_properties.hpp>
#include <benches/utils/metrics.hpp>
#include <iostream>
#include <nlohmann/json.hpp>

std::tuple<std::unique_ptr<faiss::IndexNNDescent>, double> construct_nndescent(
    const DataLoader& data_loader, const nlohmann::json& parameters) {
    int d = data_loader.dim();
    int K = parameters["K"];
    auto index = std::make_unique<faiss::IndexNNDescentFlat>(d, K);
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

nlohmann::json measure_search_performance(faiss::IndexNNDescent& index,
                                          const DataLoader& data_loader) {
    using idx_t = faiss::idx_t;

    size_t d = data_loader.dim();
    auto [nq, xq] = data_loader.load_query();
    auto [k, gt] = data_loader.load_gt();

    nlohmann::json results;

    for (int search_L : {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}) {
        index.nndescent.search_L = search_L;

        auto [qps, r_at_1] = compute_qps_recall(index, nq, xq, k, gt);

        nlohmann::json result;
        result["search_L"] = search_L;
        result["qps"] = qps;
        result["r@1"] = r_at_1;
        results.push_back(result);
    }

    return results;
}

nlohmann::json nndescent_properties(const faiss::IndexNNDescent& index) {
    const int n = index.ntotal;
    const int K = index.nndescent.K;
    const auto& neighbors = index.nndescent.final_graph;
    const auto get_range = [&](int u) -> std::tuple<int, int> {
        return {u * K, (u + 1) * K};
    };
    return graph_properties(n, neighbors, get_range);
}

int main(int argc, char** argv) {
    argparse::ArgumentParser program("bench_nndescent");
    program.add_argument("--K").default_value(64).scan<'i', int>();
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
    parameters["K"] = program.get<int>("--K");

    auto [index, construction_time] =
        construct_nndescent(data_loader, parameters);
    auto results = measure_search_performance(*index, data_loader);

    nlohmann::json output;
    output["dataset"] = dataset_name;
    output["method"] = "NN-Descent";
    output["parameters"] = parameters;
    output["construction_time"] = construction_time;
    output["search_performances"] = results;
    output["properties"] = nndescent_properties(*index);

    std::string fn_result = program.get<std::string>("--fn_result");
    std::ofstream ofs(fn_result);
    ofs << output.dump(4) << std::endl;
    std::cout << "Saved the result to \"" << fn_result << "\"" << std::endl;
}