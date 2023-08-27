#include <faiss/IndexNSG.h>

#include <argparse/argparse.hpp>
#include <benches/datasets/DataLoader.hpp>
#include <benches/utils/Timer.hpp>
#include <benches/utils/graph_properties.hpp>
#include <benches/utils/metrics.hpp>
#include <iostream>
#include <nlohmann/json.hpp>

std::tuple<std::unique_ptr<faiss::IndexNSG>, double> construct_nsg(
    const DataLoader& data_loader, const nlohmann::json& parameters) {
    int d = data_loader.dim();
    int R = parameters["R"];
    auto index = std::make_unique<faiss::IndexNSGFlat>(d, R);
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

nlohmann::json measure_search_performance(faiss::IndexNSG& index,
                                          const DataLoader& data_loader) {
    using idx_t = faiss::idx_t;

    size_t d = data_loader.dim();
    auto [nq, xq] = data_loader.load_query();
    auto [k, gt] = data_loader.load_gt();

    nlohmann::json results;

    for (int search_L : {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}) {
        index.nsg.search_L = search_L;

        auto [qps, r_at_1] = compute_qps_recall(index, nq, xq, k, gt);

        nlohmann::json result;
        result["search_L"] = search_L;
        result["qps"] = qps;
        result["r@1"] = r_at_1;
        results.push_back(result);
    }

    return results;
}

nlohmann::json nsg_properties(const faiss::IndexNSG& index) {
    const int n = index.ntotal;
    const int R = index.nsg.R;
    std::vector<int> neighbors(R * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < R; ++j) {
            neighbors[i * R + j] = index.nsg.final_graph->at(i, j);
        }
    }
    const auto get_range = [&](int u) -> std::tuple<int, int> {
        return {u * R, (u + 1) * R};
    };
    return graph_properties(n, neighbors, get_range);
}

int main(int argc, char** argv) {
    argparse::ArgumentParser program("bench_nsg");
    program.add_argument("--R").default_value(32).scan<'i', int>();
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
    parameters["R"] = program.get<int>("--R");

    auto [index, construction_time] = construct_nsg(data_loader, parameters);
    auto results = measure_search_performance(*index, data_loader);

    nlohmann::json output;
    output["dataset"] = dataset_name;
    output["method"] = "NSG";
    output["parameters"] = parameters;
    output["construction_time"] = construction_time;
    output["search_performances"] = results;
    output["properties"] = nsg_properties(*index);

    std::string fn_result = program.get<std::string>("--fn_result");
    std::ofstream ofs(fn_result);
    ofs << output.dump(4) << std::endl;
    std::cout << "Saved the result to \"" << fn_result << "\"" << std::endl;
}