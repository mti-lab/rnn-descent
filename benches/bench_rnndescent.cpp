#include <rnn-descent/IndexRNNDescent.h>

#include <argparse/argparse.hpp>
#include <benches/datasets/DataLoader.hpp>
#include <benches/utils/Timer.hpp>
#include <benches/utils/metrics.hpp>
#include <benches/utils/graph_properties.hpp>
#include <iostream>
#include <nlohmann/json.hpp>

std::tuple<std::unique_ptr<rnndescent::IndexRNNDescent>, double>
construct_rnn_descent(const DataLoader& data_loader,
                      const nlohmann::json& parameters) {
    int d = data_loader.dim();
    auto index = std::make_unique<rnndescent::IndexRNNDescent>(d);
    index->rnndescent.S = parameters["S"];
    index->rnndescent.R = parameters["R"];
    index->rnndescent.T1 = parameters["T1"];
    index->rnndescent.T2 = parameters["T2"];
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

nlohmann::json measure_search_performance(rnndescent::IndexRNNDescent& index,
                                          const DataLoader& data_loader) {
    using idx_t = faiss::idx_t;

    size_t d = data_loader.dim();
    auto [nq, xq] = data_loader.load_query();
    auto [k, gt] = data_loader.load_gt();

    nlohmann::json results;

    const int infty = 1000000;
    for (int search_L : {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}) {
        for (int K0 : {32, 48, 64, infty}) {
            index.rnndescent.search_L = search_L;
            index.rnndescent.K0 = K0;

            auto [qps, r_at_1] = compute_qps_recall(index, nq, xq, k, gt);

            nlohmann::json result;
            result["search_L"] = search_L;
            result["K0"] = K0;
            result["qps"] = qps;
            result["r@1"] = r_at_1;
            results.push_back(result);
        }
    }

    return results;
}

nlohmann::json rnndescent_properties(const rnndescent::IndexRNNDescent& index) {
    const int n = index.ntotal;
    const auto& neighbors = index.rnndescent.final_graph;
    const auto& offsets = index.rnndescent.offsets;
    const auto get_range = [&](int u) -> std::tuple<int, int> {
        return {offsets[u], offsets[u + 1]};
    };
    return graph_properties(n, neighbors, get_range);
}

int main(int argc, char** argv) {
    argparse::ArgumentParser program("bench_rnndescent");
    program.add_argument("--S").default_value(20).scan<'i', int>();
    program.add_argument("--R").default_value(96).scan<'i', int>();
    program.add_argument("--T1").default_value(4).scan<'i', int>();
    program.add_argument("--T2").default_value(15).scan<'i', int>();
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
    parameters["S"] = program.get<int>("--S");
    parameters["R"] = program.get<int>("--R");
    parameters["T1"] = program.get<int>("--T1");
    parameters["T2"] = program.get<int>("--T2");

    auto [index, construction_time] =
        construct_rnn_descent(data_loader, parameters);
    auto results = measure_search_performance(*index, data_loader);

    nlohmann::json output;
    output["dataset"] = dataset_name;
    output["method"] = "RNN-Descent";
    output["parameters"] = parameters;
    output["construction_time"] = construction_time;
    output["search_performances"] = results;
    output["properties"] = rnndescent_properties(*index);

    std::string fn_result = program.get<std::string>("--fn_result");
    std::ofstream ofs(fn_result);
    ofs << output.dump(4) << std::endl;
    std::cout << "Saved the result to \"" << fn_result << "\"" << std::endl;
}