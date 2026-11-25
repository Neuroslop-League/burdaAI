// Narrow process adapter for the pinned llama.cpp C API; tokenization stays in Rust.
#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
static double seconds(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b-a).count();
}
struct Evidence { ggml_backend_dev_t device = nullptr; bool observed = false; };
static bool observe(ggml_tensor * node, bool ask, void * userdata) {
    auto & e = *static_cast<Evidence *>(userdata);
    if (ask) {
        if (e.observed || node->op != GGML_OP_MUL_MAT || !node->buffer) return false;
        auto type = ggml_backend_buffer_get_type(node->buffer);
        return !ggml_backend_buft_is_host(type) && ggml_backend_buft_get_device(type) == e.device;
    }
    e.observed = true;
    std::cerr << "completed_gpu_matmul=" << node->name << " device="
              << ggml_backend_dev_name(e.device) << "\n";
    return true;
}
int main(int argc, char ** argv) try {
    if (argc != 7 && argc != 8) throw std::runtime_error("MODEL LAYERS STEPS TOKENIZER_ID csv-IDs generate|benchmark [last-logits.f32]");
    const auto cold = Clock::now();
    int layers = std::stoi(argv[2]), steps = std::stoi(argv[3]);
    bool bench = std::string(argv[6]) == "benchmark";
    if ((!bench && std::string(argv[6])!="generate") || layers<0 || layers>20 || steps<0)
        throw std::runtime_error("invalid profile");
    std::vector<llama_token> ids;
    std::stringstream input(argv[5]);
    for (std::string token; std::getline(input, token, ',');) ids.push_back(std::stoi(token));
    if (ids.empty() || ids.size()>4096 || (bench && (ids.size()!=256 || steps!=64)))
        throw std::runtime_error("invalid prompt or benchmark dimensions");
    const size_t first_len = ids.size();
    steps = std::min(steps, int(4096 - ids.size()));
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = layers;
    Evidence evidence;
    ggml_backend_dev_t devices[2] = {};
    if (layers) {
        for (size_t i=0;i<ggml_backend_dev_count();++i) {
            auto dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev)!=GGML_BACKEND_DEVICE_TYPE_GPU) continue;
            if (devices[0]) throw std::runtime_error("expose exactly one GPU");
            devices[0] = dev;
        }
        if (!devices[0]) throw std::runtime_error("requested GPU unavailable");
        if (std::string(ggml_backend_dev_name(devices[0])).find("Vulkan") != 0)
            throw std::runtime_error("selected GPU is not a Vulkan device");
        evidence.device = devices[0];
        std::cerr << "gpu=" << ggml_backend_dev_description(devices[0]) << " gpu_layers=" << layers << "\n";
    }
    mp.devices = devices;
    auto model = llama_model_load_from_file(argv[1],mp);
    if (!model) throw std::runtime_error("model load failed");
    char identity[160] = {};
    if (bench) {
        char format[32] = {};
        if (llama_model_meta_val_str(model,"general.file_type",format,sizeof(format))<0 ||
            std::string(format)!=std::to_string(LLAMA_FTYPE_MOSTLY_Q4_0))
            throw std::runtime_error("1 GiB benchmark requires Q4_0");
    }
    if (llama_model_meta_val_str(model,"burda.tokenizer_id",identity,sizeof(identity))<0 ||
        std::string(identity)!=argv[4]) throw std::runtime_error("incompatible tokenizer");
    auto vocab = llama_model_get_vocab(model);
    int size = llama_vocab_n_tokens(vocab);
    for (auto id : ids) if (id<0 || id>=size) throw std::runtime_error("invalid input ID");
    auto cp = llama_context_default_params();
    cp.n_ctx = std::max(32, int(ids.size())+steps);
    cp.type_k = cp.type_v = GGML_TYPE_F32;
    cp.n_batch = cp.n_ctx;
    cp.n_ubatch = std::min(64u, cp.n_batch);
    const bool cache = std::getenv("BURDA_KV") && std::string(std::getenv("BURDA_KV"))=="on";
    const int threads = std::getenv("BURDA_THREADS") ? std::stoi(std::getenv("BURDA_THREADS")) : 4;
    if (threads<1 || threads>64) throw std::runtime_error("invalid thread count");
    cp.n_threads = cp.n_threads_batch = threads;
    cp.offload_kqv = layers>0;
    if (layers) { cp.cb_eval=observe; cp.cb_eval_user_data=&evidence; }
    auto ctx = llama_init_from_model(model,cp);
    if (!ctx) throw std::runtime_error("context allocation failed");
    const auto warm=Clock::now();
    auto first=warm;
    int emitted=0;
    for (int i=0;i<steps;++i) {
        if (!cache || !i) llama_kv_self_clear(ctx);
        const int offset = cache && i ? int(ids.size())-1 : 0;
        auto batch = llama_batch_get_one(ids.data()+offset, int(ids.size())-offset);
        std::vector<llama_pos> positions(batch.n_tokens);
        for(int j=0;j<batch.n_tokens;++j) positions[j]=offset+j;
        batch.pos=positions.data();
        if (llama_decode(ctx,batch)) throw std::runtime_error("decode failed");
        llama_synchronize(ctx);
        auto logits = llama_get_logits_ith(ctx,-1);
        if (!logits || !std::all_of(logits,logits+size,[](float x){return std::isfinite(x);}))
            throw std::runtime_error("invalid logits");
        auto id = static_cast<llama_token>(std::max_element(logits,logits+size)-logits);
        if (!i) first=Clock::now();
        if (const char * trace = std::getenv("BURDA_TRACE_ALL")) {
            std::ofstream file(std::string(trace)+"/"+std::to_string(i)+".f32",std::ios::binary);
            if (!file.write(reinterpret_cast<const char*>(logits),size*sizeof(float))) throw std::runtime_error("trace failed");
        }
        if (argc==8 && !i) {
            std::ofstream file(argv[7],std::ios::binary);
            if (!file.write(reinterpret_cast<const char*>(logits),size*sizeof(float))) throw std::runtime_error("trace write failed");
        }
        if (!bench && id==llama_vocab_eos(vocab)) break;
        ids.push_back(id);
        std::cout << id << "\n";
        emitted++;
    }
    const auto end=Clock::now();
    if (layers && steps && !evidence.observed) throw std::runtime_error("GPU execution unconfirmed");
    std::cerr << "prompt=" << first_len << " emitted=" << emitted << " cold_ttft_s=" << seconds(cold,first)
              << " warm_ttft_s=" << seconds(warm,first) << " tokens_per_second_after_first="
              << (emitted>1 ? (emitted-1)/seconds(first,end) : 0) << "\n";
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
} catch (const std::exception &e) {std::cerr << e.what() << "\n"; return 1;}
