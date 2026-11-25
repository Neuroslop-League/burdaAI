// Group at most two requests per tick; share immutable weights and isolate KV contexts.
#pragma once
#include <array>
#include <future>

struct Slot {
    std::vector<llama_token> pending;
    int position = 0, remaining = 0;
};

static void serve_slots(std::array<llama_context *,2> contexts, int vocab_size, int eos) {
    std::array<Slot,2> slots;
    auto reset = [&](int s) {
        llama_kv_self_clear(contexts[s]);
        slots[s] = Slot{};
    };

    std::cout << "READY" << std::endl;
    for (std::string line; std::getline(std::cin,line);) {
        std::istringstream in(line);
        std::string verb; in >> verb;
        if (verb=="quit") break;
        if (verb=="tick") {
            std::array<std::future<int>,2> work;
            for(int s=0;s<2;s++) {
                if(!slots[s].remaining) continue;
                work[s]=std::async(std::launch::async,[&,s](){
                    auto & slot=slots[s];
                    auto batch=llama_batch_get_one(slot.pending.data(),int(slot.pending.size()));
                    std::vector<llama_pos> pos(batch.n_tokens);
                    for(auto & p:pos) p=slot.position++;
                    batch.pos=pos.data();
                    return llama_decode(contexts[s],batch);
                });
            }
            std::ostringstream reply;
            for(int s=0;s<2;s++) {
                if(!work[s].valid()) continue;
                if(work[s].get()!=0) {reset(s);reply<<" "<<s<<":error";continue;}
                float * logits=llama_get_logits_ith(contexts[s],-1);
                if (!logits || !std::all_of(logits,logits+vocab_size,[](float x){return std::isfinite(x);})) {
                    reset(s); reply << " " << s << ":error"; continue;
                }
                int id=int(std::max_element(logits,logits+vocab_size)-logits);
                bool done=(--slots[s].remaining==0 || id==eos);
                reply << " " << s << ":" << id << ":" << done;
                if(done) reset(s);
                else {slots[s].pending={id}; }
            }
            std::cout << "OK" << reply.str() << std::endl;
            continue;
        }
        int s=-1; in >> s;
        if(s<0 || s>=2) {std::cout<<"ERR slot"<<std::endl;continue;}
        if(verb=="cancel") {reset(s);std::cout<<"OK"<<std::endl;continue;}
        int limit=0; in>>limit;
        std::vector<llama_token> ids;
        int id;
        while(in>>id) ids.push_back(id);
        if(verb!="start" || slots[s].remaining || limit<=0 || limit>4096 || ids.empty() ||
           ids.size()+size_t(limit)>4096 || !in.eof() ||
           !std::all_of(ids.begin(),ids.end(),[&](int x){return x>=0 && x<vocab_size;})) {
            std::cout<<"ERR request"<<std::endl;continue;
        }
        slots[s]={std::move(ids),0,limit};
        std::cout<<"OK"<<std::endl;
    }

}
