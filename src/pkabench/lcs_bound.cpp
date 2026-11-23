// Exact longest-common-subsequence upper bound on identities in ANY alignment.
// Multiword bit-vector DP; no scores, gap rules or accepted matches are changed.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

struct Target {
    int length, percent;
    size_t words;
    std::vector<uint64_t> masks;
    Target(const char* text, int n, int p): length(n), percent(p),
        words((n+63)/64), masks(256*words,0) {
        for (int i=0;i<n;i++) masks[(unsigned char)text[i]*words+i/64] |= uint64_t(1)<<(i%64);
    }
};
struct Context {std::vector<Target> targets;};

extern "C" {
void* ka_lcs_create(const char** text, const int* lengths, const int* percents, int count) {
    try {
        auto* c=new Context();c->targets.reserve(count);
        for (int i=0;i<count;i++) c->targets.emplace_back(text[i],lengths[i],percents[i]);
        return c;
    } catch (...) {return nullptr;}
}
void ka_lcs_destroy(void* pointer) {delete static_cast<Context*>(pointer);}
int ka_lcs_bounds(void* pointer, const char* query, int n, int* output) {
    try {
        auto& targets=static_cast<Context*>(pointer)->targets;
        std::vector<uint64_t> state;
        for (size_t t=0;t<targets.size();t++) {
            auto& target=targets[t];
            const int required=(target.percent*std::max(n,target.length)+99)/100;
            if (std::min(n,target.length)<required) {output[t]=-1;continue;}
            state.assign(target.words,0);
            for (int i=0;i<n;i++) {
                auto* mask=&target.masks[(unsigned char)query[i]*target.words];
                uint64_t carry=1,borrow=0;
                for (size_t j=0;j<target.words;j++) {
                    uint64_t old=state[j],x=old|mask[j],y=(old<<1)|carry;
                    carry=old>>63;
                    __uint128_t sub=(__uint128_t)y+borrow;
                    uint64_t difference=(uint64_t)((__uint128_t)x-sub);
                    borrow=((__uint128_t)x<sub);
                    state[j]=x&~difference;
                }
            }
            int length=0;for (auto word:state) length+=__builtin_popcountll(word);
            output[t]=length;
        }
        return 0;
    } catch (...) {return -1;}
}
}
