
// #pragma GCC optimize("O3")
// #pragma GCC optimize("unroll-loops")
// #pragma GCC optimize("Ofast")
#include <bits/stdc++.h>
// INCLUDE <experimental/simd>
#include <atcoder/all>
// #include <ext/pb_ds/assoc_container.hpp> //
// clangで提出する場合はコメントアウト #include <immintrin.h> #include
// <sys/time.h> #include <x86intrin.h>
using namespace std;
constexpr bool DEBUG = true;
constexpr double TIME_LIMIT = 1.95;

// Macros
#define el '\n'
#define all(v) (v).begin(), (v).end()
using i8 = int8_t;
using u8 = uint8_t;
using i16 = int16_t;
using u16 = uint16_t;
using i32 = int32_t;
using u32 = uint32_t;
using i64 = int64_t;
using u64 = uint64_t;
using f32 = float;
using f64 = double;
#define rep(i, n) for (i64 i = 0; i < (i64)(n); i++)
template <class T> using min_queue = priority_queue<T, vector<T>, greater<T>>;
template <class T> using max_queue = priority_queue<T>;
struct uint64_hash {
    static inline uint64_t rotr(uint64_t x, unsigned k) {
        return (x >> k) | (x << (8U * sizeof(uint64_t) - k));
    }
    static inline uint64_t hash_int(uint64_t x) noexcept {
        auto h1 = x * (uint64_t)(0xA24BAED4963EE407);
        auto h2 = rotr(x, 32U) * (uint64_t)(0x9FB21C651E98DF25);
        auto h = rotr(h1 + h2, 32U);
        return h;
    }
    size_t operator()(uint64_t x) const {
        static const uint64_t FIXED_RANDOM =
            std::chrono::steady_clock::now().time_since_epoch().count();
        return hash_int(x + FIXED_RANDOM);
    }
};
// template <typename K, typename V,
//           typename Hash = uint64_hash> // clangで提出する場合はコメントアウト
// using hash_map =
//     __gnu_pbds::gp_hash_table<K, V,
//                               Hash>; // clangで提出する場合はコメントアウト
// template <typename K,
//           typename Hash = uint64_hash> // clangで提出する場合はコメントアウト
// using hash_set = hash_map<K, __gnu_pbds::null_type,
//                           Hash>; // clangで提出する場合はコメントアウト

// Constant
const double pi = 3.141592653589793238;
const i32 inf32 = 1073741823;
const i64 inf64 = 1LL << 60;
const string ABC = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const string abc = "abcdefghijklmnopqrstuvwxyz";
const int MOD = 998244353;
const array<int, 8> dx = {0, 0, -1, 1, -1, -1, 1, 1};
const array<int, 8> dy = {-1, 1, 0, 0, -1, 1, -1, 1};

// Printing
template <class T> void print_collection(ostream &out, T const &x);
template <class T, size_t... I>
void print_tuple(ostream &out, T const &a, index_sequence<I...>);
namespace std {
template <class... A> ostream &operator<<(ostream &out, tuple<A...> const &x) {
    print_tuple(out, x, index_sequence_for<A...>{});
    return out;
}
template <class... A> ostream &operator<<(ostream &out, pair<A...> const &x) {
    print_tuple(out, x, index_sequence_for<A...>{});
    return out;
}
template <class A, size_t N>
ostream &operator<<(ostream &out, array<A, N> const &x) {
    print_collection(out, x);
    return out;
}
template <class A> ostream &operator<<(ostream &out, vector<A> const &x) {
    print_collection(out, x);
    return out;
}
template <class A> ostream &operator<<(ostream &out, deque<A> const &x) {
    print_collection(out, x);
    return out;
}
template <class A> ostream &operator<<(ostream &out, multiset<A> const &x) {
    print_collection(out, x);
    return out;
}
template <class A, class B>
ostream &operator<<(ostream &out, multimap<A, B> const &x) {
    print_collection(out, x);
    return out;
}
template <class A> ostream &operator<<(ostream &out, set<A> const &x) {
    print_collection(out, x);
    return out;
}
template <class A, class B>
ostream &operator<<(ostream &out, map<A, B> const &x) {
    print_collection(out, x);
    return out;
}
template <class A, class B>
ostream &operator<<(ostream &out, unordered_set<A> const &x) {
    print_collection(out, x);
    return out;
}
} // namespace std
template <class T, size_t... I>
void print_tuple(ostream &out, T const &a, index_sequence<I...>) {
    using swallow = int[];
    out << '(';
    (void)swallow{0, (void(out << (I == 0 ? "" : ", ") << get<I>(a)), 0)...};
    out << ')';
}
template <class T> void print_collection(ostream &out, T const &x) {
    int f = 0;
    out << '[';
    for (auto const &i : x) {
        out << (f++ ? "," : "");
        out << i;
    }
    out << "]";
}
// Random
struct RNG {
    uint64_t s[2];
    RNG(u64 seed) { reset(seed); }
    RNG() { reset(time(0)); }
    using result_type = u32;
    static constexpr u32 min() { return numeric_limits<u32>::min(); }
    static constexpr u32 max() { return numeric_limits<u32>::max(); }
    u32 operator()() { return randomInt32(); }
    static __attribute__((always_inline)) inline uint64_t rotl(const uint64_t x,
                                                               int k) {
        return (x << k) | (x >> (64 - k));
    }
    inline void reset(u64 seed) {
        struct splitmix64_state {
            u64 s;
            u64 splitmix64() {
                u64 result = (s += 0x9E3779B97f4A7C15);
                result = (result ^ (result >> 30)) * 0xBF58476D1CE4E5B9;
                result = (result ^ (result >> 27)) * 0x94D049BB133111EB;
                return result ^ (result >> 31);
            }
        };
        splitmix64_state sm{seed};
        s[0] = sm.splitmix64();
        s[1] = sm.splitmix64();
    }
    uint64_t next() {
        const uint64_t s0 = s[0];
        uint64_t s1 = s[1];
        const uint64_t result = rotl(s0 * 5, 7) * 9;
        s1 ^= s0;
        s[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16); // a, b
        s[1] = rotl(s1, 37);                   // c
        return result;
    }
    inline u32 randomInt32() { return next(); }
    inline u64 randomInt64() { return next(); }
    inline u32 random32(u32 r) { return (((u64)randomInt32()) * r) >> 32; }
    inline u64 random64(u64 r) { return randomInt64() % r; }
    inline u32 randomRange32(u32 l, u32 r) { return l + random32(r - l + 1); }
    inline u64 randomRange64(u64 l, u64 r) { return l + random64(r - l + 1); }
    inline double randomDouble() {
        return (double)randomInt32() / 4294967296.0;
    }
    inline double randomDoubleOpen01() {
        return (randomInt32() + 1.0) / 4294967297.0;
    }
    inline float randomFloat() { return (float)randomInt32() / 4294967296.0; }
    inline double randomRangeDouble(double l, double r) {
        return l + randomDouble() * (r - l);
    }
    inline double randomGaussian(double mean = 0.0, double stddev = 1.0) {
        double u1 = randomDoubleOpen01();
        double u2 = randomDouble();
        double z = sqrt(-2.0 * log(u1)) * cos(2.0 * 3.141592653589793238 * u2);
        return mean + stddev * z;
    }
    template <class T> void shuffle(vector<T> &v) {
        i32 sz = v.size();
        for (i32 i = sz; i > 1; i--) {
            i32 p = random32(i);
            swap(v[i - 1], v[p]);
        }
    }
    template <class T> void shuffle(T *fr, T *to) {
        i32 sz = distance(fr, to);
        for (int i = sz; i > 1; i--) {
            int p = random32(i);
            swap(fr[i - 1], fr[p]);
        }
    }
    template <class T> inline int sample_index(vector<T> const &v) {
        return random32(v.size());
    }
    template <class T> inline T sample(vector<T> const &v) {
        return v[sample_index(v)];
    }
    struct AliasTable {
        int n;
        vector<u32> thresh;
        vector<int> alias;

        AliasTable() : n(0) {}

        template <class T> void build(const vector<T> &weights) {
            n = weights.size();
            thresh.resize(n);
            alias.resize(n);
            if constexpr (DEBUG) {
                assert(n > 0);
            }
            if (n == 0)
                return;

            double sum = 0;
            for (auto &w : weights) {
                sum += (double)w;
            }
            if constexpr (DEBUG) {
                assert(sum > 0.0);
            }

            vector<double> prob(n);
            for (int i = 0; i < n; i++)
                prob[i] = (double)weights[i] / sum * n;

            vector<int> small, large;
            small.reserve(n);
            large.reserve(n);
            for (int i = 0; i < n; i++) {
                if (prob[i] < 1.0)
                    small.push_back(i);
                else
                    large.push_back(i);
            }

            while (!small.empty() && !large.empty()) {
                int s = small.back();
                small.pop_back();
                int l = large.back();
                large.pop_back();
                thresh[s] = (u32)(prob[s] * 4294967296.0);
                alias[s] = l;
                prob[l] -= (1.0 - prob[s]);
                if (prob[l] < 1.0)
                    small.push_back(l);
                else
                    large.push_back(l);
            }
            while (!large.empty()) {
                int i = large.back();
                thresh[i] = ~(u32)0;
                alias[i] = i;
                large.pop_back();
            }
            while (!small.empty()) {
                int i = small.back();
                thresh[i] = ~(u32)0;
                alias[i] = i;
                small.pop_back();
            }
        }
    };

    inline int choices(const AliasTable &table) {
        int i = random32(table.n);
        u32 r = randomInt32();
        return r < table.thresh[i] ? i : table.alias[i];
    }
} rng;
// Timer
struct timer {
    chrono::high_resolution_clock::time_point t_begin;
    timer() { t_begin = chrono::high_resolution_clock::now(); }
    void reset() { t_begin = chrono::high_resolution_clock::now(); }
    float elapsed() const {
        return chrono::duration<float>(chrono::high_resolution_clock::now() -
                                       t_begin)
            .count();
    }
};
// Util
template <class T> T &smin(T &x, T const &y) {
    x = min(x, y);
    return x;
}
template <class T> T &smax(T &x, T const &y) {
    x = max(x, y);
    return x;
}
template <typename T> int sgn(T val) {
    if (val < 0)
        return -1;
    if (val > 0)
        return 1;
    return 0;
}
static inline string int_to_string(int val, int digits = 0) {
    string s = to_string(val);
    reverse(begin(s), end(s));
    while ((int)s.size() < digits)
        s.push_back('0');
    reverse(begin(s), end(s));
    return s;
}
// Debug
static inline void debug() { cerr << "\n"; }
template <class T, class... V> void debug(T const &t, V const &...v) {
    if constexpr (DEBUG) {
        cerr << t;
        if (sizeof...(v)) {
            cerr << ", ";
        }
        debug(v...);
    }
}
// Bits
__attribute__((always_inline)) inline u64 bit(u64 x) { return 1ull << x; }
__attribute__((always_inline)) inline void setbit(u64 &a, u32 b,
                                                  u64 value = 1) {
    a = (a & ~bit(b)) | (value << b);
}
__attribute__((always_inline)) inline u64 getbit(u64 a, u32 b) {
    return (a >> b) & 1;
}
__attribute__((always_inline)) inline u64 lsb(u64 a) {
    return __builtin_ctzll(a);
}
__attribute__((always_inline)) inline int msb(uint64_t bb) {
    return __builtin_clzll(bb) ^ 63;
}
struct Init {
    Init() {
        ios::sync_with_stdio(0);
        cin.tie(0);
    }
} init;

// ==========================================
// Problem-specific constants
// ==========================================
constexpr int n = 100;
constexpr int k = 8;
constexpr int h = 50;
constexpr int w = 50;
constexpr int t = 2500;
constexpr array<int, 4> di = {-w, w, -1, 1};
const string udlr = "UDLR";

// ==========================================
// Input & State
// ==========================================
struct Input {
    vector<string> grids;

    void input() {
        int _n, _k, _h, _w, _t;
        cin >> _n >> _k >> _h >> _w >> _t;
        assert(_n == n && _k == k && _h == h && _w == w && _t == t);

        grids.resize(n);
        for (int i = 0; i < n; ++i) {
            grids[i].reserve(h * w);
            for (int j = 0; j < h; ++j) {
                string row;
                cin >> row;
                grids[i] += row;
            }
        }
    }
};

int evaluate_grid(const string &grid) {
    return 2 * count(grid.begin(), grid.end(), 'x') +
           count(grid.begin(), grid.end(), '#');
}

array<int, k> select_grids(const Input &input) {
    vector<pair<int, int>> costs(n);
    for (int i = 0; i < n; ++i) {
        costs[i] = {evaluate_grid(input.grids[i]), i};
    }
    partial_sort(costs.begin(), costs.begin() + k, costs.end());

    array<int, k> ret;
    for (int i = 0; i < k; ++i) {
        ret[i] = costs[i].second;
    }
    return ret;
}

// ==========================================
// Beam Search (with hash deduplication)
// ==========================================
namespace beam_search {

// ハッシュによる重複除去を行うかどうか
constexpr bool USE_HASH_DEDUP = true;

// 正規-逆ガンマ分布によるベイズ推定
struct GaussInverseGamma {
    double mu, lambda, alpha, beta;

    GaussInverseGamma() : mu(0), lambda(1), alpha(1), beta(1) {}
    GaussInverseGamma(double mu, double lambda, double alpha, double beta)
        : mu(mu), lambda(lambda), alpha(alpha), beta(beta) {}

    static GaussInverseGamma
    from_pseudo_observation(double mean, double std_dev, int pseudo_count) {
        double variance = std_dev * std_dev;
        double precision = 1.0 / variance;
        double l = (double)pseudo_count;
        double a = (double)(pseudo_count * 2);
        double b = a / precision;
        return GaussInverseGamma(mean, l, a, b);
    }

    void update(double x) {
        double new_mu = (x + lambda * mu) / (lambda + 1.0);
        double new_lambda = lambda + 1.0;
        double new_alpha = alpha + 0.5;
        double dev2 = (x - mu) * (x - mu);
        double new_beta = beta + 0.5 * (lambda * dev2) / (lambda + 1.0);
        mu = new_mu;
        lambda = new_lambda;
        alpha = new_alpha;
        beta = new_beta;
    }

    pair<double, double> expected() const {
        double exp_precision = alpha / beta;
        double exp_std_dev = sqrt(1.0 / exp_precision);
        return {mu, exp_std_dev};
    }

    double get_pseudo_observation_count() const { return lambda; }

    void set_pseudo_observation_count(double count) {
        lambda = count;
        alpha = count * 0.5;
    }
};

// ベイズ推定によりビーム幅を動的に決定するクラス
struct BayesianBeamWidthSuggester {
    GaussInverseGamma dist;
    double time_limit_sec;
    int current_turn;
    int max_turn;
    int warmup_turn;
    int max_memory_turn;
    size_t min_beam_width;
    size_t max_beam_width;
    size_t current_beam_width;
    float start_time;
    float last_time;

    BayesianBeamWidthSuggester(int max_turn, int warmup_turn,
                               double time_limit_sec,
                               size_t standard_beam_width,
                               size_t min_beam_width, size_t max_beam_width,
                               timer &t)
        : time_limit_sec(time_limit_sec), current_turn(0), max_turn(max_turn),
          warmup_turn(warmup_turn), min_beam_width(min_beam_width),
          max_beam_width(max_beam_width), current_beam_width(0) {
        double mean_sec =
            time_limit_sec / ((double)max_turn * standard_beam_width);
        double stddev_sec = 0.2 * mean_sec;
        dist =
            GaussInverseGamma::from_pseudo_observation(mean_sec, stddev_sec, 3);
        max_memory_turn = max_turn / 5;
        start_time = t.elapsed();
        last_time = start_time;
    }

    size_t suggest(timer &t) {
        if (current_turn >= warmup_turn && current_beam_width > 0) {
            float now = t.elapsed();
            double elapsed = (double)(now - last_time);
            double elapsed_per_beam = elapsed / current_beam_width;
            dist.update(elapsed_per_beam);

            if (dist.get_pseudo_observation_count() >=
                (double)max_memory_turn) {
                dist.set_pseudo_observation_count((double)max_memory_turn);
            }
        }

        last_time = t.elapsed();

        double remaining_turn = (double)(max_turn - current_turn);
        double elapsed_time = (double)t.elapsed();
        double remaining_time = time_limit_sec - elapsed_time;

        auto [mean, std_dev] = dist.expected();
        double variance = std_dev * std_dev;

        double mean_remaining = remaining_turn * mean;
        double variance_remaining = remaining_turn * variance;
        double std_dev_remaining = sqrt(variance_remaining);

        constexpr double SIGMA_COEF = 3.0;
        double needed_time_per_width =
            mean_remaining + SIGMA_COEF * std_dev_remaining;

        size_t beam_width;
        if (remaining_time <= 0) {
            beam_width = min_beam_width;
        } else if (needed_time_per_width <= 0) {
            beam_width = max_beam_width;
        } else {
            beam_width = (size_t)(remaining_time / needed_time_per_width);
        }
        beam_width = max(min_beam_width, min(beam_width, max_beam_width));

        current_beam_width = beam_width;
        current_turn++;
        return beam_width;
    }
};

// ビームサーチの設定
struct Config {
    int max_turn;
    int expected_turn; // 終了目安ターン数（動的調整で残りターン数の計算に使用）
    bool dynamic_beam; // ビーム幅の動的調整を行うか
    double time_limit; // 動的調整時の終了目安時間(秒)
    size_t
        max_beam_width; // 最大ビーム幅（動的調整off時はこれをビーム幅として使用）
    size_t
        initial_beam_width; // 動的調整時の初期ビーム幅（standard_beam_widthとして兼用）
    size_t min_beam_width;      // 最小ビーム幅
    int warmup_turn;            // ウォームアップターン数
    size_t tour_capacity;       // 雑に大きくて良い -> 10^7くらい
    uint32_t hash_map_capacity; // ハッシュマップの容量 -> ビーム幅 * 派生先の数
                                // の16倍くらい

    // 現在のビーム幅を返す（動的調整offならmax_beam_width）
    size_t beam_width() const {
        return dynamic_beam ? current_beam_width_ : max_beam_width;
    }

    // 動的調整用の内部状態
    mutable size_t current_beam_width_ = 0;
};

// 連想配列
// Keyにハッシュ関数を適用しない
// open addressing with linear probing
// unordered_mapよりも速い
// nは格納する要素数(ビーム幅 * 派生先の数)よりも16倍ほど大きくする
template <class Key, class T> struct HashMap {
  public:
    explicit HashMap(uint32_t n) {
        if (n % 2 == 0) {
            ++n;
        }
        n_ = n;
        valid_.resize(n_, false);
        data_.resize(n_);
    }

    // 戻り値
    // - 存在するならtrue、存在しないならfalse
    // - index
    pair<bool, int> get_index(Key key) const {
        Key i = key % n_;
        while (valid_[i]) {
            if (data_[i].first == key) {
                return {true, i};
            }
            if (++i == n_) {
                i = 0;
            }
        }
        return {false, i};
    }

    // 指定したindexにkeyとvalueを格納する
    void set(int i, Key key, T value) {
        valid_[i] = true;
        data_[i] = {key, value};
    }

    // 指定したindexのvalueを返す
    T get(int i) const {
        assert(valid_[i]);
        return data_[i].second;
    }

    void clear() { fill(valid_.begin(), valid_.end(), false); }

  private:
    uint32_t n_;
    vector<bool> valid_;
    vector<pair<Key, T>> data_;
};

using Hash = uint32_t;

// 状態遷移を行うために必要な情報
// メモリ使用量をできるだけ小さくしてください
using Action = bitset<k + 2>;

using Cost = int;

// ターンごとのログ
struct TurnLog {
    int turn;
    size_t beam_width_limit; // 設定上のビーム幅上限
    size_t beam_width;       // 実際に生存したノード数
    size_t candidate_count;  // expandで生成された候補数
    size_t hash_dedup_count; // ハッシュ重複で弾かれた数
    size_t pruned_count;     // コストが悪くて弾かれた数
    size_t unique_parents;   // ユニークな親ノード数
    Cost best_cost, worst_cost;
    double mean_cost, std_cost;
    float elapsed_sec;
};

constexpr int eval_c1 = -2;
constexpr int eval_c2 = 1;

// 状態のコストを評価するための構造体
// メモリ使用量をできるだけ小さくしてください
struct Evaluator {
    int score;
    int penalty;

    Evaluator(int score, int penalty) : score(score), penalty(penalty) {}

    // 低いほどよい
    Cost evaluate() const { return eval_c1 * score + eval_c2 * penalty; }
};

// 展開するノードの候補を表す構造体
struct Candidate {
    Action action;
    Evaluator evaluator;
    Hash hash;
    int parent;

    Candidate(Action action, Evaluator evaluator, Hash hash, int parent)
        : action(action), evaluator(evaluator), hash(hash), parent(parent) {}
};

// ノードの候補から実際に追加するものを選ぶクラス
// ビーム幅の個数だけ、評価がよいものを選ぶ
// ハッシュ値が一致したものについては、評価がよいほうのみを残す
class Selector {
  public:
    explicit Selector(const Config &config)
        : hash_to_index_(USE_HASH_DEDUP ? config.hash_map_capacity : 1) {
        beam_width = config.beam_width();
        candidates_.reserve(beam_width);
        full_ = false;

        costs_.resize(beam_width);
        for (size_t i = 0; i < beam_width; ++i) {
            costs_[i] = {0, i};
        }
    }

    // 候補を追加する
    // ターン数最小化型の問題で、candidateによって実行可能解が得られる場合にのみ
    // finished = true とする ビーム幅分の候補をCandidateを追加したときにsegment
    // treeを構築する
    void push(const Candidate &candidate, bool finished) {
        if (finished) {
            finished_candidates_.emplace_back(candidate);
            return;
        }
        Cost cost = candidate.evaluator.evaluate();
        if constexpr (DEBUG) {
            candidate_count_++;
        }
        if (full_ && cost >= st_.all_prod().first) {
            // 保持しているどの候補よりもコストが小さくないとき
            if constexpr (DEBUG) {
                pruned_count_++;
            }
            return;
        }

        if constexpr (USE_HASH_DEDUP) {
            auto [valid, i] = hash_to_index_.get_index(candidate.hash);

            if (valid) {
                int j = hash_to_index_.get(i);
                if (candidate.hash == candidates_[j].hash) {
                    // ハッシュ値が等しいものが存在しているとき
                    if constexpr (DEBUG) {
                        hash_dedup_count_++;
                    }
                    if (full_) {
                        if (cost < st_.get(j).first) {
                            candidates_[j] = candidate;
                            st_.set(j, {cost, j});
                        }
                    } else {
                        if (cost < costs_[j].first) {
                            candidates_[j] = candidate;
                            costs_[j].first = cost;
                        }
                    }
                    return;
                }
            }
            if (full_) {
                int j = st_.all_prod().second;
                hash_to_index_.set(i, candidate.hash, j);
                candidates_[j] = candidate;
                st_.set(j, {cost, j});
            } else {
                int j = candidates_.size();
                hash_to_index_.set(i, candidate.hash, j);
                candidates_.emplace_back(candidate);
                costs_[j].first = cost;

                if (candidates_.size() == beam_width) {
                    full_ = true;
                    st_ = MaxSegtree(costs_);
                }
            }
        } else {
            if (full_) {
                int j = st_.all_prod().second;
                candidates_[j] = candidate;
                st_.set(j, {cost, j});
            } else {
                int j = candidates_.size();
                candidates_.emplace_back(candidate);
                costs_[j].first = cost;

                if (candidates_.size() == beam_width) {
                    full_ = true;
                    st_ = MaxSegtree(costs_);
                }
            }
        }
    }

    // 選んだ候補を返す
    const vector<Candidate> &select() const { return candidates_; }

    // 実行可能解が見つかったか
    bool have_finished() const { return !finished_candidates_.empty(); }

    // 実行可能解に到達するCandidateを返す
    vector<Candidate> get_finished_candidates() const {
        return finished_candidates_;
    }

    // 最もよいCandidateを返す
    Candidate calculate_best_candidate() const {
        size_t best = 0;
        for (size_t i = 0; i < candidates_.size(); ++i) {
            if (candidates_[i].evaluator.score >
                candidates_[best].evaluator.score) {
                best = i;
            }
        }
        return candidates_[best];
    }

    // DEBUG用: ターンログを収集する
    TurnLog collect_log(int turn, float elapsed_sec,
                        size_t beam_width_limit) const {
        TurnLog log{};
        log.turn = turn;
        log.beam_width_limit = beam_width_limit;
        log.beam_width = candidates_.size();
        log.candidate_count = candidate_count_;
        log.hash_dedup_count = hash_dedup_count_;
        log.pruned_count = pruned_count_;
        log.elapsed_sec = elapsed_sec;

        if (!candidates_.empty()) {
            Cost best = numeric_limits<Cost>::max();
            Cost worst = numeric_limits<Cost>::min();
            double sum = 0;
            unordered_set<int> parents;
            for (size_t i = 0; i < candidates_.size(); ++i) {
                Cost c = candidates_[i].evaluator.evaluate();
                if (c < best)
                    best = c;
                if (c > worst)
                    worst = c;
                sum += c;
                parents.insert(candidates_[i].parent);
            }
            double mean = sum / candidates_.size();
            double var_sum = 0;
            for (size_t i = 0; i < candidates_.size(); ++i) {
                double d = candidates_[i].evaluator.evaluate() - mean;
                var_sum += d * d;
            }
            log.best_cost = best;
            log.worst_cost = worst;
            log.mean_cost = mean;
            log.std_cost = sqrt(var_sum / candidates_.size());
            log.unique_parents = parents.size();
        }
        return log;
    }

    void clear() {
        candidates_.clear();
        if constexpr (USE_HASH_DEDUP) {
            hash_to_index_.clear();
        }
        full_ = false;
        if constexpr (DEBUG) {
            candidate_count_ = 0;
            hash_dedup_count_ = 0;
            pruned_count_ = 0;
        }
    }

    void update_beam_width(size_t new_beam_width) {
        if (new_beam_width == beam_width) {
            return;
        }
        beam_width = new_beam_width;
        candidates_.clear();
        candidates_.reserve(beam_width);
        if constexpr (USE_HASH_DEDUP) {
            hash_to_index_.clear();
        }
        full_ = false;
        costs_.resize(beam_width);
        for (size_t i = 0; i < beam_width; ++i) {
            costs_[i] = {0, i};
        }
    }

  private:
    // 削除可能な優先度付きキュー
    using MaxSegtree =
        atcoder::segtree<pair<Cost, int>,
                         [](pair<Cost, int> a, pair<Cost, int> b) {
                             if (a.first >= b.first) {
                                 return a;
                             } else {
                                 return b;
                             }
                         },
                         []() {
                             return make_pair(numeric_limits<Cost>::min(), -1);
                         }>;

    size_t beam_width;
    vector<Candidate> candidates_;
    HashMap<Hash, int> hash_to_index_;
    bool full_;
    vector<pair<Cost, int>> costs_;
    MaxSegtree st_;
    vector<Candidate> finished_candidates_;

    // DEBUG用カウンタ
    size_t candidate_count_ = 0;
    size_t hash_dedup_count_ = 0;
    size_t pruned_count_ = 0;
};

// 深さ優先探索に沿って更新する情報をまとめたクラス
class State {
  public:
    explicit State(const Input &input) {
        m_ = select_grids(input);
        for (int i = 0; i < k; ++i) {
            const string &grid = input.grids[m_[i]];
            positions_[i] = grid.find('@');
            visited_[i].resize(h * w, 0);
            visited_[i][positions_[i]] = 1;
            edges_[i].resize(4 * h * w, -1);
            for (int j = 0; j < h * w; ++j) {
                if (grid[j] == 'x' || grid[j] == '#') {
                    continue;
                }
                for (int d = 0; d < 4; ++d) {
                    if (grid[j + di[d]] == 'x') {
                        // edges_[i][4 * j + d] = -1;
                    } else if (grid[j + di[d]] == '#') {
                        edges_[i][4 * j + d] = j;
                    } else {
                        edges_[i][4 * j + d] = j + di[d];
                    }
                }
            }
        }
    }

    // EvaluatorとHashの初期値を返す
    pair<Evaluator, Hash> make_initial_node() { return {{0, 0}, 0}; }

    // 次の状態候補を全てselectorに追加する
    void expand(const Evaluator &evaluator, Hash _, int parent,
                Selector &selector) const {
        for (int d = 0; d < 4; ++d) {
            Action action(0);
            int score = evaluator.score;
            int penalty = 0;

            bool game_over = false;
            for (int i = 0; i < k; ++i) {
                int fr = positions_[i];
                int to = edges_[i][4 * fr + d];
                if (to == -1) {
                    game_over = true;
                    break;
                }
                if (fr != to) {
                    action[i] = true;
                    if (visited_[i][to] == 0) {
                        ++score;
                    }
                }
                penalty += visited_[i][to];
            }
            if (game_over) {
                continue;
            }
            action[k] = d % 2;
            action[k + 1] = d / 2;

            // calculate new hash
            int pos1 = positions_[0] + (action[0] ? di[d] : 0);
            int pos2 = positions_[1] + (action[1] ? di[d] : 0);
            Hash hash = (pos2 << 13) | pos1;

            selector.push(
                Candidate(action, Evaluator(score, penalty), hash, parent),
                false);
        }
    }

    // actionを実行して次の状態に遷移する
    void move_forward(Action action) {
        int d = action.to_ullong() >> k;

        for (int i = 0; i < k; ++i) {
            if (action[i]) {
                positions_[i] += di[d];
            }
            ++visited_[i][positions_[i]];
        }
    }

    // actionを実行する前の状態に遷移する
    void move_backward(Action action) {
        int d = action.to_ullong() >> k;

        for (int i = 0; i < k; ++i) {
            --visited_[i][positions_[i]];
            if (action[i]) {
                positions_[i] -= di[d];
            }
        }
    }

    array<int, k> get_m() const { return m_; }

  private:
    array<int, k> m_;
    array<vector<int>, k> edges_;
    array<vector<int>, k> visited_;
    array<int, k> positions_;
};

// Euler Tourを管理するためのクラス
class Tree {
  public:
    explicit Tree(const State &state, const Config &config) : state_(state) {
        curr_tour_.reserve(config.tour_capacity);
        next_tour_.reserve(config.tour_capacity);
        leaves_.reserve(config.max_beam_width);
        buckets_.assign(config.max_beam_width, {});
    }

    // 状態を更新しながら深さ優先探索を行い、次のノードの候補を全てselectorに追加する
    void dfs(Selector &selector) {
        if (curr_tour_.empty()) {
            // 最初のターン
            auto [evaluator, hash] = state_.make_initial_node();
            state_.expand(evaluator, hash, 0, selector);
            return;
        }

        for (auto [leaf_index, action] : curr_tour_) {
            if (leaf_index >= 0) {
                // 葉
                state_.move_forward(action);
                auto &[evaluator, hash] = leaves_[leaf_index];
                state_.expand(evaluator, hash, leaf_index, selector);
                state_.move_backward(action);
            } else if (leaf_index == -1) {
                // 前進辺
                state_.move_forward(action);
            } else {
                // 後退辺
                state_.move_backward(action);
            }
        }
    }

    // 木を更新する
    void update(const vector<Candidate> &candidates) {
        leaves_.clear();

        if (curr_tour_.empty()) {
            // 最初のターン
            for (const Candidate &candidate : candidates) {
                curr_tour_.push_back({(int)leaves_.size(), candidate.action});
                leaves_.push_back({candidate.evaluator, candidate.hash});
            }
            return;
        }

        for (const Candidate &candidate : candidates) {
            buckets_[candidate.parent].push_back(
                {candidate.action, candidate.evaluator, candidate.hash});
        }

        auto it = curr_tour_.begin();

        // 一本道を反復しないようにする
        while (it->first == -1 && it->second == curr_tour_.back().second) {
            Action action = (it++)->second;
            state_.move_forward(action);
            direct_road_.push_back(action);
            curr_tour_.pop_back();
        }

        // 葉の追加や不要な辺の削除をする
        while (it != curr_tour_.end()) {
            auto [leaf_index, action] = *(it++);
            if (leaf_index >= 0) {
                // 葉
                if (buckets_[leaf_index].empty()) {
                    continue;
                }
                next_tour_.push_back({-1, action});
                for (auto [new_action, evaluator, hash] :
                     buckets_[leaf_index]) {
                    int new_leaf_index = leaves_.size();
                    next_tour_.push_back({new_leaf_index, new_action});
                    leaves_.push_back({evaluator, hash});
                }
                buckets_[leaf_index].clear();
                next_tour_.push_back({-2, action});
            } else if (leaf_index == -1) {
                // 前進辺
                next_tour_.push_back({-1, action});
            } else {
                // 後退辺
                auto [old_leaf_index, old_action] = next_tour_.back();
                if (old_leaf_index == -1) {
                    next_tour_.pop_back();
                } else {
                    next_tour_.push_back({-2, action});
                }
            }
        }
        swap(curr_tour_, next_tour_);
        next_tour_.clear();
    }

    // 根からのパスを取得する
    vector<Action> calculate_path(int parent, int turn) const {
        vector<Action> ret = direct_road_;
        ret.reserve(turn);
        for (auto [leaf_index, action] : curr_tour_) {
            if (leaf_index >= 0) {
                if (leaf_index == parent) {
                    ret.push_back(action);
                    return ret;
                }
            } else if (leaf_index == -1) {
                ret.push_back(action);
            } else {
                ret.pop_back();
            }
        }

        unreachable();
    }

  private:
    State state_;
    vector<pair<int, Action>> curr_tour_;
    vector<pair<int, Action>> next_tour_;
    vector<pair<Evaluator, Hash>> leaves_;
    vector<vector<tuple<Action, Evaluator, Hash>>> buckets_;
    vector<Action> direct_road_;
};

// DEBUG用: ログをstderrに1行1JSONで出力する
// プレフィックス "BEAM_LOG:" で他の出力と区別する
void write_beam_log(const vector<TurnLog> &logs) {
    for (const auto &l : logs) {
        cerr << "BEAM_LOG:{"
             << "\"turn\":" << l.turn
             << ",\"beam_width_limit\":" << l.beam_width_limit
             << ",\"beam_width\":" << l.beam_width
             << ",\"candidates\":" << l.candidate_count
             << ",\"hash_dedup\":" << l.hash_dedup_count
             << ",\"pruned\":" << l.pruned_count
             << ",\"unique_parents\":" << l.unique_parents
             << ",\"best\":" << l.best_cost << ",\"worst\":" << l.worst_cost
             << ",\"mean\":" << fixed << setprecision(6) << l.mean_cost
             << ",\"std\":" << l.std_cost << ",\"elapsed\":" << l.elapsed_sec
             << "}" << endl;
    }
}

// ビームサーチを行う関数
vector<Action> beam_search(const Config &config, const State &state, timer &t) {
    Tree tree(state, config);

    // 動的調整の初期化
    unique_ptr<BayesianBeamWidthSuggester> suggester;
    if (config.dynamic_beam) {
        suggester = make_unique<BayesianBeamWidthSuggester>(
            config.expected_turn, config.warmup_turn, config.time_limit,
            config.initial_beam_width, config.min_beam_width,
            config.max_beam_width, t);
        config.current_beam_width_ = suggester->suggest(t);
    }

    // 新しいノード候補の集合
    Selector selector(config);

    // DEBUG用ログ
    vector<TurnLog> turn_logs;
    if constexpr (DEBUG) {
        turn_logs.reserve(config.max_turn);
    }

    for (int turn = 0; turn < config.max_turn; ++turn) {

        // Euler Tourでselectorに候補を追加する
        tree.dfs(selector);

        if (selector.have_finished()) {
            // ターン数最小化型の問題で実行可能解が見つかったとき
            if constexpr (DEBUG) {
                turn_logs.push_back(selector.collect_log(turn, t.elapsed(),
                                                         config.beam_width()));
                write_beam_log(turn_logs);
            }
            Candidate candidate = selector.get_finished_candidates()[0];
            vector<Action> ret =
                tree.calculate_path(candidate.parent, turn + 1);
            ret.push_back(candidate.action);
            return ret;
        }

        assert(!selector.select().empty());

        if (turn == config.max_turn - 1) {
            // ターン数固定型の問題で全ターンが終了したとき
            if constexpr (DEBUG) {
                turn_logs.push_back(selector.collect_log(turn, t.elapsed(),
                                                         config.beam_width()));
                write_beam_log(turn_logs);
            }
            Candidate best_candidate = selector.calculate_best_candidate();
            vector<Action> ret =
                tree.calculate_path(best_candidate.parent, turn + 1);
            ret.push_back(best_candidate.action);
            return ret;
        }

        // DEBUG用: ログ収集
        if constexpr (DEBUG) {
            turn_logs.push_back(
                selector.collect_log(turn, t.elapsed(), config.beam_width()));
        }

        // 木を更新する
        tree.update(selector.select());

        selector.clear();

        // 動的ビーム幅の調整（ベイズ推定）
        if (config.dynamic_beam) {
            config.current_beam_width_ = suggester->suggest(t);
            selector.update_beam_width(config.beam_width());
        }
    }

    unreachable();
}

} // namespace beam_search

// ==========================================
// Solver
// ==========================================
constexpr size_t max_beam_width = 3900;
constexpr size_t tour_capacity = 16 * max_beam_width;
constexpr uint32_t hash_map_capacity = 64 * max_beam_width;

class Solver {
    Input &in;
    timer &total_timer;
    array<int, k> m;
    string output;

  public:
    Solver(Input &in, timer &total_timer) : in(in), total_timer(total_timer) {}
    void solve() {
        beam_search::Config config = {
            .max_turn = t,
            .expected_turn = t,
            .dynamic_beam = false,
            .time_limit = 3.9,
            .max_beam_width = max_beam_width,
            .initial_beam_width = 3900,
            .min_beam_width = 100,
            .warmup_turn = 10,
            .tour_capacity = tour_capacity,
            .hash_map_capacity = hash_map_capacity,
        };
        beam_search::State state(in);
        m = state.get_m();
        vector<beam_search::Action> actions =
            beam_search::beam_search(config, state, total_timer);

        // make output
        output.resize(actions.size());
        for (size_t i = 0; i < actions.size(); ++i) {
            int d = actions[i].to_ullong() >> k;
            output[i] = udlr[d];
        }
    }
    void print() {
        for (int i = 0; i < k; ++i) {
            cout << m[i] << (i == k - 1 ? "\n" : " ");
        }
        cout << output << "\n";
    }
};

int main() {
    timer total_timer;
    Input in;
    in.input();
    Solver solver(in, total_timer);
    solver.solve();
    solver.print();
    return 0;
}
