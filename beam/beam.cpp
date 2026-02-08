
// #pragma GCC optimize("O3")
// #pragma GCC optimize("unroll-loops")
// #pragma GCC optimize("Ofast")
#include <atcoder/all>
#include <bits/stdc++.h>

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
} rng;
class FenwickWeightedSampler {
  private:
    int n_ = 0;
    vector<double> bit_;
    vector<double> weight_;
    double total_ = 0.0;

    void add_internal(int idx0, double delta) {
        int i = idx0 + 1;
        while (i <= n_) {
            bit_[i] += delta;
            i += i & -i;
        }
    }

  public:
    // Complexity: O(n)
    void init(int n, double initial_weight = 0.0) {
        n_ = n;
        bit_.assign(n_ + 1, 0.0);
        weight_.assign(n_, initial_weight);
        total_ = 0.0;
        if (initial_weight != 0.0) {
            for (int i = 0; i < n_; i++) {
                add_internal(i, initial_weight);
            }
            total_ = initial_weight * n_;
        }
    }

    // Complexity: O(n log n)
    void build(const vector<double> &weights) {
        init((int)weights.size(), 0.0);
        for (int i = 0; i < n_; i++) {
            weight_[i] = weights[i];
            add_internal(i, weights[i]);
            total_ += weights[i];
        }
    }

    // Complexity: O(log n)
    void set_weight(int idx, double new_weight) {
        assert(0 <= idx && idx < n_);
        const double delta = new_weight - weight_[idx];
        weight_[idx] = new_weight;
        add_internal(idx, delta);
        total_ += delta;
    }

    // Complexity: O(log n)
    void add_weight(int idx, double delta) {
        assert(0 <= idx && idx < n_);
        weight_[idx] += delta;
        add_internal(idx, delta);
        total_ += delta;
    }

    // Complexity: O(log n)
    int sample() const {
        assert(n_ > 0);
        assert(total_ > 0.0);
        const double r = rng.randomDouble() * total_;

        int idx = 0;
        double prefix = 0.0;
        int step = 1;
        while ((step << 1) <= n_)
            step <<= 1;
        for (int k = step; k > 0; k >>= 1) {
            const int nxt = idx + k;
            if (nxt <= n_ && prefix + bit_[nxt] <= r) {
                idx = nxt;
                prefix += bit_[nxt];
            }
        }
        if (idx >= n_)
            idx = n_ - 1;
        return idx;
    }

    int size() const { return n_; }
    double total_weight() const { return total_; }
    double weight(int idx) const {
        assert(0 <= idx && idx < n_);
        return weight_[idx];
    }
};
class AliasWeightedSampler {
  private:
    int n_ = 0;
    vector<double> weight_;
    double total_ = 0.0;
    vector<double> prob_;
    vector<int> alias_;
    bool dirty_ = true;

    // Complexity: O(n)
    void rebuild_alias_table() {
        prob_.assign(n_, 0.0);
        alias_.assign(n_, 0);
        total_ = 0.0;
        for (double w : weight_)
            total_ += w;
        assert(total_ > 0.0);

        vector<double> scaled(n_);
        vector<int> small;
        vector<int> large;
        small.reserve(n_);
        large.reserve(n_);
        for (int i = 0; i < n_; i++) {
            scaled[i] = weight_[i] * n_ / total_;
            if (scaled[i] < 1.0) {
                small.push_back(i);
            } else {
                large.push_back(i);
            }
        }

        while (!small.empty() && !large.empty()) {
            const int s = small.back();
            small.pop_back();
            const int l = large.back();
            large.pop_back();
            prob_[s] = scaled[s];
            alias_[s] = l;
            scaled[l] -= (1.0 - scaled[s]);
            if (scaled[l] < 1.0) {
                small.push_back(l);
            } else {
                large.push_back(l);
            }
        }
        while (!large.empty()) {
            const int i = large.back();
            large.pop_back();
            prob_[i] = 1.0;
            alias_[i] = i;
        }
        while (!small.empty()) {
            const int i = small.back();
            small.pop_back();
            prob_[i] = 1.0;
            alias_[i] = i;
        }
        dirty_ = false;
    }

  public:
    // Complexity: O(n)
    void init(int n, double initial_weight = 0.0) {
        n_ = n;
        weight_.assign(n_, initial_weight);
        total_ = initial_weight * n_;
        prob_.clear();
        alias_.clear();
        dirty_ = true;
    }

    // Complexity: O(n)
    void build(const vector<double> &weights) {
        n_ = (int)weights.size();
        weight_ = weights;
        dirty_ = true;
        // Build lazily on sample() to allow batch updates before first use.
    }

    // Complexity: O(1), table becomes dirty
    void set_weight(int idx, double new_weight) {
        assert(0 <= idx && idx < n_);
        weight_[idx] = new_weight;
        dirty_ = true;
    }

    // Complexity: O(1), table becomes dirty
    void add_weight(int idx, double delta) {
        assert(0 <= idx && idx < n_);
        weight_[idx] += delta;
        dirty_ = true;
    }

    // Complexity: O(1) after table is ready, O(n) if rebuild is needed
    int sample() {
        assert(n_ > 0);
        if (dirty_)
            rebuild_alias_table();
        const int i = (int)rng.random32((uint32_t)n_);
        return (rng.randomDouble() < prob_[i]) ? i : alias_[i];
    }

    int size() const { return n_; }
    double total_weight() {
        if (dirty_)
            rebuild_alias_table();
        return total_;
    }
    double weight(int idx) const {
        assert(0 <= idx && idx < n_);
        return weight_[idx];
    }
};
// Timer
struct Timer {
    chrono::steady_clock::time_point t_begin;
    Timer() { t_begin = chrono::steady_clock::now(); }
    void reset() { t_begin = chrono::steady_clock::now(); }
    float elapsed() const {
        return chrono::duration<float>(chrono::steady_clock::now() - t_begin)
            .count();
    }
} timer;
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
// Beam Search (with hash deduplication)
// ==========================================
namespace beam_search {

// ハッシュによる重複除去を行うかどうか
constexpr bool USE_HASH_DEDUP = true;

// 正規-逆ガンマ分布によるベイズ推定
// ビーム幅1あたりの1ターンの所要時間の平均と分散を推定する
struct GaussInverseGamma {
    double mu, lambda, alpha, beta;

    GaussInverseGamma() : mu(0), lambda(1), alpha(1), beta(1) {}
    GaussInverseGamma(double mu, double lambda, double alpha, double beta)
        : mu(mu), lambda(lambda), alpha(alpha), beta(beta) {}

    // 疑似観測値から初期化
    static GaussInverseGamma
    from_pseudo_observation(double mean, double std_dev, int pseudo_count) {
        double variance = std_dev * std_dev;
        double precision = 1.0 / variance;
        double l = (double)pseudo_count;
        double a = (double)(pseudo_count * 2);
        double b = a / precision;
        return GaussInverseGamma(mean, l, a, b);
    }

    // 観測値 x でベイズ更新
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

    // (平均, 標準偏差) の期待値を返す
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
// 1ターンあたりの実行時間が正規分布に従うと仮定し、
// +3σ分の余裕を持ってビーム幅を決める
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
                               size_t min_beam_width, size_t max_beam_width)
        : time_limit_sec(time_limit_sec), current_turn(0), max_turn(max_turn),
          warmup_turn(warmup_turn), min_beam_width(min_beam_width),
          max_beam_width(max_beam_width), current_beam_width(0) {
        double mean_sec =
            time_limit_sec / ((double)max_turn * standard_beam_width);
        double stddev_sec = 0.2 * mean_sec;
        dist =
            GaussInverseGamma::from_pseudo_observation(mean_sec, stddev_sec, 3);
        max_memory_turn = max_turn / 5;
        start_time = timer.elapsed();
        last_time = start_time;
    }

    // ビーム幅を提案し、内部状態を更新する
    // 各ターンの冒頭で呼ぶ
    size_t suggest() {
        // 前ターンの観測値でベイズ更新
        if (current_turn > warmup_turn && current_beam_width > 0) {
            float now = timer.elapsed();
            double elapsed = (double)(now - last_time);
            double elapsed_per_beam = elapsed / current_beam_width;
            dist.update(elapsed_per_beam);

            if (dist.get_pseudo_observation_count() >=
                (double)max_memory_turn) {
                dist.set_pseudo_observation_count((double)max_memory_turn);
            }
        }

        last_time = timer.elapsed();

        if (current_turn >= max_turn) {
            current_beam_width = min_beam_width;
            current_turn++;
            return current_beam_width;
        }

        // 安全なビーム幅を計算
        double remaining_turn = (double)(max_turn - current_turn);
        double elapsed_time = (double)timer.elapsed();
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
    size_t min_beam_width; // 最小ビーム幅
    int warmup_turn; // ウォームアップターン数（最初のXターンの観測は捨てる）
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

using Hash = uint32_t; // TODO

// 状態遷移を行うために必要な情報
// メモリ使用量をできるだけ小さくしてください
struct Action {
    // TODO

    Action() {
        // TODO
    }

    bool operator==(const Action &other) const {
        // TODO
    }
};

using Cost = int;

// ターンごとのログ
struct TurnLog {
    int turn;
    size_t beam_width_limit; // 設定上のビーム幅上限
    size_t beam_width;
    size_t candidate_count;  // expandで生成された候補数
    size_t hash_dedup_count; // ハッシュ重複で弾かれた数
    size_t pruned_count;     // コストが悪くて弾かれた数
    size_t unique_parents;   // ユニークな親ノード数
    Cost best_cost, worst_cost;
    double mean_cost, std_cost;
    float elapsed_sec;
};

// 状態のコストを評価するための構造体
// メモリ使用量をできるだけ小さくしてください
struct Evaluator {
    // TODO

    Evaluator() {
        // TODO
    }

    // 低いほどよい
    Cost evaluate() const {
        // TODO
    }
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
                if constexpr (DEBUG) {
                    pruned_count_++;
                }
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
                if constexpr (DEBUG) {
                    pruned_count_++;
                }
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
        if (full_) {
            size_t best = 0;
            for (size_t i = 0; i < beam_width; ++i) {
                if (st_.get(i).first < st_.get(best).first) {
                    best = i;
                }
            }
            return candidates_[best];
        } else {
            size_t best = 0;
            for (size_t i = 0; i < candidates_.size(); ++i) {
                if (costs_[i].first < costs_[best].first) {
                    best = i;
                }
            }
            return candidates_[best];
        }
    }

    // DEBUG用: ターンログを収集し、カウンタをリセットする
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
        log.best_cost = 0;
        log.worst_cost = 0;
        log.mean_cost = 0.0;
        log.std_cost = 0.0;
        log.unique_parents = 0;

        // コスト統計とユニーク親の集計
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
    explicit State() {
        // TODO
    }

    // EvaluatorとHashの初期値を返す
    pair<Evaluator, Hash> make_initial_node() {
        // TODO
    }

    // 次の状態候補を全てselectorに追加する
    // 引数
    //   evaluator : 今の評価器
    //   hash      : 今のハッシュ値
    //   parent    : 今のノードID（次のノードにとって親となる）
    void expand(const Evaluator &evaluator, Hash hash, int parent,
                Selector &selector) {
        // TODO
    }

    // actionを実行して次の状態に遷移する
    void move_forward(Action action) {
        // TODO
    }

    // actionを実行する前の状態に遷移する
    // 今の状態は、親からactionを実行して遷移した状態である
    void move_backward(Action action) {
        // TODO
    }

  private:
    // TODO
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
        // cerr << curr_tour_.size() << endl;

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
vector<Action> beam_search(const Config &config, const State &state) {
    Tree tree(state, config);

    // 動的調整の初期化
    unique_ptr<BayesianBeamWidthSuggester> suggester;
    if (config.dynamic_beam) {
        suggester = make_unique<BayesianBeamWidthSuggester>(
            config.expected_turn, config.warmup_turn, config.time_limit,
            config.initial_beam_width, config.min_beam_width,
            config.max_beam_width);
        config.current_beam_width_ = suggester->suggest();
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
                turn_logs.push_back(selector.collect_log(turn, timer.elapsed(),
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
                turn_logs.push_back(selector.collect_log(turn, timer.elapsed(),
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
            turn_logs.push_back(selector.collect_log(turn, timer.elapsed(),
                                                     config.beam_width()));
        }

        // 木を更新する
        tree.update(selector.select());

        selector.clear();

        // 動的ビーム幅の調整（ベイズ推定）
        if (config.dynamic_beam) {
            config.current_beam_width_ = suggester->suggest();
            selector.update_beam_width(config.beam_width());
        }
    }

    unreachable();
}

} // namespace beam_search

// ==========================================
// Input & State
// ==========================================
struct Input {
    void input() {
        // todo: 入力読み込み
    }
} in;

// ==========================================
// Solver
// ==========================================
class Solver {

  public:
    Solver() {}
    void solve() {
        // todo: 解法
    }
    void print() {
        // todo: 解の出力
    }
};

int main() {
    in.input();
    Solver solver;
    solver.solve();
    solver.print();
    return 0;
}
