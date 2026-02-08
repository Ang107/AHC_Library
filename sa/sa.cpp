
// #pragma GCC optimize("O3")
// #pragma GCC optimize("unroll-loops")
// #pragma GCC optimize("Ofast")
#include <bits/stdc++.h>

using namespace std;
constexpr bool DEBUG = true;
// ==========================================
// SA Parameters (調整箇所)
// ==========================================
using score_t = int64_t;        // スコアの型
constexpr bool MAXIMIZE = true; // true: 最大化, false: 最小化
constexpr double TIME_LIMIT = 1.95;
constexpr double START_TEMP = 1000.0;
constexpr double END_TEMP = 10.0;
constexpr int TIME_CHECK_INTERVAL = 0x7F; // 時間チェック頻度 (ビットマスク)
constexpr int STATS_INTERVAL = 10 * (TIME_CHECK_INTERVAL + 1); // 統計出力頻度
constexpr bool USE_EXPONENTIAL_DECAY =
    true; // true: 指数減衰, false: べき乗減衰
constexpr double POWER_DECAY_EXP =
    1.0; // べき乗減衰の指数 (1.0で線形, >1で序盤高温維持, <1で序盤急冷)
constexpr bool ALLOW_WORSE_MOVES = true; // true: SA, false: 山登り

// 近傍操作の定義
enum MoveType { SWAP, INSERT, REVERSE, NUM_MOVES };

// 近傍の名前（enum順に対応）
vector<string> MOVE_NAMES = {"SWAP", "INSERT", "REVERSE"};

// 近傍の重み（整数、合計値は任意）
// SWAP, INSERT, REVERSE の順
vector<double> MOVE_WEIGHTS = {1, 1, 1};

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
// 入出力高速化
struct Init {
    Init() {
        ios::sync_with_stdio(0);
        cin.tie(0);
    }
} init;
// テーブルサイズ（温度・logで共通）
constexpr int TABLE_SIZE = 1024;

// 温度減衰テーブル
double temp_table[TABLE_SIZE + 1];

inline void init_temp_table() {
    for (int i = 0; i <= TABLE_SIZE; i++) {
        double progress = (double)i / TABLE_SIZE;
        if constexpr (USE_EXPONENTIAL_DECAY) {
            // 指数減衰
            double ratio = END_TEMP / START_TEMP;
            temp_table[i] = START_TEMP * pow(ratio, progress);
        } else {
            // べき乗減衰: start - (start - end) * progress^x (x=1で線形)
            double p = pow(progress, POWER_DECAY_EXP);
            temp_table[i] = START_TEMP - (START_TEMP - END_TEMP) * p;
        }
    }
}

inline double temp_decay(double progress) {
    // テーブルルックアップ + 線形補間
    double index = progress * TABLE_SIZE;
    int idx = (int)index;
    if (idx >= TABLE_SIZE)
        return temp_table[TABLE_SIZE];

    double frac = index - idx;
    return temp_table[idx] * (1.0 - frac) + temp_table[idx + 1] * frac;
}

// logテーブル
double log_table[TABLE_SIZE + 1];

inline void init_log_table() {
    for (int i = 0; i <= TABLE_SIZE; i++) {
        double x = (double)i / TABLE_SIZE;
        if (x == 0.0) {
            log_table[i] = -10.0; // log(0) = -inf を有限値で近似
        } else {
            log_table[i] = log(x);
        }
    }
}

inline double log_fast(double x) {
    if constexpr (DEBUG)
        assert(0.0 < x && x < 1.0);

    double index = x * TABLE_SIZE;
    int idx = (int)index;
    if (idx >= TABLE_SIZE)
        return log_table[TABLE_SIZE];

    double frac = index - idx;
    return log_table[idx] * (1.0 - frac) + log_table[idx + 1] * frac;
}

// ガウシアンテーブル（Box-Mullerの事前計算）
constexpr int GAUSS_TABLE_SIZE = 131072;
double gauss_table[GAUSS_TABLE_SIZE];

inline void init_gauss_table() {
    for (int i = 0; i < GAUSS_TABLE_SIZE; i++) {
        double u1 = rng.randomDoubleOpen01();
        double u2 = rng.randomDouble();
        gauss_table[i] =
            sqrt(-2.0 * log(u1)) * cos(2.0 * 3.141592653589793238 * u2);
    }
}

inline double fast_gaussian(double mean, double stddev) {
    return mean + stddev * gauss_table[rng.random32(GAUSS_TABLE_SIZE)];
}

// ==========================================
// Input & State
// ==========================================
struct Input {
    void input() {
        // todo: 入力読み込み
    }
} in;

struct State {
    score_t score = -1;
    // todo: 状態変数

    State() {
        // todo: 初期解生成
    }

    score_t calc_score_full() {
        score_t result = 0;
        // todo: フルスコア計算
        return result;
    }
};

// ==========================================
// SA
// ==========================================

// 多点スタートに対応するためグローバル変数で持つ
struct MoveStats {
    i64 try_cnt = 0;
    i64 evaluated_cnt = 0;
    i64 accept_cnt = 0;
    i64 improve_cnt = 0;
    i64 update_best_cnt = 0;
} stats[NUM_MOVES];
i64 iter_count = 0;

class SA {
    const double SA_TIME_LIMIT;
    const State &init_state;
    AliasWeightedSampler &move_selector;

    inline void print_stats(i64 iter, f64 time, f64 temp, score_t score,
                            score_t best) {
        if constexpr (!DEBUG)
            return;
        cerr << "VISUALIZE: {";
        cerr << "\"iter\": " << iter << ", ";
        cerr << "\"time\": " << fixed << setprecision(3) << time << ", ";
        cerr << "\"temp\": " << fixed << setprecision(3) << temp << ", ";
        cerr << "\"score\": " << score << ", ";
        cerr << "\"best_score\": " << best << ", ";
        cerr << "\"moves\": [";
        for (int i = 0; i < NUM_MOVES; i++) {
            if (i > 0)
                cerr << ", ";
            cerr << "{\"name\": \"" << MOVE_NAMES[i]
                 << "\", \"tried\": " << stats[i].try_cnt
                 << ", \"evaluated\": " << stats[i].evaluated_cnt
                 << ", \"accepted\": " << stats[i].accept_cnt
                 << ", \"improved\": " << stats[i].improve_cnt
                 << ", \"updated_best\": " << stats[i].update_best_cnt << "}";
        }
        cerr << "]}" << el;
    }

  public:
    SA(const double SA_TIME_LIMIT, const State &init_state,
       AliasWeightedSampler &move_selector)
        : SA_TIME_LIMIT(SA_TIME_LIMIT), init_state(init_state),
          move_selector(move_selector) {}

    State run() {
        State state = init_state;
        state.score = state.calc_score_full();
        State best_state = state;

        f64 current_temp = START_TEMP;
        f64 time_start = timer.elapsed();
        f64 time_now = time_start;

        while (true) {
            iter_count++;

            // 時間チェック
            if ((iter_count & TIME_CHECK_INTERVAL) == 0) {
                time_now = timer.elapsed();
                if (time_now > SA_TIME_LIMIT)
                    break;
                if constexpr (ALLOW_WORSE_MOVES) {
                    f64 progress =
                        (time_now - time_start) / (SA_TIME_LIMIT - time_start);
                    current_temp = temp_decay(progress);
                }
                if (DEBUG && (iter_count % STATS_INTERVAL == 0)) {
                    if constexpr (ALLOW_WORSE_MOVES)
                        print_stats(iter_count, time_now, current_temp,
                                    state.score, best_state.score);
                    else
                        print_stats(iter_count, time_now, current_temp,
                                    state.score, state.score);
                }
            }

            // 採用閾値の事前計算
            f64 threshold;
            if constexpr (!ALLOW_WORSE_MOVES) {
                threshold = 0.0;
            } else {
                f64 rand_val = rng.randomDoubleOpen01();
                threshold = current_temp * log_fast(rand_val);
            }

            // 近傍選択
            int move_type = move_selector.sample();
            if constexpr (DEBUG)
                stats[move_type].try_cnt++;

            // 遷移適用と差分計算
            score_t next_score = state.score;
            // todo: ロールバック用変数の宣言

            if (move_type == SWAP) {
                // todo: SWAP操作
            } else if (move_type == INSERT) {
                // todo: INSERT操作
            } else if (move_type == REVERSE) {
                // todo: REVERSE操作
            }

            if constexpr (DEBUG)
                stats[move_type].evaluated_cnt++;

            // 採用判定（閾値ベース）
            f64 delta = (f64)(next_score - state.score);
            if constexpr (!MAXIMIZE)
                delta = -delta;

            bool accept = (delta >= threshold);
            if constexpr (DEBUG) {
                if (delta > 0)
                    stats[move_type].improve_cnt++;
            }

            // 更新 or ロールバック
            if (accept) {
                if constexpr (DEBUG)
                    stats[move_type].accept_cnt++;

                bool is_better;
                if constexpr (ALLOW_WORSE_MOVES) {
                    is_better = MAXIMIZE ? (next_score > best_state.score)
                                         : (next_score < best_state.score);
                } else {
                    is_better = MAXIMIZE ? (next_score > state.score)
                                         : (next_score < state.score);
                }

                state.score = next_score;

                if (is_better) {
                    if constexpr (ALLOW_WORSE_MOVES) {
                        best_state = state;
                    }
                    if constexpr (DEBUG)
                        stats[move_type].update_best_cnt++;
                }
                // todo: 更新
            } else {
                // todo: ロールバック
            }
        }
        if constexpr (!ALLOW_WORSE_MOVES)
            best_state = state;
        // DEBUGに関わらず出力
        cerr << "Time: " << fixed << setprecision(3) << (timer.elapsed())
             << " Iter: " << iter_count << " Score: " << best_state.score << el;
        return best_state;
    }
};

// ==========================================
// Solver
// ==========================================
class Solver {

  public:
    Solver() {}
    void solve() {
        // 近傍選択テーブルの初期化
        AliasWeightedSampler move_selector;
        if constexpr (DEBUG) {
            assert((int)MOVE_NAMES.size() == NUM_MOVES);
            assert((int)MOVE_WEIGHTS.size() == NUM_MOVES);
        }
        move_selector.build(MOVE_WEIGHTS);
        // todo: SAの実行など
    }
    void print() {
        // todo: 解の出力
    }
};

int main() {
    init_temp_table(); // 温度減衰テーブルの初期化
    init_log_table();  // logテーブルの初期化
    // init_gauss_table(); // ガウシアンテーブルの初期化
    in.input();
    Solver solver;
    solver.solve();
    solver.print();
    return 0;
}
