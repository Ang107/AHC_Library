// #pragma GCC optimize("O3")
// #pragma GCC optimize("unroll-loops")
// #pragma GCC optimize("Ofast")
#include <bits/stdc++.h>
// INCLUDE <experimental/simd>
// INCLUDE <atcoder/all>
// #include <ext/pb_ds/assoc_container.hpp>
// #include <immintrin.h>
// #include <sys/time.h>
// #include <x86intrin.h>
using namespace std;
constexpr bool DEBUG = true;
// ==========================================
// SA Parameters (調整箇所)
// ==========================================
using score_t = int64_t;        // スコアの型
constexpr bool MAXIMIZE = true; // true: 最大化, false: 最小化
constexpr double TIME_LIMIT = 1.95;
constexpr double START_TEMP = 100000;
constexpr double END_TEMP = 100;
constexpr int TIME_CHECK_INTERVAL = 0x7F; // 時間チェック頻度 (ビットマスク)
constexpr int STATS_INTERVAL = 10 * (TIME_CHECK_INTERVAL + 1); // 統計出力頻度
constexpr bool USE_EXPONENTIAL_DECAY = true; // true: 指数減衰, false: 線形減衰
constexpr bool ALLOW_WORSE_MOVES = false;    // true: SA, false: 山登り

// 近傍操作の定義
enum MoveType { SWAP, INSERT, DELEATE, MOVE, RANGE_FILL, NUM_MOVES };

// 近傍の名前（enum順に対応）
vector<string> MOVE_NAMES = {"SWAP", "INSERT", "DELEATE", "MOVE", "RANGE_FILL"};

// 近傍の重み（整数、合計値は任意）
// SWAP, INSERT, DELEATE, MOVE, RANGE_FILL の順
vector<int> MOVE_WEIGHTS = {1, 2, 2, 1, 1};

// 近傍選択用（整数配列法）
vector<int> move_selector;
int selector_size;

// 近傍の重み（整数）から選択配列を構築
inline void build_move_selector(const vector<int> &weights) {
    move_selector.clear();
    for (int i = 0; i < (int)weights.size(); i++) {
        for (int j = 0; j < weights[i]; j++) {
            move_selector.push_back(i);
        }
    }
    selector_size = move_selector.size();
}

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
            // 線形減衰
            temp_table[i] = START_TEMP + (END_TEMP - START_TEMP) * progress;
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
    if (x <= 0.0)
        return -10.0;
    if (x >= 1.0)
        return 0.0;

    double index = x * TABLE_SIZE;
    int idx = (int)index;
    if (idx >= TABLE_SIZE)
        return log_table[TABLE_SIZE];

    double frac = index - idx;
    return log_table[idx] * (1.0 - frac) + log_table[idx + 1] * frac;
}

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
// template <typename K, typename V, typename Hash = uint64_hash>
// using hash_map = __gnu_pbds::gp_hash_table<K, V, Hash>;
// template <typename K, typename Hash = uint64_hash>
// using hash_set = hash_map<K, __gnu_pbds::null_type, Hash>;

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
// 問題固有の定数
// ==========================================
constexpr int N = 10;
constexpr int L = 4;
constexpr int T = 500;
constexpr int K = 1;

// チェックポイント間隔（アクション数ベース）
constexpr int CHECKPOINT_INTERVAL = 20;

// ==========================================
// Input & State
// ==========================================
struct Input {
    vector<int> a;
    vector<vector<i64>> c;
    void input() {
        int n, l, t, k;
        cin >> n >> l >> t >> k;
        a.resize(n);
        c.resize(l, vector<i64>(n));
        rep(i, n) cin >> a[i];
        rep(i, l) rep(j, n) cin >> c[i][j];
    }
};

// チェックポイント構造体
struct Checkpoint {
    array<array<i64, N>, L> B;
    array<array<i64, N>, L> P;
    i64 apples;
    int turn;
    int action_idx;
};

struct State {
    score_t score = -1;
    // actions: 強化する機械の順序リスト (level, id)
    vector<pair<int, int>> actions;

    // チェックポイント管理
    vector<Checkpoint> checkpoints;
    int dirty_from = 0; // この action_idx 以降が変更された

    State(const Input &in) {
        actions.push_back({0, 0});
        dirty_from = 0;
    }

    // チェックポイントを無効化（dirty_from以降）
    void invalidate_from(int action_idx) {
        dirty_from = min(dirty_from, action_idx);
    }

    // チェックポイントをすべてクリア
    void clear_checkpoints() {
        checkpoints.clear();
        dirty_from = 0;
    }

    // dirty_from より前の有効なチェックポイントを探す
    int find_valid_checkpoint() const {
        int best = -1;
        for (int i = 0; i < (int)checkpoints.size(); i++) {
            // dirty_from: 変更があった最初のアクション位置
            // action_idx < dirty_from のチェックポイントのみ有効
            if (checkpoints[i].action_idx < dirty_from) {
                best = i;
            } else {
                break;
            }
        }
        return best;
    }

    // シミュレーションを実行（チェックポイントから再開可能、読み取り専用）
    pair<score_t, int> simulate_readonly(const Input &in) const {
        array<array<i64, N>, L> B;
        array<array<i64, N>, L> P;
        i64 apples;
        int action_idx;
        int start_turn;

        // 有効なチェックポイントを探す
        int cp_idx = find_valid_checkpoint();

        if (cp_idx >= 0) {
            const Checkpoint &cp = checkpoints[cp_idx];
            B = cp.B;
            P = cp.P;
            apples = cp.apples;
            action_idx = cp.action_idx;
            start_turn = cp.turn;
        } else {
            for (int i = 0; i < L; i++) {
                for (int j = 0; j < N; j++) {
                    B[i][j] = 1;
                    P[i][j] = 0;
                }
            }
            apples = K;
            action_idx = 0;
            start_turn = 0;
        }

        for (int t = start_turn; t < T; t++) {
            if (action_idx < (int)actions.size()) {
                auto [level, id] = actions[action_idx];
                i64 cost = in.c[level][id] * (P[level][id] + 1);
                if (apples >= cost) {
                    apples -= cost;
                    P[level][id]++;
                    action_idx++;
                }
            }

            for (int i = 0; i < L; i++) {
                for (int j = 0; j < N; j++) {
                    if (i == 0) {
                        apples += (i64)in.a[j] * B[i][j] * P[i][j];
                    } else {
                        B[i - 1][j] += B[i][j] * P[i][j];
                    }
                }
            }
        }

        if (apples <= 0)
            return {0, action_idx};
        return {(score_t)round(100000.0 * log2((double)apples)), action_idx};
    }

    // チェックポイントを更新しながらシミュレーション（採用時のみ呼ぶ）
    void update_checkpoints(const Input &in) {
        // dirty_from以降のチェックポイントを削除
        int cp_idx = find_valid_checkpoint();
        if (cp_idx >= 0) {
            checkpoints.resize(cp_idx + 1);
        } else {
            checkpoints.clear();
        }

        array<array<i64, N>, L> B;
        array<array<i64, N>, L> P;
        i64 apples;
        int action_idx;
        int start_turn;

        if (!checkpoints.empty()) {
            const Checkpoint &cp = checkpoints.back();
            B = cp.B;
            P = cp.P;
            apples = cp.apples;
            action_idx = cp.action_idx;
            start_turn =
                cp.turn; // ターン終了後の状態なので、このターンから開始
        } else {
            for (int i = 0; i < L; i++) {
                for (int j = 0; j < N; j++) {
                    B[i][j] = 1;
                    P[i][j] = 0;
                }
            }
            apples = K;
            action_idx = 0;
            start_turn = 0;
        }

        int next_checkpoint_action =
            (checkpoints.empty())
                ? CHECKPOINT_INTERVAL
                : checkpoints.back().action_idx + CHECKPOINT_INTERVAL;

        for (int t = start_turn; t < T; t++) {
            if (action_idx < (int)actions.size()) {
                auto [level, id] = actions[action_idx];
                i64 cost = in.c[level][id] * (P[level][id] + 1);
                if (apples >= cost) {
                    apples -= cost;
                    P[level][id]++;
                    action_idx++;
                }
            }

            for (int i = 0; i < L; i++) {
                for (int j = 0; j < N; j++) {
                    if (i == 0) {
                        apples += (i64)in.a[j] * B[i][j] * P[i][j];
                    } else {
                        B[i - 1][j] += B[i][j] * P[i][j];
                    }
                }
            }

            // ターン終了後にチェックポイント作成
            if (action_idx >= next_checkpoint_action) {
                Checkpoint cp;
                cp.B = B;
                cp.P = P;
                cp.apples = apples;
                cp.turn = t + 1; // 次のターンから開始
                cp.action_idx = action_idx;
                checkpoints.push_back(cp);
                next_checkpoint_action = action_idx + CHECKPOINT_INTERVAL;
            }
        }

        dirty_from = (int)actions.size();
    }

    score_t calc_score_full(const Input &in) {
        auto [sc, executed] = simulate_readonly(in);
        return sc;
    }

    // 実行不可能なアクションを削除
    void trim_actions(const Input &in) {
        dirty_from = 0;
        checkpoints.clear();
        auto [sc, executed] = simulate_readonly(in);
        if (executed < (int)actions.size()) {
            actions.resize(executed);
        }
    }
};

// ==========================================
// SA
// ==========================================
struct MoveStats {
    i64 try_cnt = 0;
    i64 evaluated_cnt = 0;
    i64 accept_cnt = 0;
    i64 improve_cnt = 0;
    i64 update_best_cnt = 0;
} stats[NUM_MOVES];
i64 iter_count = 0;

class SA {
    const Input &in;
    const timer &total_timer;
    const double SA_TIME_LIMIT;
    const State &init_state;

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
    SA(const Input &in, const timer &total_timer, const double SA_TIME_LIMIT,
       const State &init_state)
        : in(in), total_timer(total_timer), SA_TIME_LIMIT(SA_TIME_LIMIT),
          init_state(init_state) {}

    State run() {
        State state = init_state;
        state.score = state.calc_score_full(in);
        State best_state = state;

        f64 current_temp = START_TEMP;
        f64 time_start = total_timer.elapsed();
        f64 time_now = time_start;

        // スコア停滞検出用
        score_t prev_score = state.score;
        int stagnation_count = 0;
        constexpr int STAGNATION_THRESHOLD = 1000000;
        constexpr int STAGNATION_DELETE_COUNT = 1;

        while (true) {
            iter_count++;

            // 時間チェック
            if ((iter_count & TIME_CHECK_INTERVAL) == 0) {
                time_now = total_timer.elapsed();
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
                f64 rand_val = rng.randomDouble();
                threshold = current_temp * log_fast(rand_val);
            }

            // 近傍選択
            int move_type = move_selector[rng.random32(selector_size)];
            if constexpr (DEBUG)
                stats[move_type].try_cnt++;

            // 遷移適用と差分計算
            score_t next_score = state.score;
            int n_actions = state.actions.size();

            // ロールバック用変数
            int rollback_type = -1;
            int rb_idx1 = -1, rb_idx2 = -1;
            pair<int, int> rb_old_action;
            vector<pair<int, int>> rb_old_actions; // RANGE_FILL用

            if (move_type == SWAP) {
                // SWAP: リスト内の2つの要素を入れ替え
                if (n_actions >= 2) {
                    rb_idx1 = rng.random32(n_actions);
                    rb_idx2 = rng.random32(n_actions);
                    if (rb_idx1 != rb_idx2 &&
                        state.actions[rb_idx1] != state.actions[rb_idx2]) {
                        swap(state.actions[rb_idx1], state.actions[rb_idx2]);
                        state.invalidate_from(min(rb_idx1, rb_idx2));
                        next_score = state.calc_score_full(in);
                        rollback_type = SWAP;
                    } else {
                        continue;
                    }
                }
            } else if (move_type == INSERT) {
                // INSERT: リストの任意の位置に新しい強化を追加
                rb_idx1 = rng.random32(n_actions + 1); // 挿入位置
                int level = rng.random32(L);
                int id = rng.random32(N);
                state.actions.insert(state.actions.begin() + rb_idx1,
                                     {level, id});
                state.invalidate_from(rb_idx1);
                next_score = state.calc_score_full(in);
                rollback_type = INSERT;
            } else if (move_type == DELEATE) {
                // DELETE: リストから強化を削除
                if (n_actions >= 1) {
                    rb_idx1 = rng.random32(n_actions);
                    rb_old_action = state.actions[rb_idx1];
                    state.actions.erase(state.actions.begin() + rb_idx1);
                    state.invalidate_from(rb_idx1);
                    next_score = state.calc_score_full(in);
                    rollback_type = DELEATE;
                }
            } else if (move_type == MOVE) {
                // MOVE: 要素を削除して別の位置に挿入
                if (n_actions >= 2) {
                    rb_idx1 = rng.random32(n_actions);     // 削除位置
                    rb_idx2 = rng.random32(n_actions - 1); // 挿入位置（削除後）
                    // rb_idx1を削除した後の挿入位置rb_idx2が元の位置と同じならスキップ
                    // 削除後の挿入位置rb_idx2 == rb_idx1の場合、元の位置に戻る
                    if (rb_idx2 == rb_idx1) {
                        continue;
                    }
                    auto old_actions = state.actions;
                    rb_old_action = state.actions[rb_idx1];
                    state.actions.erase(state.actions.begin() + rb_idx1);
                    state.actions.insert(state.actions.begin() + rb_idx2,
                                         rb_old_action);
                    if (old_actions == state.actions)
                        continue;
                    state.invalidate_from(min(rb_idx1, rb_idx2));
                    next_score = state.calc_score_full(in);
                    rollback_type = MOVE;
                } else {
                    continue;
                }
            } else if (move_type == RANGE_FILL) {
                // RANGE_FILL:
                // ランダムな区間のすべてのidを一つ選んだランダムなものに変更（レベルは維持）
                if (n_actions >= 1) {
                    rb_idx1 = rng.random32(n_actions);
                    rb_idx2 = rng.random32(n_actions);
                    if (rb_idx1 > rb_idx2)
                        swap(rb_idx1, rb_idx2);
                    int new_id = rng.random32(N);
                    // 元のアクションを保存（ロールバック用）
                    rb_old_actions.assign(state.actions.begin() + rb_idx1,
                                          state.actions.begin() + rb_idx2 + 1);
                    // 変更があるかチェック
                    bool has_change = false;
                    for (int i = rb_idx1; i <= rb_idx2; i++) {
                        if (state.actions[i].second != new_id) {
                            has_change = true;
                            break;
                        }
                    }
                    if (!has_change)
                        continue;
                    // 区間内のidのみを新しいものに変更（レベルは維持）
                    for (int i = rb_idx1; i <= rb_idx2; i++) {
                        state.actions[i].second = new_id;
                    }
                    state.invalidate_from(rb_idx1);
                    next_score = state.calc_score_full(in);
                    rollback_type = RANGE_FILL;
                } else {
                    continue;
                }
            }

            if constexpr (DEBUG)
                stats[move_type].evaluated_cnt++;

            // 採用判定（閾値ベース）
            f64 delta = (f64)(next_score - state.score);
            if constexpr (!MAXIMIZE)
                delta = -delta;

            bool accept = (delta >= threshold);
            if constexpr (DEBUG) {
                if (delta > 0.0)
                    stats[move_type].improve_cnt++;
            }

            // 更新 or ロールバック
            if (accept) {
                state.score = next_score;
                state.update_checkpoints(in); // 採用時のみチェックポイント更新
                if constexpr (DEBUG)
                    stats[move_type].accept_cnt++;

                bool is_better = MAXIMIZE ? (next_score > best_state.score)
                                          : (next_score < best_state.score);
                if (is_better) {
                    best_state = state;
                    if constexpr (DEBUG)
                        stats[move_type].update_best_cnt++;
                }
            } else {
                // ロールバック（actionsは元に戻るのでチェックポイントは有効）
                if (rollback_type == SWAP) {
                    swap(state.actions[rb_idx1], state.actions[rb_idx2]);
                } else if (rollback_type == INSERT) {
                    state.actions.erase(state.actions.begin() + rb_idx1);
                } else if (rollback_type == DELEATE) {
                    state.actions.insert(state.actions.begin() + rb_idx1,
                                         rb_old_action);
                } else if (rollback_type == MOVE) {
                    state.actions.erase(state.actions.begin() + rb_idx2);
                    state.actions.insert(state.actions.begin() + rb_idx1,
                                         rb_old_action);
                } else if (rollback_type == RANGE_FILL) {
                    for (int i = rb_idx1; i <= rb_idx2; i++) {
                        state.actions[i] = rb_old_actions[i - rb_idx1];
                    }
                }
                // ロールバック後はチェックポイント有効、dirty_fromリセット
                state.dirty_from = (int)state.actions.size();
            }

            // スコア停滞チェック
            if (state.score == prev_score) {
                stagnation_count++;
                if (stagnation_count >= STAGNATION_THRESHOLD) {
                    // ランダムに削除
                    int delete_count =
                        min(STAGNATION_DELETE_COUNT, (int)state.actions.size());
                    int min_deleted_idx = (int)state.actions.size();
                    for (int i = 0; i < delete_count; i++) {
                        if (state.actions.empty())
                            break;
                        int idx = rng.random32(state.actions.size());
                        min_deleted_idx = min(min_deleted_idx, idx);
                        state.actions.erase(state.actions.begin() + idx);
                    }
                    state.invalidate_from(min_deleted_idx);
                    state.score = state.calc_score_full(in);
                    state.update_checkpoints(in);
                    stagnation_count = 0;
                }
            } else {
                stagnation_count = 0;
                prev_score = state.score;
            }
        }

        // DEBUGに関わらず出力
        cerr << "Time: " << fixed << setprecision(3) << (total_timer.elapsed())
             << " Iter: " << iter_count << " Score: " << best_state.score << el;
        return best_state;
    }
};

// ==========================================
// Solver
// ==========================================
class Solver {
    Input &in;
    timer &total_timer;
    State best_state;

  public:
    Solver(Input &in, timer &total_timer)
        : in(in), total_timer(total_timer), best_state(in) {}

    void solve() {
        // 4回の焼きなまし時間制限
        double t = 0.2;
        for (double i = t; i < 1.7; i += t) {
            // 初期解の生成
            State init_state(in);

            // SAの実行
            SA sa(in, total_timer, i, init_state);
            State result = sa.run();

            // 実行不可能なアクションを削除
            result.trim_actions(in);
            result.score = result.calc_score_full(in);

            // 最良解の更新
            if (i == 0 || result.score > best_state.score) {
                best_state = result;
                cerr << "Updated best: " << best_state.score
                     << " at time limit " << i << endl;
            }
        }
        SA sa(in, total_timer, 1.99, best_state);
        best_state = sa.run();
    }

    void print() {
        // シミュレーションを実行して、各ターンのアクションを決定
        vector<vector<i64>> B(L, vector<i64>(N, 1));
        vector<vector<i64>> P(L, vector<i64>(N, 0));
        i64 apples = K;
        int action_idx = 0;

        for (int t = 0; t < T; t++) {
            // このターンで実行するアクションを決定
            pair<int, int> action_this_turn = {-1, -1};

            if (action_idx < (int)best_state.actions.size()) {
                auto [level, id] = best_state.actions[action_idx];
                i64 cost = in.c[level][id] * (P[level][id] + 1);
                if (apples >= cost) {
                    apples -= cost;
                    P[level][id]++;
                    action_this_turn = {level, id};
                    action_idx++;
                }
            }

            // 出力
            if (action_this_turn.first >= 0) {
                cout << action_this_turn.first << " " << action_this_turn.second
                     << el;
            } else {
                cout << -1 << el;
            }

            // 生産処理
            for (int i = 0; i < L; i++) {
                for (int j = 0; j < N; j++) {
                    if (i == 0) {
                        apples += (i64)in.a[j] * B[i][j] * P[i][j];
                    } else {
                        B[i - 1][j] += B[i][j] * P[i][j];
                    }
                }
            }
        }
    }
};

int main() {
    timer total_timer;
    init_temp_table();                 // 温度減衰テーブルの初期化
    init_log_table();                  // logテーブルの初期化
    build_move_selector(MOVE_WEIGHTS); // 近傍選択テーブルの初期化
    Input in;
    in.input();
    Solver solver(in, total_timer);
    solver.solve();
    solver.print();
    return 0;
}
