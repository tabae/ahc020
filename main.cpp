#include <bits/stdc++.h>
#include <sys/time.h>
#include <atcoder/all>
using namespace std;
using namespace atcoder;
using ll = long long;
using pl = std::pair<long long, long long>;
using mint = atcoder::modint1000000007;
//using mint = atcoder::modint998244353;
#define rep(i, srt, end) for (long long i = (srt); i < (long long)(end); i++)
#define all(v) (v).begin(), (v).end()
#define dump(var)  do{ if(debug::enable) { std::cerr << #var << " : "; debug::print(var); } } while(0);
constexpr int di[4] = {1, -1, 0, 0};
constexpr int dj[4] = {0, 0, 1, -1};
constexpr long long inf = 1LL<<60;
template<typename T> inline void chmax(T &a, T b) { a = std::max(a, b); };
template<typename T> inline void chmin(T &a, T b) { a = std::min(a, b); };
template<typename T> void vprint(const std::vector<T>& v) { for(int i = 0; i < v.size(); i++) { std::cout << v[i] << (i == v.size()-1 ? "\n" : " "); } }
template<typename T> void vprintln(const std::vector<T>& v) { for(int i = 0; i < v.size(); i++) { std::cout << v[i] << "\n"; } }
template<int M> void vprint(const std::vector<atcoder::static_modint<M>>& v) { for(int i = 0; i < v.size(); i++) { std::cout << v[i].val() << (i == v.size()-1 ? "\n" : " "); } }
template<int M> void vprintln(const std::vector<atcoder::static_modint<M>>& v) { for(int i = 0; i < v.size(); i++) { std::cout << v[i].val() << "\n"; } }
namespace debug {    
    #if defined(ONLINE_JUDGE)
    const bool enable = false;
    #else
    const bool enable = true;
    #endif
    template<typename T> void push(const T& e) { std::cerr << e; }
    template<int M> void push(const atcoder::static_modint<M>& e) { std::cerr << e.val(); }
    template<typename T1, typename T2> void push(const std::pair<T1, T2>& e) { std::cerr << "("; push(e.first); std::cerr << ","; push(e.second); std::cerr << ")"; }   
    template<typename T1, typename T2, typename T3> void push(const std::tuple<T1, T2, T3>& e) { std::cerr << "("; push(get<0>(e)); std::cerr << ","; push(get<1>(e)); std::cerr << ","; push(get<2>(e)); std::cerr << ")"; }   
    template<typename T> void print(const T& e) { push(e); std::cerr << "\n"; }
    template<typename T> void print(const std::vector<T>& v) { for(int i = 0; i < v.size(); i++) { push(v[i]); std::cerr << " "; } std::cerr << "\n"; }
    template<typename T> void print(const std::vector<std::vector<T>>& v) { std::cerr << "\n"; for(int i = 0; i < v.size(); i++) { std::cerr << i << ": "; print(v[i]); } }
};

// return maximum x such that x^2 <= n
// requirements: n <= 4e18 
long long isqrt(long long n) {
    long long res;
    long long l = 0, r = 2000000001;
    while(r - l > 1) {
        long long m = l + (r - l) / 2;
        if(m * m <= n) l = m;
        else r = m;
    }
    return r;
}

struct RandGenerator {
    random_device seed_gen;
    mt19937 engine;
    mt19937_64 engine64;
    static const int pshift = 1000000000;
    RandGenerator() : engine(seed_gen()), engine64(seed_gen()) {}
    /*mod以下の乱数を返す（32bit）*/
    int rand(int mod) {
        return engine() % mod;
    }
    /*mod以下の乱数を返す（64bit）*/
    long long randll(long long mod) {
        return engine64() % mod;
    } 
    /*確率pでTrueを返す*/
    bool pjudge(double p) {
        int p_int;
        if(p > 1) p_int = pshift;
        else p_int = p * pshift;
        return rand(pshift) < p_int;
    }
} ryuka;
struct timer {
    double global_start;
    double gettime() {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        return tv.tv_sec + tv.tv_usec * 1e-6;
    }
    void init() {
        global_start = gettime();
    }
    double elapsed() {
        return gettime() - global_start;
    }
} toki;

struct Input {
    ll n, m, k;
    vector<pl> nodes, houses;
    vector<tuple<ll,ll,ll>> edges;
    void read() {
        cin >> n >> m >> k;
        nodes.resize(n);
        edges.resize(m);
        houses.resize(k);
        for(auto& [x, y] : nodes) cin >> x >> y;
        for(auto& [u, v, w] : edges) {
            cin >> u >> v >> w;
            u--; v--;
        }
        for(auto& [a, b]: houses) cin >> a >> b;
    }
} input;

struct Output {
    vector<ll> p, b;
    Output() {
        p.resize(input.n, 0);
        b.resize(input.m, 0);
    }
    void print() {
        vprint(p);
        vprint(b);
    }
};

struct State {
    Output output;
    long long score;
    State() : score(-inf) {};
    static State initState(ll num_unit = 6);
    static State generateState(const State& input_state);
};

template<class STATE>
struct IterationControl {
    int iteration_counter;
    int swap_counter;
    double average_time;
    double start_time;
    IterationControl() : iteration_counter(0), swap_counter(0) {}
    STATE climb(double time_limit, STATE initial_state);
    STATE anneal(double time_limit, double temp_start, double temp_end, STATE initial_state);
};

namespace Utils {
    long long calcScore(const Output& output);
    atcoder::dsu getUF(const Output& output);
    vector<bool> getIsConnect(const Output& output);
    vector<vector<pl>> generateGraph(const Output& output);
    ll countCoveredHouse(const Output& output);
    long long calcSquaredDist(const pl& a, const pl& b);
    long long calcCost(const Output& output);
    vector<ll> getMST();
    tuple<ll,ll,ll,ll> getCorners();
};

int main() {
    toki.init();
    input.read();
    State ans = State::initState();
    ll best_num_unit = 6;
    rep(num_unit, 3, 10) {
        State tmp = State::initState(num_unit);
        if(tmp.score > ans.score) {
            ans = tmp;
            best_num_unit = num_unit;
        }
    }
    ans.output.print();
    dump(best_num_unit);
    cerr << "[INFO] - main - MyScore = " << ans.score << "\n";
    return 0;
}

tuple<ll,ll,ll,ll> Utils::getCorners() {
    ll min_x = inf;
    ll min_y = inf;
    ll max_x = -inf;
    ll max_y = -inf;
    for(auto [x, y]: input.nodes) {
        chmin(min_x, x);
        chmin(min_y, y);
        chmax(max_x, x);
        chmax(max_y, y);
    }
    for(auto [x, y]: input.houses) {
        chmin(min_x, x);
        chmin(min_y, y);
        chmax(max_x, x);
        chmax(max_y, y);
    }
    return tuple<ll,ll,ll,ll>({min_x, max_x, min_y, max_y});
}

vector<ll> Utils::getMST() {
    vector<int> idx(input.m);
    iota(all(idx), 0);
    sort(all(idx), [&](auto l, auto r) {
        return get<2>(input.edges[l]) < get<2>(input.edges[r]);
    });
    atcoder::dsu uf(input.n);
    vector<ll> res(input.m, false);
    for(auto i : idx) {
        auto [u, v, w] = input.edges[i];
        if(!uf.same(u, v)) {
            uf.merge(u, v);
            res[i] = true;
        }
    }
    return res;
}
    
long long Utils::calcCost(const Output& output) {
    ll res = 0;
    rep(i, 0, input.n) res += output.p[i] * output.p[i];
    rep(i, 0, input.m) {
        if(output.b[i]) {
            auto [u, v, w] = input.edges[i];
            res += w;
        }
    }
    return res;
}

long long Utils::calcSquaredDist(const pl& a, const pl& b) {
    ll dx = a.first - b.first;
    ll dy = a.second - b.second;
    return dx * dx + dy * dy;
}

ll Utils::countCoveredHouse(const Output& output) {
    auto is_connect = getIsConnect(output);
    vector<bool> ok(input.k, false);
    rep(i, 0, input.n) {
        if(!is_connect[i]) continue;
        rep(j, 0, input.k) {
            ll dist = calcSquaredDist(input.nodes[i], input.houses[j]);
            if(dist <= output.p[i] * output.p[i]) ok[j] = true;
        }
    }
    if(debug::enable) {
        rep(j, 0, input.k) {
            if(!ok[j]) {
                dump(j);
                dump(input.houses[j]);
            }
        }
    }
    ll res = 0;
    rep(i, 0, input.k) if(ok[i]) res++;
    return res;
}

atcoder::dsu Utils::getUF(const Output& output) {
    atcoder::dsu uf(input.n);
    rep(i, 0, input.m) {
        if(output.b[i]) {
            auto [u, v, _] = input.edges[i];
            uf.merge(u, v);
        }
    }
    return uf;
}

vector<bool> Utils::getIsConnect(const Output& output) {
    atcoder::dsu uf = Utils::getUF(output);
    vector<bool> res(input.n, false);
    rep(i, 0, input.n) if(uf.same(0, i)) res[i] = true;
    return res;
}

vector<vector<pl>> Utils::generateGraph(const Output& output) {
    vector<vector<pl>> G(input.n);
    rep(i, 0, input.m) {
        if(output.b[i]) {
            auto [u, v, w] = input.edges[i];
            G[u].push_back({v, w});
            G[v].push_back({u, w});
        }    
    }
    return G;
}

long long Utils::calcScore(const Output& output) {
    long long res = 0;
    ll covered_house = Utils::countCoveredHouse(output);
    dump(covered_house);
    dump(input.k);
    constexpr ll M = 1000000;
    for(auto e: output.p) {
        if(e > 5000) return 0;
    }
    if(covered_house == input.k) {
        ll s = Utils::calcCost(output);
        res = M + M * M * 100 / (s + M * 10);
    } else {
        res = M * (covered_house + 1) / input.k;
    }
    return res;
}

State State::initState(ll num_unit) {
    State res;
    res.output.b = Utils::getMST();
    auto [min_x, max_x, min_y, max_y] = Utils::getCorners();
    ll len_x = max_x - min_x;
    ll len_y = max_y - min_y;
    ll unit_x = (len_x + num_unit - 1) / num_unit;
    ll unit_y = (len_y + num_unit - 1) / num_unit;
    vector<ll> x_id_node(input.n, num_unit-1), y_id_node(input.n, num_unit-1);
    vector<ll> x_id_house(input.k, num_unit-1), y_id_house(input.k, num_unit-1);
    rep(i, 0, input.n) {
        ll l_x = min_x;
        ll l_y = min_y;
        ll r_x = min_x + unit_x;
        ll r_y = min_y + unit_y;
        rep(j, 0, num_unit) {
            if(l_x <= input.nodes[i].first  && input.nodes[i].first  < r_x) x_id_node[i] = j;
            if(l_y <= input.nodes[i].second &&input.nodes[i].second < r_y) y_id_node[i] = j;
            l_x += unit_x;
            l_y += unit_y;
            r_x += unit_x;
            r_y += unit_y;
        }   
    }
    rep(i, 0, input.k) {
        ll l_x = min_x;
        ll l_y = min_y;
        ll r_x = min_x + unit_x;
        ll r_y = min_y + unit_y;
        rep(j, 0, num_unit) {
            if(l_x <= input.houses[i].first  && input.houses[i].first  < r_x) x_id_house[i] = j;
            if(l_y <= input.houses[i].second && input.houses[i].second < r_y) y_id_house[i] = j;
            l_x += unit_x;
            l_y += unit_y;
            r_x += unit_x;
            r_y += unit_y;
        }   
    }
    dump(x_id_node[num_unit]);
    dump(y_id_node[num_unit]);
    dump(x_id_house[num_unit]);
    dump(y_id_house[num_unit]);
    vector<vector<ll>> node_group((num_unit+1) * (num_unit+1)); 
    vector<vector<ll>> house_group((num_unit+1) * (num_unit+1)); 
    auto f = [&](ll x, ll y) -> ll {
        return x * (num_unit + 1) + y;
    }; 
    rep(i, 0, input.n) {
        node_group[f(x_id_node[i], y_id_node[i])].push_back(i);
    }
    rep(i, 0, input.k) {
        house_group[f(x_id_house[i], y_id_house[i])].push_back(i);
    }
    rep(x, 0, num_unit + 1) {
        rep(y, 0, num_unit + 1) {
            ll id = f(x, y);
            ll x_center = unit_x * x + unit_x / 2;
            ll y_center = unit_y * y + unit_y / 2;
            pl center = {x_center, y_center};
            ll center_node = -1, min_dist = inf;
            for(auto i : node_group[id]) {
                ll dist = Utils::calcSquaredDist(center, input.nodes[i]);
                if(dist < min_dist) {
                    min_dist = dist;
                    center_node = i;
                }
            }
            if(center_node == -1) {
                dump(num_unit);
                dump(x);
                dump(y);
                continue;
            }
            ll max_dist = 0;
            for(auto i : house_group[id]) {
                ll dist = Utils::calcSquaredDist(input.nodes[center_node], input.houses[i]);
                chmax(max_dist, dist);
            }
            res.output.p[center_node] = isqrt(max_dist);
        }
    }
    res.score = Utils::calcScore(res.output);
    return res;
}

State State::generateState(const State& input_state) {
    State res = input_state;
    res.score = Utils::calcScore(res.output);
    return res;
}

template<class STATE>
STATE IterationControl<STATE>::climb(double time_limit, STATE initial_state) {
    start_time = toki.gettime();
    average_time = 0;
    STATE best_state = initial_state;
    double time_stamp = start_time;
    cerr << "[INFO] - IterationControl::climb - Starts climbing...\n";
    while(time_stamp - start_time + average_time < time_limit) {
        STATE current_state = STATE::generateState(best_state);
        if(current_state.score > best_state.score) {
            swap(best_state, current_state);
            swap_counter++;
        }
        iteration_counter++;
        time_stamp = toki.gettime();
        average_time = (time_stamp - start_time) / iteration_counter;
    }
    cerr << "[INFO] - IterationControl::climb - Iterated " << iteration_counter << " times and swapped " << swap_counter << " times.\n";
    return best_state;
}

template<class STATE>
STATE IterationControl<STATE>::anneal(double time_limit, double temp_start, double temp_end, STATE initial_state) {
    start_time = toki.gettime();
    average_time = 0;
    STATE best_state = initial_state;
    double elapsed_time = 0;
    cerr << "[INFO] - IterationControl::anneal - Starts annealing...\n";
    while(elapsed_time + average_time < time_limit) {
        double normalized_time = elapsed_time / time_limit;
        double temp_current = pow(temp_start, 1.0 - normalized_time) * pow(temp_end, normalized_time);
        STATE current_state = STATE::generateState(best_state);
        long long delta = current_state.score - best_state.score;
        if(delta > 0 || ryuka.pjudge(exp(1.0 * delta / temp_current)) ) {
            swap(best_state, current_state);
            swap_counter++;
        }
        iteration_counter++;
        elapsed_time = toki.gettime() - start_time;
        average_time = elapsed_time / iteration_counter;
    }
    cerr << "[INFO] - IterationControl::anneal - Iterated " << iteration_counter << " times and swapped " << swap_counter << " times.\n";
    return best_state;
}
