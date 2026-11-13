// common.hpp —— QuadFP-Cascade C++ 版公共常量、RNG、统计、序列化、矩阵与指纹工具
// ============================================================================
// 与 Python 版 00_common.py / 06 / 07 对应；全部使用 C++11 标准库（无第三方依赖）。
// 论文：《多尺度指纹级联框架 QuadFP-Cascade》（Computer Communications 251 (2026) 108495）
#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <cstdlib>
#include <sys/stat.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <direct.h>
#endif

// ----------------------------------------------------------------------
// 固定参数（与 Python 00_common.py 完全一致）
// ----------------------------------------------------------------------
static const long long SEED = 20261014;
static const long long R_SEED = SEED + 7;

static const int SEG_LEN = 2000;
static const double SEG_OVERLAP = 0.25;
static const int SEG_STEP = 1500;            // round(2000*(1-0.25))

static const int SCALE_WIN[3] = {32, 64, 128};
static const int SCALE_STEP[3] = {8, 16, 32};
static const int N_SCALES = 3;

static const int NUM_SYMBOLS = 7;
static const int FP_BITS = 128;
static const int FP_BYTES = 16;              // 128/8

// Table 1 七级符号阈值（论文固定）
static const double DELTA_BINS[6] = {-0.15, -0.06, -0.02, 0.02, 0.06, 0.15};

// LSH 教学配置
static const int LSH_BANDS = 8;
static const int LSH_BAND_BITS = 16;

static const double THETA_LO_Q = 0.30;
static const double THETA_HI_Q = 0.60;

// 01 合成数据参数
static const int CONN_LEN = 8000;
static const int N_CONN_PER_KIND = 8;

// channel id：0=合法，1..5=五类隐蔽
static const char* CHANNEL_NAMES[6] =
    {"LEGIT", "IPCTC", "LNCTC", "TRCTC", "Jitterbug", "EpsilonKlibur"};
static const char* PROTOCOL_NAMES[3] = {"HTTP", "HTTPS", "SSH"};

// ----------------------------------------------------------------------
// 路径
// ----------------------------------------------------------------------
inline std::string out_dir() {
    return std::string("outputs");
}
inline void ensure_out_dir() {
#ifdef _WIN32
    _mkdir("outputs");
#else
    mkdir("outputs", 0755);
#endif
}
inline std::string out_path(const std::string& f) {
    ensure_out_dir();
    return out_dir() + "/" + f;
}
// 读取数据来源标记 outputs/data_source.txt
// （01_generate_ipd 写 synthetic；01_load_caida 写 CAIDA/<subset>）。缺省视为 synthetic。
inline std::string get_data_source() {
    std::ifstream f((out_dir() + "/data_source.txt").c_str());
    std::string s;
    if (f) std::getline(f, s); else s = "synthetic";
    while (!s.empty() && (s.back() == '\r' || s.back() == '\n' || s.back() == ' '))
        s.pop_back();
    if (s.empty()) s = "synthetic";
    return s;
}
inline void init_console() {
#ifdef _WIN32
    SetConsoleOutputCP(65001);
#endif
}
inline void banner(const std::string& t) {
    std::cout << "\n" << std::string(70, '=') << "\n" << t << "\n"
              << std::string(70, '=') << "\n";
}
inline void sub_banner(const std::string& t) {
    std::cout << "\n" << std::string(66, '-') << "\n" << t << "\n"
              << std::string(66, '-') << "\n";
}

// ----------------------------------------------------------------------
// RNG：std::mt19937 + Box-Muller 正态（替代 numpy default_rng；数值不逐位相同，
// 但同分布、同参数、固定种子，保证判别结构与结论一致）
// ----------------------------------------------------------------------
struct RNG {
    std::mt19937 eng;
    bool has_spare;
    double spare;
    RNG(long long seed) : eng((unsigned)seed), has_spare(false), spare(0.0) {}

    double standard_normal() {            // 极坐标（Marsaglia）Box-Muller
        if (has_spare) { has_spare = false; return spare; }
        double u, v, s;
        do {
            u = 2.0 * uniform01() - 1.0;
            v = 2.0 * uniform01() - 1.0;
            s = u * u + v * v;
        } while (s <= 1e-12 || s >= 1.0);
        double mul = std::sqrt(-2.0 * std::log(s) / s);
        spare = v * mul; has_spare = true;
        return u * mul;
    }
    double uniform01() {
        return std::generate_canonical<double, 53>(eng);
    }
    double uniform(double lo, double hi) {
        return lo + (hi - lo) * uniform01();
    }
    // [lo, hi) 整数（numpy r.integers(lo,hi)）
    int integer(int lo, int hi) {
        if (hi <= lo) return lo;
        std::uniform_int_distribution<int> d(lo, hi - 1);
        return d(eng);
    }
};

// ----------------------------------------------------------------------
// 统计工具（numpy 等价：linear 插值分位）
// ----------------------------------------------------------------------
inline double mean_vec(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    double s = 0.0; for (double x : v) s += x; return s / (double)v.size();
}
inline double median_vec(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    int n = (int)v.size();
    if (n % 2 == 1) return v[n / 2];
    return 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
inline double quantile_vec(std::vector<double> v, double q) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double pos = q * (double)(v.size() - 1);
    int lo = (int)std::floor(pos), hi = (int)std::ceil(pos);
    double frac = pos - lo;
    if (hi == lo) return v[lo];
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}
inline double sigmoid(double x) {
    if (x < -60) x = -60; if (x > 60) x = 60;
    return 1.0 / (1.0 + std::exp(-x));
}
inline int count_windows(int n, int win, int step) {
    if (n < win) return 0;
    return (n - win) / step + 1;
}

// ----------------------------------------------------------------------
// 二进制中间结果归档（替代 .npz）：仅数值数组，扁平行优先 + shape
// ----------------------------------------------------------------------
struct BinArc {
    std::string name;
    char tag;                              // 'd' double / 'i' int / 'b' uint8
    std::vector<int> shape;
    std::vector<double> d;
    std::vector<int> i;
    std::vector<unsigned char> b;
    long long numel() const {
        long long n = 1; for (int s : shape) n *= s; return n;
    }
};
struct Store {
    std::vector<BinArc> arcs;

    BinArc* find(const std::string& nm) {
        for (size_t k = 0; k < arcs.size(); ++k)
            if (arcs[k].name == nm) return &arcs[k];
        return NULL;
    }
    void put_d(const std::string& nm, std::vector<int> shape,
               const std::vector<double>& data) {
        BinArc a; a.name = nm; a.tag = 'd'; a.shape = shape; a.d = data;
        arcs.push_back(a);
    }
    void put_d(const std::string& nm, double scalar) {
        put_d(nm, std::vector<int>(1, 1), std::vector<double>(1, scalar));
    }
    void put_i(const std::string& nm, std::vector<int> shape,
               const std::vector<int>& data) {
        BinArc a; a.name = nm; a.tag = 'i'; a.shape = shape; a.i = data;
        arcs.push_back(a);
    }
    void put_i(const std::string& nm, int scalar) {
        put_i(nm, std::vector<int>(1, 1), std::vector<int>(1, scalar));
    }
    void put_b(const std::string& nm, std::vector<int> shape,
               const std::vector<unsigned char>& data) {
        BinArc a; a.name = nm; a.tag = 'b'; a.shape = shape; a.b = data;
        arcs.push_back(a);
    }

    void save(const std::string& path) const {
        std::ofstream f(path.c_str(), std::ios::binary);
        int na = (int)arcs.size();
        f.write((char*)&na, sizeof(int));
        for (size_t k = 0; k < arcs.size(); ++k) {
            const BinArc& a = arcs[k];
            int nl = (int)a.name.size();
            f.write((char*)&nl, sizeof(int));
            f.write(a.name.data(), nl);
            f.write((char*)&a.tag, 1);
            int nd = (int)a.shape.size();
            f.write((char*)&nd, sizeof(int));
            for (int q = 0; q < nd; ++q) f.write((char*)&a.shape[q], sizeof(int));
            int n = (int)a.numel();
            f.write((char*)&n, sizeof(int));
            if (a.tag == 'd') f.write((char*)a.d.data(), n * sizeof(double));
            else if (a.tag == 'i') f.write((char*)a.i.data(), n * sizeof(int));
            else f.write((char*)a.b.data(), n * sizeof(unsigned char));
        }
    }
    static Store load(const std::string& path) {
        Store s;
        std::ifstream f(path.c_str(), std::ios::binary);
        if (!f) { std::cerr << "无法打开 " << path << "，请先运行上游步骤。\n"; std::exit(1); }
        int na; f.read((char*)&na, sizeof(int));
        for (int k = 0; k < na; ++k) {
            BinArc a; int nl;
            f.read((char*)&nl, sizeof(int));
            a.name.resize(nl); f.read(&a.name[0], nl);
            f.read((char*)&a.tag, 1);
            int nd; f.read((char*)&nd, sizeof(int));
            a.shape.resize(nd);
            for (int q = 0; q < nd; ++q) f.read((char*)&a.shape[q], sizeof(int));
            int n; f.read((char*)&n, sizeof(int));
            if (a.tag == 'd') { a.d.resize(n); f.read((char*)a.d.data(), n * sizeof(double)); }
            else if (a.tag == 'i') { a.i.resize(n); f.read((char*)a.i.data(), n * sizeof(int)); }
            else { a.b.resize(n); f.read((char*)a.b.data(), n * sizeof(unsigned char)); }
            s.arcs.push_back(a);
        }
        return s;
    }
};

// 多维扁平索引（行优先）
inline long long off3(const std::vector<int>& sh, int i, int j, int k) {
    return (long long)i * sh[1] * sh[2] + (long long)j * sh[2] + k;
}
inline long long off2(const std::vector<int>& sh, int i, int j) {
    return (long long)i * sh[1] + j;
}

// ----------------------------------------------------------------------
// 指纹：128 位 -> 16 字节（MSB first，等价 numpy.packbits）
// ----------------------------------------------------------------------
inline void pack_bits(const int bits[FP_BITS], unsigned char out[FP_BYTES]) {
    for (int byte = 0; byte < FP_BYTES; ++byte) {
        unsigned char v = 0;
        for (int t = 0; t < 8; ++t)
            if (bits[byte * 8 + t]) v |= (unsigned char)(1 << (7 - t));
        out[byte] = v;
    }
}
inline int popcount8(unsigned char x) {
    int c = 0; while (x) { c += (x & 1); x >>= 1; } return c;
}
// 两个 16 字节指纹的汉明距离
inline int hamming_bits(const unsigned char* a, const unsigned char* b) {
    int d = 0;
    for (int t = 0; t < FP_BYTES; ++t) d += popcount8((unsigned char)(a[t] ^ b[t]));
    return d;
}
inline double hamming_sim(const unsigned char* a, const unsigned char* b) {
    return 1.0 - hamming_bits(a, b) / (double)FP_BITS;
}
// band b（16 位）的整数签名，等价 np 16 位点积
inline int band_signature(const unsigned char* h, int b) {
    return ((int)h[2 * b] << 8) | (int)h[2 * b + 1];
}

// ----------------------------------------------------------------------
// 07 两级聚合（位移内中位数 -> 位移间 top3 均值），09 复用
// ----------------------------------------------------------------------
static const int MAX_SHIFT = 5;
static const int TOP_SHIFTS = 3;
// 一个尺度下两段窗口指纹的鲁棒相似度
inline double scale_similarity(const unsigned char* Ha, const unsigned char* Hb,
                               int nw, int max_shift = MAX_SHIFT,
                               int top_shifts = TOP_SHIFTS) {
    std::vector<double> shift_scores;
    for (int tau = -max_shift; tau <= max_shift; ++tau) {
        std::vector<double> sims;
        for (int ia = 0; ia < nw; ++ia) {
            int ib = ia + tau;
            if (ib >= 0 && ib < nw)
                sims.push_back(hamming_sim(Ha + ia * FP_BYTES, Hb + ib * FP_BYTES));
        }
        if (!sims.empty()) shift_scores.push_back(median_vec(sims));
    }
    std::sort(shift_scores.begin(), shift_scores.end());
    int k = std::min(top_shifts, (int)shift_scores.size());
    double s = 0.0;
    for (int t = (int)shift_scores.size() - k; t < (int)shift_scores.size(); ++t)
        s += shift_scores[t];
    return s / (double)k;
}

// ----------------------------------------------------------------------
// 简单稠密矩阵（GCN 用）
// ----------------------------------------------------------------------
struct Mat {
    int r, c;
    std::vector<double> a;
    Mat() : r(0), c(0) {}
    Mat(int r_, int c_) : r(r_), c(c_), a(r_ * c_, 0.0) {}
    double& at(int i, int j) { return a[i * c + j]; }
    double get(int i, int j) const { return a[i * c + j]; }
};
inline Mat mat_mul(const Mat& A, const Mat& B) {
    Mat C(A.r, B.c);
    for (int i = 0; i < A.r; ++i)
        for (int k = 0; k < A.c; ++k) {
            double aik = A.get(i, k);
            if (aik == 0.0) continue;
            for (int j = 0; j < B.c; ++j)
                C.at(i, j) += aik * B.get(k, j);
        }
    return C;
}
