/**
 *  @file mace_weights.hpp
 *  @brief Host-side loader for the MACE-OFF flat weight blob.
 *
 *  Reads a file produced by @c python/smallfield/mace_export.py:
 *    - 28-byte preamble (magic "SF_MACE\0", version, manifest_offset, manifest_length)
 *    - data region (concatenated tensors, fp64)
 *    - JSON manifest at the end describing each tensor's name / dtype / shape / offset
 *
 *  The loader downcasts every fp64 tensor to fp32 (the working precision for
 *  the SmallField inference path) and copies it into a single contiguous device
 *  allocation.  Pointers into that allocation are exposed via
 *  @ref mace_weights_device_t.
 *
 *  Header-only.  RDKit-free.  No external JSON library — we parse the small,
 *  well-known schema ourselves with a minimal recursive-descent parser.
 */

#pragma once

#include "smallfield/module_fwd.hpp"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>

#include <cuda_runtime.h>

#include "smallfield/types/mace_types.hpp"

namespace smallfield::parameters {

namespace mace_detail_ {

/** Byte-for-byte preamble layout — must match `mace_export.py:PREAMBLE_FMT`.
 *
 *  We declare the struct `packed` because the natural alignment of `uint64_t`
 *  would insert 4 bytes of padding after `version`, taking sizeof from 28 to
 *  32 bytes; the Python side uses `struct.pack("<8sIQQ", ...)` which is the
 *  packed form.  The static_assert below is the regression alarm. */
#pragma pack(push, 1)
struct preamble_t {
    char magic[8];
    std::uint32_t version;
    std::uint64_t manifest_offset;
    std::uint64_t manifest_length;
};
#pragma pack(pop)
static_assert(sizeof(preamble_t) == 28, "preamble must be 28 bytes");

struct manifest_entry_t {
    std::string name;
    std::string dtype;
    std::vector<std::int64_t> shape;
    std::uint64_t offset;
    std::uint64_t nbytes;
};

/**
 *  @brief Tiny subset JSON parser for the manifest schema we control.
 *
 *  Handles: objects, arrays, strings (with simple escape \"), numbers, bool, null.
 *  Does NOT handle unicode escapes, exponents, or fancy numeric types — and
 *  does not need to, because the producer is `json.dumps(..., indent=0)` from
 *  Python with a known schema.  Throws std::runtime_error on bad input.
 */
class json_parser_t {
public:
    explicit json_parser_t(std::string const &s) : s_(s), p_(0) {}

    void skip_ws_() {
        while (p_ < s_.size() && (s_[p_] == ' ' || s_[p_] == '\t' || s_[p_] == '\n' || s_[p_] == '\r' || s_[p_] == ',')) ++p_;
    }

    void expect_(char c) {
        skip_ws_();
        if (p_ >= s_.size() || s_[p_] != c) throw std::runtime_error("manifest JSON: expected '" + std::string(1, c) + "'");
        ++p_;
    }

    std::string parse_string_() {
        skip_ws_();
        if (p_ >= s_.size() || s_[p_] != '"') throw std::runtime_error("manifest JSON: expected string");
        ++p_;
        std::string out;
        while (p_ < s_.size() && s_[p_] != '"') {
            if (s_[p_] == '\\' && p_ + 1 < s_.size()) { out.push_back(s_[p_ + 1]); p_ += 2; }
            else { out.push_back(s_[p_]); ++p_; }
        }
        if (p_ >= s_.size()) throw std::runtime_error("manifest JSON: unterminated string");
        ++p_;
        return out;
    }

    double parse_number_() {
        skip_ws_();
        std::size_t start = p_;
        if (p_ < s_.size() && (s_[p_] == '-' || s_[p_] == '+')) ++p_;
        while (p_ < s_.size() && (std::isdigit(static_cast<unsigned char>(s_[p_])) || s_[p_] == '.' || s_[p_] == 'e' || s_[p_] == 'E' || s_[p_] == '-' || s_[p_] == '+')) ++p_;
        return std::stod(s_.substr(start, p_ - start));
    }

    bool peek_(char c) { skip_ws_(); return p_ < s_.size() && s_[p_] == c; }
    char peek_() { skip_ws_(); return p_ < s_.size() ? s_[p_] : '\0'; }
    bool eof_() { skip_ws_(); return p_ >= s_.size(); }

    // Members are public so the (recursive) value-skip in `parse_manifest_`
    // can advance `p_` without going through accessors.  The parser is
    // package-private (under `mace_detail_`), so this is a controlled exposure.
    std::string const &s_;
    std::size_t p_;
};

/** Parse the full manifest into `(header_kv, tensors)`. */
inline void parse_manifest_(std::string const &raw,
                             std::unordered_map<std::string, double> &header_num,
                             std::unordered_map<std::string, std::string> &header_str,
                             std::vector<manifest_entry_t> &tensors) {
    json_parser_t j(raw);
    j.expect_('{');
    while (!j.peek_('}')) {
        std::string key = j.parse_string_();
        j.expect_(':');
        if (key == "header") {
            j.expect_('{');
            while (!j.peek_('}')) {
                std::string hk = j.parse_string_();
                j.expect_(':');
                if (j.peek_('"')) header_str[hk] = j.parse_string_();
                else if (j.peek_('-') || std::isdigit(static_cast<unsigned char>(j.peek_()))) header_num[hk] = j.parse_number_();
                else if (j.peek_('t') || j.peek_('f')) {
                    // bool — read alpha
                    std::size_t st = j.p_;
                    while (j.p_ < raw.size() && std::isalpha(static_cast<unsigned char>(raw[j.p_]))) ++j.p_;
                    header_num[hk] = (raw.substr(st, j.p_ - st) == "true") ? 1.0 : 0.0;
                } else if (j.peek_('n')) {
                    // null
                    j.p_ += 4;
                } else if (j.peek_('[')) {
                    // Array value — skip with bracket-depth tracking.  Header arrays
                    // are tolerated but not captured by the simple num/str maps;
                    // when a caller needs the array values it should mirror them as
                    // explicit tensors in the blob (see `mace_export.py`'s
                    // `graph_atom_offset` which is emitted both as a header array
                    // and as an int64 tensor).
                    int bracket_depth = 0;
                    do {
                        if (j.s_[j.p_] == '[') ++bracket_depth;
                        else if (j.s_[j.p_] == ']') --bracket_depth;
                        ++j.p_;
                    } while (bracket_depth > 0 && j.p_ < j.s_.size());
                } else if (j.peek_('{')) {
                    // Nested object value — skip the same way.
                    int brace_depth = 0;
                    do {
                        if (j.s_[j.p_] == '{') ++brace_depth;
                        else if (j.s_[j.p_] == '}') --brace_depth;
                        ++j.p_;
                    } while (brace_depth > 0 && j.p_ < j.s_.size());
                }
                if (j.peek_(',')) { ++j.p_; }
            }
            j.expect_('}');
        } else if (key == "tensors") {
            j.expect_('[');
            while (!j.peek_(']')) {
                j.expect_('{');
                manifest_entry_t e;
                while (!j.peek_('}')) {
                    std::string ek = j.parse_string_();
                    j.expect_(':');
                    if (ek == "name") e.name = j.parse_string_();
                    else if (ek == "dtype") e.dtype = j.parse_string_();
                    else if (ek == "shape") {
                        j.expect_('[');
                        while (!j.peek_(']')) {
                            e.shape.push_back(static_cast<std::int64_t>(j.parse_number_()));
                            if (j.peek_(',')) { ++j.p_; }
                        }
                        j.expect_(']');
                    }
                    else if (ek == "offset") e.offset = static_cast<std::uint64_t>(j.parse_number_());
                    else if (ek == "nbytes") e.nbytes = static_cast<std::uint64_t>(j.parse_number_());
                    if (j.peek_(',')) { ++j.p_; }
                }
                j.expect_('}');
                tensors.push_back(std::move(e));
                if (j.peek_(',')) { ++j.p_; }
            }
            j.expect_(']');
        } else {
            // unknown top-level key — skip a value (best-effort)
            if (j.peek_('"')) (void)j.parse_string_();
            else if (j.peek_('{')) {
                int depth = 0;
                do {
                    if (j.s_[j.p_] == '{') ++depth;
                    else if (j.s_[j.p_] == '}') --depth;
                    ++j.p_;
                } while (depth > 0 && j.p_ < j.s_.size());
            }
            else (void)j.parse_number_();
        }
        if (j.peek_(',')) { ++j.p_; }
    }
    j.expect_('}');
}

} // namespace mace_detail_

/**
 *  @brief Owning, RAII container for a fp32 device-resident MACE weight set.
 *
 *  Construction reads the file, downcasts fp64 → fp32, copies to device,
 *  builds @ref mace_weights_device_t pointers.  Destruction frees the device
 *  allocation.  Throws std::runtime_error on any I/O or format error.
 */
class mace_weights_t {
public:
    mace_weights_t() = default;
    mace_weights_t(mace_weights_t const &) = delete;
    mace_weights_t &operator=(mace_weights_t const &) = delete;
    mace_weights_t(mace_weights_t &&other) noexcept { swap_(other); }
    mace_weights_t &operator=(mace_weights_t &&other) noexcept { if (&other != this) { free_(); swap_(other); } return *this; }
    ~mace_weights_t() { free_(); }

    /** Load weights and resolve @ref mace_weights_device_t pointers. */
    void load(std::string const &path) {
        free_();
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("mace_weights: cannot open " + path);

        mace_detail_::preamble_t pre{};
        f.read(reinterpret_cast<char *>(&pre), sizeof(pre));
        if (std::memcmp(pre.magic, "SF_MACE\0", 8) != 0) throw std::runtime_error("mace_weights: bad magic in " + path);
        if (pre.version != 1u) throw std::runtime_error("mace_weights: unsupported version");

        // read JSON manifest
        std::string manifest_raw;
        manifest_raw.resize(pre.manifest_length);
        f.seekg(static_cast<std::streamoff>(pre.manifest_offset), std::ios::beg);
        f.read(manifest_raw.data(), static_cast<std::streamsize>(pre.manifest_length));

        std::unordered_map<std::string, double> hnum;
        std::unordered_map<std::string, std::string> hstr;
        std::vector<mace_detail_::manifest_entry_t> tensors;
        mace_detail_::parse_manifest_(manifest_raw, hnum, hstr, tensors);

        arch_.n_elements        = static_cast<std::uint32_t>(hnum.at("n_elements"));
        arch_.n_radial          = static_cast<std::uint32_t>(hnum.at("n_radial"));
        arch_.n_channels        = static_cast<std::uint32_t>(hnum.at("n_channels"));
        arch_.n_irreps_msg      = static_cast<std::uint32_t>(hnum.at("n_irreps_msg"));
        arch_.l_max             = static_cast<std::uint32_t>(hnum.at("L_max"));
        arch_.correlation_order = static_cast<std::uint32_t>(hnum.at("correlation_order"));
        arch_.num_interactions  = static_cast<std::uint32_t>(hnum.at("num_interactions"));
        arch_.scale             = static_cast<float>(hnum.at("scale"));
        arch_.shift             = static_cast<float>(hnum.at("shift"));
        arch_.cutoff_p          = static_cast<float>(hnum.at("cutoff_p"));
        arch_.r_max             = static_cast<float>(hnum.at("r_max"));
        arch_.bessel_prefactor  = static_cast<float>(hnum.at("bessel_prefactor"));
        // avg_num_neighbors was added in v1 schema; fall back to 1.0 for older blobs.
        arch_.avg_num_neighbors = hnum.count("avg_num_neighbors")
                                  ? static_cast<float>(hnum.at("avg_num_neighbors"))
                                  : 1.0f;

        // Read every tensor as fp64 from the file, downcast to fp32 in a host scratch buffer.
        // Build a map from tensor name to (host_offset, n_elements).
        std::unordered_map<std::string, std::pair<std::uint64_t, std::uint64_t>> name_to_offset_count;
        std::vector<float> host_blob;
        host_blob.reserve(8 * 1024 * 1024 / sizeof(float));
        // also the int64 atomic_numbers go through a separate path (kept as int64)
        std::vector<std::int64_t> host_atomic_numbers;

        for (auto const &t : tensors) {
            std::uint64_t n = 1;
            for (auto d : t.shape) n *= static_cast<std::uint64_t>(d);
            f.seekg(static_cast<std::streamoff>(sizeof(pre) + t.offset), std::ios::beg);

            if (t.dtype == "int64") {
                // Only atomic_numbers expected
                std::vector<std::int64_t> tmp(n);
                f.read(reinterpret_cast<char *>(tmp.data()), static_cast<std::streamsize>(n * sizeof(std::int64_t)));
                if (t.name == "atomic_numbers") host_atomic_numbers = std::move(tmp);
                continue;
            }
            if (t.dtype != "float64") throw std::runtime_error("mace_weights: unexpected dtype " + t.dtype + " for " + t.name);
            std::vector<double> tmp(n);
            f.read(reinterpret_cast<char *>(tmp.data()), static_cast<std::streamsize>(n * sizeof(double)));
            std::uint64_t host_offset = host_blob.size();
            host_blob.resize(host_blob.size() + n);
            for (std::uint64_t i = 0; i < n; ++i) host_blob[host_offset + i] = static_cast<float>(tmp[i]);
            name_to_offset_count[t.name] = {host_offset, n};
        }

        // Allocate device blobs and copy
        if (!host_atomic_numbers.empty()) {
            cudaMalloc(reinterpret_cast<void **>(&d_atomic_numbers_), host_atomic_numbers.size() * sizeof(std::int64_t));
            cudaMemcpy(d_atomic_numbers_, host_atomic_numbers.data(), host_atomic_numbers.size() * sizeof(std::int64_t), cudaMemcpyHostToDevice);
        }
        cudaMalloc(reinterpret_cast<void **>(&d_blob_), host_blob.size() * sizeof(float));
        cudaMemcpy(d_blob_, host_blob.data(), host_blob.size() * sizeof(float), cudaMemcpyHostToDevice);
        d_blob_count_ = host_blob.size();

        // Resolve mace_weights_device_t pointers.
        auto pos = [&](char const *name) -> float const * {
            auto it = name_to_offset_count.find(name);
            if (it == name_to_offset_count.end()) return nullptr;
            return d_blob_ + it->second.first;
        };
        wd_.atomic_numbers   = d_atomic_numbers_;
        wd_.atomic_energies  = pos("atomic_energies");
        wd_.bessel_weights   = pos("bessel_weights");
        wd_.node_embedding   = pos("node_embedding.weight");
        for (std::uint32_t k = 0; k < arch_.num_interactions && k < 2u; ++k) {
            auto &L = wd_.layers[k];
            std::string base = "interactions." + std::to_string(k);
            L.linear_up_w     = pos((base + ".linear_up.weight").c_str());
            L.radial_mlp_w0   = pos((base + ".conv_tp_weights.layer0.weight").c_str());
            L.radial_mlp_w1   = pos((base + ".conv_tp_weights.layer1.weight").c_str());
            L.radial_mlp_w2   = pos((base + ".conv_tp_weights.layer2.weight").c_str());
            L.radial_mlp_w3   = pos((base + ".conv_tp_weights.layer3.weight").c_str());
            L.linear_post_tp_w = pos((base + ".linear.weight").c_str());
            L.skip_tp_w       = pos((base + ".skip_tp.weight").c_str());
            std::string pbase = "products." + std::to_string(k);
            L.u_matrix_1      = pos((pbase + ".U_matrix_1").c_str());
            L.u_matrix_2      = pos((pbase + ".U_matrix_2").c_str());
            L.u_matrix_3      = pos((pbase + ".U_matrix_3").c_str());
            L.weights_max     = pos((pbase + ".weights_max").c_str());
            L.weights_lo_0    = pos((pbase + ".weights.0").c_str());
            L.weights_lo_1    = pos((pbase + ".weights.1").c_str());
            L.product_linear_w = pos((pbase + ".linear.weight").c_str());
        }
        wd_.readout_linear_w = pos("readouts.0.linear.weight");
        // Last interaction's nonlinear readout
        std::string last = "readouts." + std::to_string(arch_.num_interactions - 1);
        wd_.readout_mlp_w0 = pos((last + ".linear_1.weight").c_str());
        wd_.readout_mlp_w1 = pos((last + ".linear_2.weight").c_str());
    }

    types::mace_arch_t const &arch() const noexcept { return arch_; }
    types::mace_weights_device_t const &device() const noexcept { return wd_; }
    std::size_t device_bytes() const noexcept { return d_blob_count_ * sizeof(float); }

private:
    void swap_(mace_weights_t &o) noexcept {
        std::swap(arch_, o.arch_);
        std::swap(wd_, o.wd_);
        std::swap(d_blob_, o.d_blob_);
        std::swap(d_blob_count_, o.d_blob_count_);
        std::swap(d_atomic_numbers_, o.d_atomic_numbers_);
    }
    void free_() noexcept {
        if (d_blob_) cudaFree(d_blob_);
        if (d_atomic_numbers_) cudaFree(d_atomic_numbers_);
        d_blob_ = nullptr;
        d_atomic_numbers_ = nullptr;
        d_blob_count_ = 0;
        wd_ = {};
        arch_ = {};
    }

    types::mace_arch_t arch_{};
    types::mace_weights_device_t wd_{};
    float *d_blob_ = nullptr;
    std::int64_t *d_atomic_numbers_ = nullptr;
    std::size_t d_blob_count_ = 0;
};

} // namespace smallfield::parameters
