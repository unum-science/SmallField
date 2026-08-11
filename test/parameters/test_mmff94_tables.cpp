/**
 *  @file test/parameters/test_mmff94_tables.cpp
 *  @brief Cross-validate extracted MMFF94 tables against RDKit's live `operator()` lookups on the host.
 *
 *  Validates device-side lookups byte-equal the host lookups via CUDA kernels.
 */

#include <cmath>   // `std::abs`
#include <cstring> // `std::memcpy`

#include <iostream>    // RDKit's `AtomTyper.h` needs it but does not include it
#include <random>      // `std::mt19937`, `std::uniform_int_distribution`
#include <sstream>     // RDKit's `AtomTyper.h` needs it but does not include it
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector`

#include <fmt/format.h> // `fmt::println`, `fmt::group_digits`

#include <boost/shared_ptr.hpp>
#include <cuda_runtime.h>
#include <RDGeneral/types.h>
#include <ForceField/MMFF/Params.h>
#include <GraphMol/ForceFieldHelpers/MMFF/AtomTyper.h>

#include "smallfield/parameters/mmff94_tables.hpp"
#include "smallfield/parameters/mmff94_tables_rdkit.hpp"

#include "test/parameters/test_mmff94_tables_kernels.hpp"
#include "test/test_helpers.hpp"

namespace sf = smallfield;
namespace sftest = smallfield::tests;

static constexpr std::string_view test_name_k = "test_mmff94_tables";

namespace smallfield::tests {

static int host_failure_count_ = 0;
static int host_check_count_ = 0;
static int device_failure_count_ = 0;
static int device_check_count_ = 0;

static bool approximately_equal_(float lhs, double rhs, float tolerance = 1e-5f) {
    float const rhs_as_float = static_cast<float>(rhs);
    return std::abs(lhs - rhs_as_float) < tolerance;
}

static bool bitwise_equal_(float lhs, float rhs) {
    std::uint32_t lhs_bits;
    std::uint32_t rhs_bits;
    std::memcpy(&lhs_bits, &lhs, sizeof(lhs_bits));
    std::memcpy(&rhs_bits, &rhs, sizeof(rhs_bits));
    return lhs_bits == rhs_bits;
}

/** RDKit's live lookups are the reference; the extracted tables are what is under test. */
static sftest::diagnostic_reporter_t host_reporter_ {"rdkit", "ours"};
/** The host lookup is the reference; the device kernel reading the uploaded tables is under test. */
static sftest::diagnostic_reporter_t device_reporter_ {"host", "device"};

/*  `comparison_group_t::field` invokes its predicate as `equals(reference, observed)`, so each suite's
 *  comparison is wrapped rather than reimplemented -- the tolerance and the bit test below are the same
 *  two functions the hand-rolled `if` chains used. */
static constexpr auto rdkit_within_tolerance_ = [](double rdkit_value, float our_value) {
    return approximately_equal_(our_value, rdkit_value);
};
static constexpr auto bits_identical_ = [](float host_value, float device_value) {
    return bitwise_equal_(host_value, device_value);
};
static constexpr auto exactly_equal_ = [](auto reference, auto observed) { return reference == observed; };

static void validate_vdw_lookup_(sf::parameters::mmff94_tables_host_t const &host_tables) {
    int const checks_before = host_check_count_;
    auto const host_lookup_view = host_tables.view();
    auto const *vdw_collection = RDKit::MMFF::DefaultParameters::getMMFFVdW();
    for (std::uint8_t atom_type = 1; atom_type <= host_tables.max_atom_type; ++atom_type) {
        auto const *our_entry = sf::parameters::lookup_vdw(host_lookup_view, atom_type);
        auto const *rdkit_entry = (*vdw_collection)(atom_type);
        if (!our_entry && !rdkit_entry) continue;

        ++host_check_count_;
        if (!our_entry || !rdkit_entry) {
            host_reporter_.mismatch("vdw", "atom_type={}", atom_type)
                .field("present", rdkit_entry != nullptr, our_entry != nullptr, exactly_equal_);
            ++host_failure_count_;
            continue;
        }

        auto entry = host_reporter_.mismatch("vdw", "atom_type={}", atom_type);
        entry.field("alpha", rdkit_entry->alpha_i, our_entry->alpha_i, rdkit_within_tolerance_)
            .field("n_effective", rdkit_entry->N_i, our_entry->n_effective, rdkit_within_tolerance_)
            .field("a_scale", rdkit_entry->A_i, our_entry->a_scale, rdkit_within_tolerance_)
            .field("g_factor", rdkit_entry->G_i, our_entry->g_factor, rdkit_within_tolerance_)
            .field("r_star", rdkit_entry->R_star, our_entry->r_star, rdkit_within_tolerance_);
        if (entry.reported_field_count() != 0) ++host_failure_count_;
    }
    fmt::println("  {} entries compared against RDKit", fmt::group_digits(host_check_count_ - checks_before));
    host_reporter_.summarize("vdw");
}

static void validate_bond_lookup_(sf::parameters::mmff94_tables_host_t const &host_tables) {
    int const checks_before = host_check_count_;
    auto const host_lookup_view = host_tables.view();
    if (host_tables.bond.empty()) {
        fmt::println("  NOTE mmff94_tables field=bond_table entries=0 bond_checks=skipped");
        return;
    }

    auto const *bond_collection = RDKit::MMFF::DefaultParameters::getMMFFBond();
    std::mt19937 generator(42);
    std::uniform_int_distribution<std::uint32_t> atom_type_dist(1, host_tables.max_atom_type);
    std::uniform_int_distribution<std::uint32_t> bond_type_dist(0, 1);
    for (int trial_index = 0; trial_index < 5000; ++trial_index) {
        std::uint8_t atom_type_i = static_cast<std::uint8_t>(atom_type_dist(generator));
        std::uint8_t atom_type_j = static_cast<std::uint8_t>(atom_type_dist(generator));
        std::uint8_t bond_type = static_cast<std::uint8_t>(bond_type_dist(generator));
        auto const *our_entry = sf::parameters::lookup_bond(host_lookup_view, bond_type, atom_type_i, atom_type_j);
        auto const *rdkit_entry = (*bond_collection)(bond_type, atom_type_i, atom_type_j);
        if (!our_entry && !rdkit_entry) continue;

        ++host_check_count_;
        if (!our_entry || !rdkit_entry) {
            host_reporter_.mismatch("bond", "bond_type={} atom_i={} atom_j={}", bond_type, atom_type_i, atom_type_j)
                .field("present", rdkit_entry != nullptr, our_entry != nullptr, exactly_equal_);
            ++host_failure_count_;
            continue;
        }

        auto entry = host_reporter_.mismatch("bond", "bond_type={} atom_i={} atom_j={}", bond_type, atom_type_i,
                                             atom_type_j);
        entry.field("force_constant", rdkit_entry->kb, our_entry->force_constant, rdkit_within_tolerance_)
            .field("equilibrium_distance", rdkit_entry->r0, our_entry->equilibrium_distance, rdkit_within_tolerance_);
        if (entry.reported_field_count() != 0) ++host_failure_count_;
    }
    fmt::println("  {} entries compared against RDKit", fmt::group_digits(host_check_count_ - checks_before));
    host_reporter_.summarize("bond");
}

static void validate_angle_lookup_(sf::parameters::mmff94_tables_host_t const &host_tables) {
    int const checks_before = host_check_count_;
    auto const host_lookup_view = host_tables.view();
    auto const *angle_collection = RDKit::MMFF::DefaultParameters::getMMFFAngle();
    auto const *definition_collection = RDKit::MMFF::DefaultParameters::getMMFFDef();
    std::mt19937 generator(137);
    std::uniform_int_distribution<std::uint32_t> atom_type_dist(1, 95);
    std::uniform_int_distribution<std::uint32_t> angle_type_dist(0, 7);
    for (int trial_index = 0; trial_index < 5000; ++trial_index) {
        std::uint8_t atom_type_i = static_cast<std::uint8_t>(atom_type_dist(generator));
        std::uint8_t atom_type_j_center = static_cast<std::uint8_t>(atom_type_dist(generator));
        std::uint8_t atom_type_k = static_cast<std::uint8_t>(atom_type_dist(generator));
        std::uint8_t angle_type = static_cast<std::uint8_t>(angle_type_dist(generator));
        auto const *our_entry = sf::parameters::lookup_angle(host_lookup_view, angle_type, atom_type_i,
                                                             atom_type_j_center, atom_type_k);
        auto const *rdkit_entry = (*angle_collection)(definition_collection, angle_type, atom_type_i,
                                                      atom_type_j_center, atom_type_k);
        if (!our_entry && !rdkit_entry) continue;

        ++host_check_count_;
        if (!our_entry || !rdkit_entry) {
            host_reporter_
                .mismatch("angle", "angle_type={} atom_i={} atom_j={} atom_k={}", angle_type, atom_type_i,
                          atom_type_j_center, atom_type_k)
                .field("present", rdkit_entry != nullptr, our_entry != nullptr, exactly_equal_);
            ++host_failure_count_;
            continue;
        }

        auto entry = host_reporter_.mismatch("angle", "angle_type={} atom_i={} atom_j={} atom_k={}", angle_type,
                                             atom_type_i, atom_type_j_center, atom_type_k);
        entry.field("force_constant", rdkit_entry->ka, our_entry->force_constant, rdkit_within_tolerance_)
            .field("theta0_deg", rdkit_entry->theta0, our_entry->theta0_deg, rdkit_within_tolerance_);
        if (entry.reported_field_count() != 0) ++host_failure_count_;
    }
    fmt::println("  {} entries compared against RDKit", fmt::group_digits(host_check_count_ - checks_before));
    host_reporter_.summarize("angle");
}

/**
 *  @brief Device cross-validation — upload tables, build random queries, launch kernels, compare
 *  bit-wise to host lookup.
 */
static void run_device_lookup_cross_checks_(sf::parameters::mmff94_tables_host_t const &host_tables) {
    sf::parameters::mmff94_tables_view_t device_lookup_view {};
    void *device_handle = sftest::device_tables_create(host_tables, device_lookup_view);
    if (device_handle == nullptr) {
        // Deliberately not the word `FAILED`: this line is narration, and a CI log filtered on the verdict
        // grammar must not answer a `FAIL` grep with a line that never decided anything.
        fmt::println("  NOTE mmff94_tables field=device_upload available=0 device_cross_checks=skipped");
        // Returning here used to skip every `summarize` below, and the vacuous-section count only rises
        // inside `summarize` -- so the whole device cross-validation vanished behind a passing verdict.
        // Closing an empty section is what makes the skip visible to the gate in `main`.
        device_reporter_.summarize("device_upload");
        return;
    }

    auto const host_lookup_view = host_tables.view();
    std::mt19937 generator(2026);

    {
        constexpr int probe_count = 200;
        std::vector<sftest::vdw_query_t> queries(probe_count);
        std::vector<sftest::vdw_result_t> results(probe_count);
        for (int probe_index = 0; probe_index < probe_count; ++probe_index) {
            queries[probe_index].atom_type = static_cast<std::uint8_t>(probe_index % (host_tables.max_atom_type + 1));
        }

        sftest::vdw_query_t *query_device = nullptr;
        sftest::vdw_result_t *result_device = nullptr;
        cudaMalloc(&query_device, probe_count * sizeof(sftest::vdw_query_t));
        cudaMalloc(&result_device, probe_count * sizeof(sftest::vdw_result_t));
        cudaMemcpy(query_device, queries.data(), probe_count * sizeof(sftest::vdw_query_t), cudaMemcpyHostToDevice);
        sftest::launch_device_vdw_lookup(device_lookup_view, query_device, result_device, probe_count);
        cudaDeviceSynchronize();
        cudaMemcpy(results.data(), result_device, probe_count * sizeof(sftest::vdw_result_t), cudaMemcpyDeviceToHost);
        cudaFree(query_device);
        cudaFree(result_device);
        for (int probe_index = 0; probe_index < probe_count; ++probe_index) {
            auto const *host_entry = sf::parameters::lookup_vdw(host_lookup_view, queries[probe_index].atom_type);
            ++device_check_count_;
            bool const host_hit = host_entry != nullptr;
            auto probe = device_reporter_.mismatch("vdw", "atom_type={}", queries[probe_index].atom_type);
            if (host_hit != (results[probe_index].hit != 0)) {
                probe.field("hit", host_hit, results[probe_index].hit != 0, exactly_equal_);
                ++device_failure_count_;
                continue;
            }
            if (!host_hit) continue;
            probe.field("alpha", host_entry->alpha_i, results[probe_index].alpha, bits_identical_)
                .field("n_effective", host_entry->n_effective, results[probe_index].effective_electron_count,
                       bits_identical_)
                .field("r_star", host_entry->r_star, results[probe_index].r_star, bits_identical_);
            if (probe.reported_field_count() != 0) ++device_failure_count_;
        }
        device_reporter_.summarize("vdw");
    }

    {
        constexpr int probe_count = 5000;
        std::vector<sftest::angle_query_t> queries(probe_count);
        std::vector<sftest::angle_result_t> results(probe_count);
        std::uniform_int_distribution<std::uint32_t> atom_type_dist(1, 95);
        std::uniform_int_distribution<std::uint32_t> angle_type_dist(0, 7);
        for (int probe_index = 0; probe_index < probe_count; ++probe_index) {
            queries[probe_index].atom_type_i = static_cast<std::uint8_t>(atom_type_dist(generator));
            queries[probe_index].atom_type_j_center = static_cast<std::uint8_t>(atom_type_dist(generator));
            queries[probe_index].atom_type_k = static_cast<std::uint8_t>(atom_type_dist(generator));
            queries[probe_index].angle_type = static_cast<std::uint8_t>(angle_type_dist(generator));
        }

        sftest::angle_query_t *query_device = nullptr;
        sftest::angle_result_t *result_device = nullptr;
        cudaMalloc(&query_device, probe_count * sizeof(sftest::angle_query_t));
        cudaMalloc(&result_device, probe_count * sizeof(sftest::angle_result_t));
        cudaMemcpy(query_device, queries.data(), probe_count * sizeof(sftest::angle_query_t), cudaMemcpyHostToDevice);
        sftest::launch_device_angle_lookup(device_lookup_view, query_device, result_device, probe_count);
        cudaDeviceSynchronize();
        cudaMemcpy(results.data(), result_device, probe_count * sizeof(sftest::angle_result_t), cudaMemcpyDeviceToHost);
        cudaFree(query_device);
        cudaFree(result_device);
        for (int probe_index = 0; probe_index < probe_count; ++probe_index) {
            auto const *host_entry = sf::parameters::lookup_angle(
                host_lookup_view, queries[probe_index].angle_type, queries[probe_index].atom_type_i,
                queries[probe_index].atom_type_j_center, queries[probe_index].atom_type_k);
            ++device_check_count_;
            bool const host_hit = host_entry != nullptr;
            auto probe = device_reporter_.mismatch("angle", "angle_type={} atom_i={} atom_j={} atom_k={}",
                                                   queries[probe_index].angle_type, queries[probe_index].atom_type_i,
                                                   queries[probe_index].atom_type_j_center,
                                                   queries[probe_index].atom_type_k);
            if (host_hit != (results[probe_index].hit != 0)) {
                probe.field("hit", host_hit, results[probe_index].hit != 0, exactly_equal_);
                ++device_failure_count_;
                continue;
            }
            if (!host_hit) continue;
            probe
                .field("force_constant", host_entry->force_constant, results[probe_index].force_constant,
                       bits_identical_)
                .field("theta0_deg", host_entry->theta0_deg, results[probe_index].equilibrium_angle_degrees,
                       bits_identical_);
            if (probe.reported_field_count() != 0) ++device_failure_count_;
        }
        device_reporter_.summarize("angle");
    }

    {
        constexpr int probe_count = 5000;
        std::vector<sftest::torsion_query_t> queries(probe_count);
        std::vector<sftest::torsion_result_t> results(probe_count);
        std::uniform_int_distribution<std::uint32_t> atom_type_dist(1, 95);
        std::uniform_int_distribution<std::uint32_t> torsion_type_dist(0, 5);
        for (int probe_index = 0; probe_index < probe_count; ++probe_index) {
            queries[probe_index].atom_type_i = static_cast<std::uint8_t>(atom_type_dist(generator));
            queries[probe_index].atom_type_j = static_cast<std::uint8_t>(atom_type_dist(generator));
            queries[probe_index].atom_type_k = static_cast<std::uint8_t>(atom_type_dist(generator));
            queries[probe_index].atom_type_l = static_cast<std::uint8_t>(atom_type_dist(generator));
            queries[probe_index].primary_torsion_type = static_cast<std::uint8_t>(torsion_type_dist(generator));
            queries[probe_index].secondary_torsion_type = static_cast<std::uint8_t>(torsion_type_dist(generator));
        }

        sftest::torsion_query_t *query_device = nullptr;
        sftest::torsion_result_t *result_device = nullptr;
        cudaMalloc(&query_device, probe_count * sizeof(sftest::torsion_query_t));
        cudaMalloc(&result_device, probe_count * sizeof(sftest::torsion_result_t));
        cudaMemcpy(query_device, queries.data(), probe_count * sizeof(sftest::torsion_query_t), cudaMemcpyHostToDevice);
        sftest::launch_device_torsion_lookup(device_lookup_view, query_device, result_device, probe_count);
        cudaDeviceSynchronize();
        cudaMemcpy(results.data(), result_device, probe_count * sizeof(sftest::torsion_result_t),
                   cudaMemcpyDeviceToHost);
        cudaFree(query_device);
        cudaFree(result_device);
        for (int probe_index = 0; probe_index < probe_count; ++probe_index) {
            auto const lookup_result = sf::parameters::lookup_torsion(
                host_lookup_view, queries[probe_index].primary_torsion_type,
                queries[probe_index].secondary_torsion_type, queries[probe_index].atom_type_i,
                queries[probe_index].atom_type_j, queries[probe_index].atom_type_k, queries[probe_index].atom_type_l);
            ++device_check_count_;
            bool const host_hit = lookup_result.second != nullptr;
            auto probe = device_reporter_.mismatch(
                "torsion", "primary_torsion_type={} secondary_torsion_type={} atom_i={} atom_j={} atom_k={} atom_l={}",
                queries[probe_index].primary_torsion_type, queries[probe_index].secondary_torsion_type,
                queries[probe_index].atom_type_i, queries[probe_index].atom_type_j, queries[probe_index].atom_type_k,
                queries[probe_index].atom_type_l);
            if (host_hit != (results[probe_index].hit != 0)) {
                probe.field("hit", host_hit, results[probe_index].hit != 0, exactly_equal_);
                ++device_failure_count_;
                continue;
            }
            if (!host_hit) continue;
            probe
                .field("torsion_v1", lookup_result.second->torsion_v1, results[probe_index].torsion_v1, bits_identical_)
                .field("torsion_v2", lookup_result.second->torsion_v2, results[probe_index].torsion_v2, bits_identical_)
                .field("torsion_v3", lookup_result.second->torsion_v3, results[probe_index].torsion_v3,
                       bits_identical_);
            if (probe.reported_field_count() != 0) ++device_failure_count_;
        }
        device_reporter_.summarize("torsion");
    }

    fmt::println("  device lookups: {} checks, {} mismatches", fmt::group_digits(device_check_count_),
                 fmt::group_digits(device_failure_count_));
    sftest::device_tables_destroy(device_handle);
}

struct lookup_mismatch_counts_t {
    int check_count = 0;
    int failure_count = 0;
};

static void run_device_vdw_smoke_check_(sf::parameters::mmff94_tables_host_t const &host_tables,
                                        lookup_mismatch_counts_t &counts) {
    sf::parameters::mmff94_tables_view_t device_lookup_view {};
    void *device_handle = sftest::device_tables_create(host_tables, device_lookup_view);
    if (device_handle == nullptr) return;

    auto const host_lookup_view = host_tables.view();
    constexpr int probe_count = 120;
    std::vector<sftest::vdw_query_t> queries(probe_count);
    std::vector<sftest::vdw_result_t> results(probe_count);
    for (int probe_index = 0; probe_index < probe_count; ++probe_index) {
        queries[probe_index].atom_type = static_cast<std::uint8_t>(probe_index % (host_tables.max_atom_type + 2));
    }

    sftest::vdw_query_t *query_device = nullptr;
    sftest::vdw_result_t *result_device = nullptr;
    cudaMalloc(&query_device, probe_count * sizeof(sftest::vdw_query_t));
    cudaMalloc(&result_device, probe_count * sizeof(sftest::vdw_result_t));
    cudaMemcpy(query_device, queries.data(), probe_count * sizeof(sftest::vdw_query_t), cudaMemcpyHostToDevice);
    sftest::launch_device_vdw_lookup(device_lookup_view, query_device, result_device, probe_count);
    cudaDeviceSynchronize();
    cudaMemcpy(results.data(), result_device, probe_count * sizeof(sftest::vdw_result_t), cudaMemcpyDeviceToHost);
    cudaFree(query_device);
    cudaFree(result_device);
    for (int probe_index = 0; probe_index < probe_count; ++probe_index) {
        auto const *host_entry = sf::parameters::lookup_vdw(host_lookup_view, queries[probe_index].atom_type);
        ++counts.check_count;
        bool const host_hit = host_entry != nullptr;
        device_reporter_.mismatch("vdw_smoke", "atom_type={}", queries[probe_index].atom_type)
            .field("hit", host_hit, results[probe_index].hit != 0, exactly_equal_);
        if (host_hit != (results[probe_index].hit != 0)) ++counts.failure_count;
    }
    device_reporter_.summarize("vdw_smoke");

    sftest::device_tables_destroy(device_handle);
}

} // namespace smallfield::tests

int main() {
    fmt::println("test_mmff94_tables: host + device MMFF94 table-lookup gate");
    std::fflush(stdout);

    sf::parameters::mmff94_tables_host_t host_tables;
    fmt::println("calling extract_mmff94_tables_from_rdkit...");
    std::fflush(stdout);
    sf::parameters::extract_mmff94_tables_from_rdkit(host_tables);
    fmt::println("extract done: vdw={} bond={} angle={} tor={} oop={} stbn={} dfsb={} def={} pbci={} chg={} " //
                 "hl={} max_at={}",
                 fmt::group_digits(host_tables.vdw.size()), fmt::group_digits(host_tables.bond.size()),
                 fmt::group_digits(host_tables.angle.size()), fmt::group_digits(host_tables.torsion.size()),
                 fmt::group_digits(host_tables.oop.size()), fmt::group_digits(host_tables.stbn.size()),
                 fmt::group_digits(host_tables.dfsb.size()), fmt::group_digits(host_tables.def.size()),
                 fmt::group_digits(host_tables.pbci.size()), fmt::group_digits(host_tables.chg.size()),
                 fmt::group_digits(host_tables.hl.size()), host_tables.max_atom_type);
    std::fflush(stdout);

    fmt::println("  max_atom_type={}, vdw_power={:.3f}", host_tables.max_atom_type, host_tables.vdw_power);
    std::fflush(stdout);

    fmt::println("--- test_vdw ---");
    std::fflush(stdout);
    sftest::validate_vdw_lookup_(host_tables);
    fmt::println("--- test_bond ---");
    std::fflush(stdout);
    sftest::validate_bond_lookup_(host_tables);
    fmt::println("--- test_angle ---");
    std::fflush(stdout);
    sftest::validate_angle_lookup_(host_tables);
    fmt::println("--- device lookups ---");
    std::fflush(stdout);
    sftest::lookup_mismatch_counts_t vdw_counts;
    sftest::run_device_vdw_smoke_check_(host_tables, vdw_counts);
    fmt::println("  VdW-only device: {} checks, {} fails", fmt::group_digits(vdw_counts.check_count),
                 fmt::group_digits(vdw_counts.failure_count));
    sftest::run_device_lookup_cross_checks_(host_tables);

    fmt::println("checks={} failures={} device-checks={} device-failures={}",
                 fmt::group_digits(sftest::host_check_count_), fmt::group_digits(sftest::host_failure_count_),
                 fmt::group_digits(sftest::device_check_count_), fmt::group_digits(sftest::device_failure_count_));
    if (sftest::host_failure_count_ > 0 || sftest::device_failure_count_ > 0)
        return sftest::report_fail(test_name_k, "{} host and {} device lookup mismatches vs RDKit",
                                   sftest::host_failure_count_, sftest::device_failure_count_);
    // A section that compared nothing is indistinguishable from a passing one in the counters above,
    // so the reporters gate on having actually compared something as well as on having agreed.
    if (!sftest::host_reporter_.passed() || !sftest::device_reporter_.passed())
        return sftest::report_fail(test_name_k, "a lookup section mismatched or compared nothing at all");
    return sftest::report_pass(test_name_k);
}
