/**
 *  @file test/cli/test_shard_sealing.cpp
 *  @brief Pin the two ways an input shard leaves the set a later run is offered: its sentinel, and parking.
 *
 *  Completeness downstream is the `.encoded` sentinel and nothing else, so a shard that ends a run with
 *  neither a sentinel nor a move out of the corpus is offered again on every pass, forever, while the run
 *  logs success. Two shapes of that were live here. A sentinel whose write failed reported nothing a caller
 *  could count, so a shard whose parquet was durable and whose seal was not was counted complete. And parking
 *  renamed the path it was handed, which under the production runner is a symlink into a staging view -- so a
 *  shard that faulted the driver had its link moved, its file left where the corpus is globbed, and its fault
 *  repeated on the next pass.
 *
 *  Both are filesystem behaviour, so both are asserted against a real temporary tree rather than a fake.
 */

#include <cstdio> // `std::fopen`

#include <filesystem>   // `std::filesystem::path`
#include <string_view>  // `std::string_view`
#include <system_error> // `std::error_code`

#include <fmt/format.h> // `fmt::println`

#include "cli/shard_inventory.hpp"

#include "test/test_helpers.hpp"

namespace fs = std::filesystem;
namespace sfcli = smallfield::cli;
namespace sftest = smallfield::tests;

static constexpr std::string_view test_name_k = "test_shard_sealing";

namespace smallfield::tests {

/** Write an empty file at @p path, standing in for a shard whose bytes nothing here reads. */
static void touch_(fs::path const &path) {
    FILE *file = std::fopen(path.string().c_str(), "wb");
    sf_verify_(file != nullptr, fmt::format("the probe tree could not be populated at {}", path.string()));
    std::fclose(file);
}

#pragma region Sealing

/** A sentinel that wrote reports it, one that could not reports which half stopped it, and neither lies. */
static void check_sentinel_outcomes_(fs::path const &directory, diagnostic_reporter_t &reporter) {
    fs::path const sealed = directory / "sealed.3D.parquet";
    touch_(sealed);
    reporter.mismatch("sentinel", "case=writable")
        .field("outcome", static_cast<int>(sfcli::sentinel_write_t::sealed_k),
               static_cast<int>(sfcli::write_encoded_sentinel(sealed)), exactly_equal);

    std::error_code error_code;
    fs::path sentinel = sealed;
    sentinel += sfcli::encoded_sentinel_suffix_k;
    reporter.mismatch("sentinel", "case=writable")
        .field("sentinel_exists", true, fs::exists(sentinel, error_code), exactly_equal);

    // A directory occupying the sentinel's own name is the cheapest way to fail the `fopen` alone: the shard
    // beside it is written and durable, which is exactly the case that used to count as complete.
    fs::path const blocked = directory / "blocked.3D.parquet";
    touch_(blocked);
    fs::path blocked_sentinel = blocked;
    blocked_sentinel += sfcli::encoded_sentinel_suffix_k;
    fs::create_directories(blocked_sentinel, error_code);
    sf_verify_(!error_code, "the probe tree could not block a sentinel path with a directory");
    reporter.mismatch("sentinel", "case=unwritable")
        .field("outcome", static_cast<int>(sfcli::sentinel_write_t::open_failed_k),
               static_cast<int>(sfcli::write_encoded_sentinel(blocked)), exactly_equal);
}

#pragma endregion Sealing

#pragma region Parking

/**
 *  @brief Parking a staging symlink must move the shard it names, not the link.
 *
 *  The production runner rebuilds its view of symlinks on every pass, so a link moved aside is recreated and
 *  the shard behind it is offered again -- which is the loop this asserts is closed. The `_skip` folder is a
 *  sibling of the shard's own directory rather than of the view's, or the corpus would keep it.
 */
static void check_parking_resolves_the_link_(fs::path const &directory, diagnostic_reporter_t &reporter) {
    fs::path const corpus = directory / "corpus";
    fs::path const view = directory / "ready";
    std::error_code error_code;
    fs::create_directories(corpus, error_code);
    fs::create_directories(view, error_code);
    sf_verify_(!error_code, "the probe tree could not be given a corpus and a staging view");

    fs::path const shard = corpus / "faulted.parquet";
    fs::path const link = view / "faulted.parquet";
    touch_(shard);
    fs::create_symlink(shard, link, error_code);
    sf_verify_(!error_code, "the probe tree could not be given a staging symlink");

    reporter.mismatch("park", "case=staging_symlink")
        .field("outcome", static_cast<int>(sfcli::park_outcome_t::parked_k),
               static_cast<int>(sfcli::park_input_shard(link)), exactly_equal);

    fs::path const parked = directory / "corpus_skip" / "faulted.parquet";
    reporter.mismatch("park", "case=staging_symlink")
        .field("shard_in_corpus", false, fs::exists(shard, error_code), exactly_equal)
        .field("shard_in_skip", true, fs::exists(parked, error_code), exactly_equal)
        // A dangling link left in the view would be offered again by an inventory that resolves it.
        .field("link_in_view", false, fs::is_symlink(link, error_code), exactly_equal)
        // The view is not the corpus, so nothing of it is renamed and no `_skip` of it is created.
        .field("view_skip_created", false, fs::exists(directory / "ready_skip", error_code), exactly_equal);
}

/** Pointed straight at a corpus, parking moves the file itself, which is what it always claimed to do. */
static void check_parking_moves_a_plain_file_(fs::path const &directory, diagnostic_reporter_t &reporter) {
    fs::path const corpus = directory / "flat";
    std::error_code error_code;
    fs::create_directories(corpus, error_code);
    fs::path const shard = corpus / "plain.parquet";
    touch_(shard);

    reporter.mismatch("park", "case=plain_file")
        .field("outcome", static_cast<int>(sfcli::park_outcome_t::parked_k),
               static_cast<int>(sfcli::park_input_shard(shard)), exactly_equal);
    reporter.mismatch("park", "case=plain_file")
        .field("shard_in_corpus", false, fs::exists(shard, error_code), exactly_equal)
        .field("shard_in_skip", true, fs::exists(directory / "flat_skip" / "plain.parquet", error_code), exactly_equal);
}

/** A shard that could not be moved is still offered, so the refusal has to reach the caller's counter. */
static void check_parking_reports_a_refusal_(fs::path const &directory, diagnostic_reporter_t &reporter) {
    reporter.mismatch("park", "case=absent_shard")
        .field("outcome", static_cast<int>(sfcli::park_outcome_t::rename_failed_k),
               static_cast<int>(sfcli::park_input_shard(directory / "flat" / "never_existed.parquet")), exactly_equal);
}

#pragma endregion Parking

} // namespace smallfield::tests

int main() {
    fmt::println("{}: a sentinel reports whether it sealed, and parking moves the shard rather than the link",
                 test_name_k);

    std::error_code error_code;
    fs::path const directory = fs::temp_directory_path(error_code) / "smallfield_test_shard_sealing";
    sf_verify_(!error_code, "no temporary directory is available to build a probe tree in");
    fs::remove_all(directory, error_code);
    fs::create_directories(directory, error_code);
    sf_verify_(!error_code, fmt::format("probe tree could not be created: {}", directory.string()));

    sftest::diagnostic_reporter_t reporter("expected", "observed");
    sftest::check_sentinel_outcomes_(directory, reporter);
    reporter.summarize("sentinel_outcomes");
    sftest::check_parking_resolves_the_link_(directory, reporter);
    sftest::check_parking_moves_a_plain_file_(directory, reporter);
    sftest::check_parking_reports_a_refusal_(directory, reporter);
    reporter.summarize("shard_parking");

    fs::remove_all(directory, error_code);
    if (!reporter.passed())
        return sftest::report_fail(test_name_k, "{} mismatches over {} comparisons", reporter.mismatch_count(),
                                   reporter.comparison_count());
    return sftest::report_pass(test_name_k);
}
