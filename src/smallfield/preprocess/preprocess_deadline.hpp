/**
 *  @file preprocess_deadline.hpp
 *  @brief Stack-owned per-`molecule` soft deadline for the `preprocess-step`.
 *
 *  The CSD-torsion dispatcher's `RDKit::SubstructMatch` occasionally drops
 *  into VF2 worst-case on specific polycyclic-aromatic molecules and does
 *  not terminate within any practical wall time.  A stuck worker stalls
 *  the OMP-for `preprocess-step`, freezes the 4-role pipeline, and idles
 *  every GPU until the wrapper script externally kills the process.
 *
 *  Each OMP-for iteration stack-allocates a `preprocess_deadline_t`, passes
 *  a pointer through `prepare_molecule_into` → `extract_etk_torsions_native`,
 *  and the `extraAtomCheck` lambda captures the pointer by value.  Every
 *  VF2 atom comparison consults `exceeded`; once past the deadline the
 *  callback returns `false` and VF2 unwinds in O(candidates) rather than
 *  the original unbounded recursion — no exceptions through RDKit
 *  internals, no thread cancellation, no process signals.
 *
 *  The `tripped` flag latches on the first `exceeded` trigger so the
 *  OMP-for body can query it post-facto and decide whether to drop the
 *  partial scratch and record the `molecule` for
 *  `conformer_status_t::preprocess_timeout`.  Because each struct instance
 *  lives on exactly one OMP worker's stack, no `thread_local` or
 *  process-wide state is needed.
 */

#pragma once

#include "smallfield/module_fwd.hpp"

#include <chrono>

namespace smallfield::preprocess {

/**
 *  @brief Per-`molecule` soft deadline + latched tripped-flag.  Instantiated
 *  once per iteration on an OMP worker's stack and passed by pointer into
 *  the preprocess-dispatch call chain; a `nullptr` pointer means "no
 *  deadline" and every `exceeded` call returns false.
 */
struct preprocess_deadline_t {
    std::chrono::steady_clock::time_point deadline {};
    bool tripped = false;

    /** @brief Stamp the deadline `budget` in the future and clear the tripped flag. */
    void reset(std::chrono::milliseconds budget) noexcept {
        deadline = std::chrono::steady_clock::now() + budget;
        tripped = false;
    }

    /** @brief True once the current steady-clock time is past the deadline; latches tripped. */
    bool exceeded() noexcept {
        if (std::chrono::steady_clock::now() > deadline) {
            tripped = true;
            return true;
        }
        return false;
    }
};

/** @brief Convenience: `d == nullptr` → never exceeded; else delegates to `d->exceeded()`. */
inline bool exceeded_or_null(preprocess_deadline_t *d) noexcept { return d != nullptr && d->exceeded(); }

} // namespace smallfield::preprocess
