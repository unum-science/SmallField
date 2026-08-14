"""The stereo harness's own verdict, exercised on synthetic reports.

`bench/validate_stereo.py` needs a GPU-produced shard folder before it can measure anything, so the
part testable without one is the part that decides pass from fail — and that is the part that was
missing. Every case below is a report the harness could produce; the assertion is what it answers.

The 22.9 % and 47.8 % figures are not invented: they are what the shipped PubChem corpus scores
today, and before this gate existed that run returned 0.
"""

import pytest

from validate_stereo import (
    CONTROL_INVERSION_LIMIT,
    GEOMETRY_UNUSABLE_LIMIT,
    MINIMUM_DECLARED,
    SHARD_CHIRAL_INVERSION_LIMIT,
    SHARD_EZ_INVERSION_LIMIT,
    FidelityReport,
    StereoScore,
    control_is_sound,
    shard_is_within_budget,
)


def report_of(
    *,
    chiral_declared: int = 1000,
    chiral_inverted: int = 0,
    ez_declared: int = 600,
    ez_inverted: int = 0,
    rows_scored: int = 1500,
    geometry_unusable: int = 0,
) -> FidelityReport:
    """A report whose control is clean, so every case below turns on the subject alone."""
    report = FidelityReport()
    report.control_chiral = StereoScore(molecules=rows_scored, declared=chiral_declared, inverted=0)
    report.control_ez = StereoScore(molecules=rows_scored, declared=ez_declared, inverted=0)
    report.shard_chiral = StereoScore(molecules=rows_scored, declared=chiral_declared, inverted=chiral_inverted)
    report.shard_ez = StereoScore(molecules=rows_scored, declared=ez_declared, inverted=ez_inverted)
    report.rows_scored = rows_scored
    report.geometry_unusable = geometry_unusable
    return report


def within_budget(report: FidelityReport) -> bool:
    return shard_is_within_budget(
        report, SHARD_CHIRAL_INVERSION_LIMIT, SHARD_EZ_INVERSION_LIMIT, GEOMETRY_UNUSABLE_LIMIT
    )


def test_clean_shard_passes():
    assert within_budget(report_of())


def test_shipped_corpus_chiral_rate_fails():
    """22.878 % of declared centers, which is what the shipped corpus scores and what exited 0."""
    assert not within_budget(report_of(chiral_declared=695, chiral_inverted=159))


def test_shipped_corpus_ez_rate_fails():
    """47.771 % of declared bonds, same corpus, same previously-passing run."""
    assert not within_budget(report_of(ez_declared=157, ez_inverted=75))


@pytest.mark.parametrize("declared", [0, MINIMUM_DECLARED - 1])
def test_too_few_declared_is_a_failure_not_a_clean_zero(declared):
    """`percent` divides by `max(declared, 1)`, so nothing declared renders as a perfect 0.000 %."""
    assert not within_budget(report_of(chiral_declared=declared, chiral_inverted=0))


def test_unusable_geometry_cannot_buy_a_clean_rate():
    """Every row failing the bond-length window leaves the ratio, so the rate alone reads perfect."""
    report = report_of(rows_scored=1500, geometry_unusable=1500)
    assert not within_budget(report)


def test_no_row_scored_is_a_failure():
    assert not within_budget(report_of(rows_scored=0))


def test_missing_rdkit_is_a_skip_not_a_verdict():
    report = report_of(chiral_declared=695, chiral_inverted=159)
    report.skipped = True
    assert within_budget(report)
    assert control_is_sound(report)


def test_control_still_refuses_a_broken_instrument():
    report = report_of()
    report.control_chiral.inverted = int(report.control_chiral.declared * CONTROL_INVERSION_LIMIT / 100.0) + 1
    assert not control_is_sound(report)


def test_control_scoring_nothing_is_a_failure():
    report = report_of()
    report.control_chiral.molecules = 0
    assert not control_is_sound(report)
