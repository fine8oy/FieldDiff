"""Models vendored for this experiment.

The package deliberately has no eager re-exports.  Experiment modules import
the exact implementation file they use, which makes the dependency boundary
auditable and avoids pulling unrelated legacy models into the run.
"""
