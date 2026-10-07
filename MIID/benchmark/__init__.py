# MIID/benchmark/__init__.py
#
# Offline predictor for the grading API's validation_score.
#
# The grader never reports per-variation scores back to a miner, so this
# package re-measures what it checks (identity, face count, aspect, variation
# type and intensity, accessory, copy-paste) on images the miner already
# produced and maps the measurements through the published score sheet.
#
# Run it with:  python -m MIID.benchmark --help
# Rules and their sources are documented in scoring.py.
