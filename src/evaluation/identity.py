"""Identifiers for the paper method and its evaluation datasets."""
BENCHMARKS = ("aime2024", "aime2025", "gpqa_diamond", "livecodebench_v6")
def validate_identity(method, report_method=None):
    if method != "trajpd" or report_method not in (None, "trajpd"):
        raise ValueError("Expected the paper method identifier")
    return method
