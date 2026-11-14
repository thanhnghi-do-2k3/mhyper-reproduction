from __future__ import annotations

import hashlib

UPSTREAM_REPOSITORY = "https://github.com/zjukg/M-Hyper"
UPSTREAM_SHA = "cb390b6a54e08a6ac76d664cf2c2f6e1ff13ff76"
RELEASED_PATCH_SHA = "NONE"
EVALUATION_CORRECTED_PATCH_SHA = hashlib.sha256(
    b"correct-validation-split;deterministic-train-holdout;device-injection;typed-metrics;no-dynamic-exec"
).hexdigest()


def implementation_fingerprint(variant: str) -> str:
    patch_sha = {
        "released_code": RELEASED_PATCH_SHA,
        "evaluation_corrected": EVALUATION_CORRECTED_PATCH_SHA,
        "research_variant": EVALUATION_CORRECTED_PATCH_SHA,
    }.get(variant, "lab-smoke")
    return hashlib.sha256(f"{UPSTREAM_SHA}|{patch_sha}|{variant}".encode()).hexdigest()
