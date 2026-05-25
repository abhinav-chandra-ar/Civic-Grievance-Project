"""
DB proof script — run after submitting a grievance with a photo via the browser.

Usage
-----
  python verify_image_upload_db.py

What it checks
--------------
1. Fetches the latest Attachment row.
2. Asserts image_file IS NOT None (bytes reached FileStorage).
3. Asserts the file physically exists on disk.
4. Prints storage_reference, content_hash, file_size_bytes.
5. Prints Grievance.status_metadata.vision_analysis — must show a verdict.

Expected output (PASS)
-----------------------
  [1] Latest Attachment
      id               : 42
      attachment_code  : ATT-2026-XXXXXX
      storage_reference: grievance_attachments/2026/05/25/road_crack.jpg
      original_filename: road_crack.jpg
      content_hash     : <64 hex chars>
      file_size_bytes  : 84210

  [2] image_file field
      image_file.name  : grievance_attachments/2026/05/25/road_crack.jpg
      image_file != None: True
      File on disk     : PASS  (path: /.../.../road_crack.jpg)

  [3] Vision metadata (Attachment)
      image_validation_metadata           : {...}
      image_issue_classification_metadata : {vision_class: ..., vision_confidence: ...}
      image_text_consistency_metadata     : {consistency_verdict: supports|contradicts|uncertain}

  [4] vision_analysis in Grievance.status_metadata
      vision_analysis: {
        'vision_class': 'road_damage',
        'confidence': 0.72,
        'consistency_verdict': 'supports',
        'provider': 'clip_vit_b32',
        'attachment_code': 'ATT-2026-XXXXXX'
      }

  ALL CHECKS PASSED
"""
from __future__ import annotations

import os
import sys
import django

# ---------------------------------------------------------------------------
# Bootstrap Django
# ---------------------------------------------------------------------------
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "grievance_core.settings.dev")
django.setup()

# ---------------------------------------------------------------------------
# Imports (after setup)
# ---------------------------------------------------------------------------
from pathlib import Path

from apps.attachments.models import Attachment

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SEP = "-" * 60
PASS_TAG = "\033[32mPASS\033[0m"
FAIL_TAG = "\033[31mFAIL\033[0m"
failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    tag = PASS_TAG if condition else FAIL_TAG
    suffix = f"  ({detail})" if detail else ""
    print(f"      {tag}  {label}{suffix}")
    if not condition:
        failures.append(label)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(SEP)
    print("  grievance-core  —  image upload DB proof")
    print(SEP)

    try:
        a = Attachment.objects.latest("id")
    except Attachment.DoesNotExist:
        print(FAIL_TAG + "  No Attachment rows found.  Submit a grievance with a photo first.")
        sys.exit(1)

    # [1] Basic attachment info
    print(f"\n[1] Latest Attachment")
    print(f"      id               : {a.id}")
    print(f"      attachment_code  : {a.attachment_code}")
    print(f"      storage_reference: {a.storage_reference}")
    print(f"      original_filename: {a.original_filename}")
    print(f"      content_hash     : {a.content_hash}")
    print(f"      file_size_bytes  : {a.file_size_bytes:,}")

    # [2] image_file field
    print(f"\n[2] image_file field")
    has_image_file = bool(a.image_file and a.image_file.name)
    print(f"      image_file.name  : {a.image_file.name if has_image_file else 'None'}")
    check("image_file != None", has_image_file, "bytes reached Django FileStorage")

    if has_image_file:
        try:
            disk_path = Path(a.image_file.path)
            file_exists = disk_path.exists()
        except Exception:
            file_exists = False
            disk_path = Path("<unavailable>")
        check("File exists on disk", file_exists, str(disk_path))

    # [3] Vision metadata on Attachment
    print(f"\n[3] Vision metadata (Attachment)")
    img_val  = a.image_validation_metadata
    img_cls  = a.image_issue_classification_metadata
    img_cons = a.image_text_consistency_metadata

    check(
        "image_validation_metadata populated",
        bool(img_val),
        repr(img_val)[:80] if img_val else "empty dict",
    )
    check(
        "image_issue_classification_metadata populated",
        bool(img_cls),
        repr(img_cls)[:80] if img_cls else "empty dict",
    )
    check(
        "image_text_consistency_metadata populated",
        bool(img_cons),
        repr(img_cons)[:80] if img_cons else "empty dict",
    )

    provider = (img_cls or {}).get("vision_provider") or (img_val or {}).get("provider", "")
    check("vision_provider != local_stub_v1", provider != "local_stub_v1", f"provider={provider!r}")

    # [4] vision_analysis in Grievance.status_metadata
    print(f"\n[4] vision_analysis in Grievance.status_metadata")
    g = a.grievance
    print(f"      Grievance        : {g.tracking_code}  status={g.status}")
    sm = g.status_metadata or {}
    vision = sm.get("vision_analysis")
    check("vision_analysis key present", vision is not None, repr(vision)[:120] if vision else "missing")
    if vision:
        verdict = vision.get("consistency_verdict", "")
        check(
            "consistency_verdict set",
            verdict in {"supports", "contradicts", "uncertain"},
            f"verdict={verdict!r}",
        )
        print(f"      vision_analysis  : {vision}")

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print(f"\n{SEP}")
    if failures:
        print(f"  {FAIL_TAG}  {len(failures)} check(s) failed:")
        for f in failures:
            print(f"    - {f}")
        sys.exit(1)
    else:
        print(f"  \033[32mALL CHECKS PASSED\033[0m")
    print(SEP)


if __name__ == "__main__":
    main()
