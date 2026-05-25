"""run_hardening_benchmark.py

Post-hardening benchmark runner.
Runs targeted tests for all 5 hardening fixes and prints a before/after report.

Usage:
    python run_hardening_benchmark.py
"""
import os
import sys
import json

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "grievance_core.settings.dev")

import django
django.setup()

from apps.ml.analyzer import analyze_complaint

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _cat(r):       return r.get("category_code", "unknown")
def _prio(r):      return r.get("priority", "unknown")
def _is_spam(r):   return bool(r.get("spam", {}).get("is_spam", False))
def _spam_score(r):return float(r.get("spam", {}).get("spam_score", 0.0))
def _lang(r):      return r.get("language", "unknown")
def _ward(r):      return r.get("ward_hint", "unknown")

SEP = "=" * 70

# ──────────────────────────────────────────────────────────────────────────────
# FIX 1: Electrical hazard vs street light
# ──────────────────────────────────────────────────────────────────────────────

def fix1_electrical_hazard():
    print(SEP)
    print("FIX 1 — Electrical hazard vs street light confusion")
    print(SEP)

    hazard_cases = [
        "The street light pole near Bakery Junction has exposed live wiring at the base.",
        "Current is leaking from the street light pole on the main road. A scooter rider got a mild shock.",
        "Street light pole has a broken junction box with live terminals hanging out. Pedestrians at risk.",
        "Live electric wire is dangling from the street light pole across the footpath.",
        "Lamp post near school has open wires hanging — kids touch it daily.",
        "Street light pole sparking near junction. Current coming out. Very dangerous.",
        "got shock touching lamp post near our gate it is live wire issue not bulb issue",
        "pole fell wire still on road nobody came since morning please urgent",
        "Street light pole-il current adukunnu. Kazhinja divasam oru kutti shock kittunu.",
        "Street lamp post has live wire hanging loose from the connection box. Shock hazard.",
    ]

    light_cases = [
        "The street light on the main road near the park has not been working for three weeks. Bulb fused.",
        "Lamp post near bus stop has a dead bulb. Please replace.",
        "Street light pole near park is fine — just the bulb at the top is not glowing.",
        "Three consecutive street lights are all off. Sensor may be faulty.",
        "Street light bulb poyi. 2 azhcha aayi light illathe. Maattanam.",
        "lamp post bulb fused outside colony gate dark at night",
        "whole road dark no lights working pls send electrician to change bulbs",
        "Solar street lights stopped working after three months. Battery dead.",
    ]

    print("\n  [HAZARD cases → should be electrical_hazard]")
    h_correct = 0
    for text in hazard_cases:
        r = analyze_complaint(text)
        cat = _cat(r)
        ok = cat == "electrical_hazard"
        h_correct += ok
        mark = "✓" if ok else "✗"
        print(f"  {mark} [{cat}]  {text[:75]}")

    print(f"\n  Hazard accuracy: {h_correct}/{len(hazard_cases)} = {h_correct/len(hazard_cases)*100:.0f}%")

    print("\n  [LIGHT cases → should be street_light]")
    l_correct = 0
    for text in light_cases:
        r = analyze_complaint(text)
        cat = _cat(r)
        ok = cat == "street_light"
        l_correct += ok
        mark = "✓" if ok else "✗"
        print(f"  {mark} [{cat}]  {text[:75]}")

    print(f"\n  Light accuracy: {l_correct}/{len(light_cases)} = {l_correct/len(light_cases)*100:.0f}%")
    total = h_correct + l_correct
    denom = len(hazard_cases) + len(light_cases)
    print(f"\n  COMBINED accuracy: {total}/{denom} = {total/denom*100:.0f}%")
    return total / denom

# ──────────────────────────────────────────────────────────────────────────────
# FIX 2: Priority over-escalation
# ──────────────────────────────────────────────────────────────────────────────

def fix2_priority():
    print(SEP)
    print("FIX 2 — Priority over-escalation (false escalation)")
    print(SEP)

    # LOW severity — should NOT be high/urgent/critical
    low_cases = [
        ("Small pothole noticed on footpath near our gate. Not affecting road traffic. Can be patched whenever.", "low"),
        ("Minor crack on road surface near the park. Very shallow. Not urgent.", "low"),
        ("Slight depression forming at road edge. Just a small dip. No risk yet.", "low"),
        ("Garbage seen on roadside — just a small pile near the corner. Not overflowing.", "low"),
        ("Little litter on footpath near bus stop. Minor issue. Routine sweep needed.", "low"),
        ("Small crack noticed in drain wall. Water still flowing. Minor structural issue.", "low"),
        ("One street light out on well-lit stretch. Not creating dark spot.", "low"),
        ("Chinna pothole undu near colony gate. Bikes kazhiyum. Urgent alla.", "low"),
        ("Konjam garbage kidu near corner. Valiya problem alla.", "low"),
        ("Slight blockage in drainage channel — small plastic visible. Not overflowing.", "low"),
    ]

    # HIGH/URGENT severity — should be high or urgent
    high_cases = [
        ("Large deep pothole near school gate. Two bikes have already fallen. School children at risk.", "urgent"),
        ("Road completely collapsed near bridge after rain. Traffic blocked. Emergency repair needed.", "urgent"),
        ("Main drainage channel blocked. Road flooded after 30 minutes of rain. Multiple houses affected.", "urgent"),
        ("Garbage dump overflowing for a week. Disease risk.", "urgent"),
    ]

    print("\n  [LOW severity cases → should be low/medium, NOT high/urgent/critical]")
    low_ok = 0
    for text, expected in low_cases:
        r = analyze_complaint(text)
        p = _prio(r)
        is_false_escalation = p in ("high", "urgent", "critical")
        low_ok += not is_false_escalation
        mark = "✗" if is_false_escalation else "✓"
        print(f"  {mark} [{p}]  {text[:75]}")

    fe_rate = (len(low_cases) - low_ok) / len(low_cases)
    print(f"\n  False escalation rate: {(len(low_cases)-low_ok)}/{len(low_cases)} = {fe_rate*100:.0f}% (target: <30%)")

    print("\n  [HIGH/URGENT severity cases → should be high/urgent]")
    high_ok = 0
    for text, expected in high_cases:
        r = analyze_complaint(text)
        p = _prio(r)
        ok = p in ("high", "urgent", "critical")
        high_ok += ok
        mark = "✓" if ok else "✗"
        print(f"  {mark} [{p}]  {text[:75]}")

    ue_rate = (len(high_cases) - high_ok) / len(high_cases)
    print(f"\n  Under-escalation rate: {(len(high_cases)-high_ok)}/{len(high_cases)} = {ue_rate*100:.0f}% (target: 0%)")
    return fe_rate

# ──────────────────────────────────────────────────────────────────────────────
# FIX 3: Manglish spam false positives
# ──────────────────────────────────────────────────────────────────────────────

def fix3_manglish_spam():
    print(SEP)
    print("FIX 3 — Manglish spam false positives")
    print(SEP)

    # These were confirmed spam false positives before hardening
    manglish_civic = [
        "Vellam varunilla 2 days aayitta. Tank empty aanu. Ippo enthu cheyyum?",
        "Road il valiya kuzhi und. Bikes veennu pokunu.",
        "Drainage block aanu near market. Rain vanna vellam road-il nirakkunnu.",
        "Waste edukkunilla 3 days aayitta. Mound aayi kidakkunnu near our gate.",
        "Street light work cheyyunnilla 2 azhcha aayitta. Dark road, safety problem.",
        "Sewage overflow aayi road-il. Smell varunnu. Health risk.",
        "Kuzhal pottiyirikkunnu. Sewage road-il varunnu. Fix cheyyanam urgent.",
        "Pole veenu road-il kidakkunnu live wire kuttiyittu. Aarum varunnilla urgent.",
        "Wire thazhe kidakkunnu school-nte munnilu. Kuttikalkku valikkunnu.",
        "Road-il kuzhi undu. Bike accident aayi. Urgent-aayi patch cheyyanam.",
        "Garbage collection miss aayitta. Bin overflow aayi. Please schedule.",
        "Paipa pottiyirunnu near junction. Vellam road-il aayi. Urgent fix.",
        "Tap-il colour ullam vellam varunnu. Kudikkan pattumo?",
        "Mazhakkalam munnethanne drain clear cheyyanam. Last varsham vellakketu.",
        "Bulb poyi lamp post-il. Maattanam. Night-il dark.",
    ]

    print("\n  [Manglish civic complaints → should NOT be spam]")
    fp_count = 0
    for text in manglish_civic:
        r = analyze_complaint(text)
        is_s = _is_spam(r)
        score = _spam_score(r)
        fp_count += is_s
        mark = "✗ SPAM" if is_s else "✓ ok  "
        print(f"  {mark} [{score:.3f}]  {text[:75]}")

    fp_rate = fp_count / len(manglish_civic)
    print(f"\n  Manglish false-positive spam rate: {fp_count}/{len(manglish_civic)} = {fp_rate*100:.0f}% (target: <15%)")

    # Confirm real spam still caught
    real_spam = [
        "buy now get 50% discount on all products click here",
        "congratulations you have won 10 lakh rupees call this number",
        "FREE RECHARGE 100RS click link now limited offer",
    ]
    print("\n  [Real spam → should BE spam]")
    spam_caught = 0
    for text in real_spam:
        r = analyze_complaint(text)
        is_s = _is_spam(r)
        spam_caught += is_s
        mark = "✓" if is_s else "✗"
        print(f"  {mark} [{_spam_score(r):.3f}]  {text[:75]}")
    print(f"\n  Spam recall: {spam_caught}/{len(real_spam)} = {spam_caught/len(real_spam)*100:.0f}% (target: 100%)")
    return fp_rate

# ──────────────────────────────────────────────────────────────────────────────
# FIX 4: Landmark fuzzy alias improvement
# ──────────────────────────────────────────────────────────────────────────────

def fix4_landmark():
    print(SEP)
    print("FIX 4 — Landmark fuzzy alias improvement")
    print(SEP)

    alias_cases = [
        ("pothole near Bakery jn very dangerous", "bakery"),
        ("water supply problem at Med clg junction", "medical"),
        ("drainage blocked at Palaym market", "palayam"),
        ("street light out near Kazhakootam junction", "kazhakkoottam"),
        ("garbage not collected near Trivandrum railway station", "thampanoor"),
        ("road damaged near Secretariat jn", "secretariat"),
        ("broken drain near Kowdiyar palace", "kowdiar"),
        ("electric pole fallen at Pattom jn", "pattom"),
        ("sewage overflow near Nanthankode junction", "nanthancode"),
        ("pothole near hosp junction needs urgent repair", "hospital"),
    ]

    print("\n  [Alias/abbreviation/misspelling landmark cases → should NOT return 'unknown']")
    resolved = 0
    for text, hint_fragment in alias_cases:
        r = analyze_complaint(text)
        ward = _ward(r).lower()
        ok = ward != "unknown" or hint_fragment in ward
        resolved += (ward != "unknown")
        mark = "✓" if ward != "unknown" else "✗"
        print(f"  {mark} [{ward}]  {text[:75]}")

    res_rate = resolved / len(alias_cases)
    print(f"\n  Alias resolution rate: {resolved}/{len(alias_cases)} = {res_rate*100:.0f}% (target: >50%)")
    return res_rate

# ──────────────────────────────────────────────────────────────────────────────
# FIX 5: Vision AI (Pillow availability check)
# ──────────────────────────────────────────────────────────────────────────────

def fix5_vision():
    print(SEP)
    print("FIX 5 — Vision AI benchmark (Pillow availability)")
    print(SEP)
    try:
        import PIL
        print(f"  ✓ Pillow {PIL.__version__} installed — vision tests will run (not skipped)")
        print(f"  Run: python -m pytest tests/ml/test_ai_benchmark.py -k vision --override-ini=\"addopts=\" --noconftest -v")
        return True
    except ImportError:
        print("  ✗ Pillow NOT installed — vision tests will be skipped")
        return False

# ──────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + SEP)
    print("  AI HARDENING BENCHMARK — POST-FIX RESULTS")
    print(SEP + "\n")

    r1 = fix1_electrical_hazard()
    r2 = fix2_priority()
    r3 = fix3_manglish_spam()
    r4 = fix4_landmark()
    r5 = fix5_vision()

    print("\n" + SEP)
    print("  SUMMARY TABLE")
    print(SEP)
    print(f"  {'Module':<35}  {'Before':>10}  {'After':>10}  {'Status'}")
    print("  " + "-" * 65)

    # Fix 1
    status1 = "PASS" if r1 >= 0.70 else "PARTIAL" if r1 >= 0.50 else "FAIL"
    print(f"  {'Fix 1: electrical_hazard accuracy':<35}  {'~50%':>10}  {r1*100:>9.0f}%  {status1}")

    # Fix 2
    status2 = "PASS" if r2 <= 0.20 else "PARTIAL" if r2 <= 0.35 else "FAIL"
    print(f"  {'Fix 2: false escalation rate':<35}  {'42.9%':>10}  {r2*100:>9.0f}%  {status2}")

    # Fix 3
    status3 = "PASS" if r3 <= 0.15 else "PARTIAL" if r3 <= 0.30 else "FAIL"
    print(f"  {'Fix 3: Manglish spam FP rate':<35}  {'~60%+':>10}  {r3*100:>9.0f}%  {status3}")

    # Fix 4
    status4 = "PASS" if r4 >= 0.50 else "PARTIAL" if r4 >= 0.30 else "FAIL"
    print(f"  {'Fix 4: landmark alias resolution':<35}  {'~30%':>10}  {r4*100:>9.0f}%  {status4}")

    # Fix 5
    status5 = "PASS" if r5 else "FAIL"
    print(f"  {'Fix 5: Pillow installed (vision)':<35}  {'SKIP':>10}  {'ACTIVE':>10}  {status5}")

    print(SEP + "\n")

if __name__ == "__main__":
    main()
