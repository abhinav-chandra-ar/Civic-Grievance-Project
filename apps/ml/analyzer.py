"""apps/ml/analyzer.py

Lightweight civic intelligence engine for Thiruvananthapuram (TVMC) grievance text.

Design constraints
------------------
* Pure functions only — no Django imports, no database access, no side effects.
* Input: raw complaint text in any combination of English, Malayalam Unicode,
  or Manglish (Malayalam written in Latin script).
* Output: structured analysis dicts consumed by the NLP adapter at
  apps/integrations/clients/nlp.py.

Function inventory
------------------
detect_language()           → script and language identification
normalize_text()            → Unicode NFKC normalisation + whitespace cleanup
classify_issue()            → civic category classification (9 categories)
detect_department()         → category-code → department-code routing
extract_landmarks()         → location alias matching with ward hints
predict_priority()          → severity inference (low/medium/high/urgent/critical)
detect_spam()               → gibberish, repetition and abuse detection
detect_possible_duplicate() → token-similarity duplicate heuristic
score_analysis()            → weighted confidence aggregation
analyze_complaint()         → main orchestrator — calls all of the above
"""
from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from typing import Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ML inference thresholds
# ---------------------------------------------------------------------------
# When ML confidence ≥ _ML_PRIMARY_THRESHOLD, the ML prediction is used as-is.
# When ML confidence is in [_ML_BLEND_THRESHOLD, _ML_PRIMARY_THRESHOLD), the ML
# and rule-engine results are blended (confidence-weighted average).
# When ML confidence < _ML_BLEND_THRESHOLD, the rule engine result is used.

_ML_PRIMARY_THRESHOLD = 0.55
_ML_BLEND_THRESHOLD   = 0.30

# ---------------------------------------------------------------------------
# Unicode helpers
# ---------------------------------------------------------------------------

# Malayalam Unicode block U+0D00–U+0D7F
_MALAYALAM_RE = re.compile(r"[ഀ-ൿ]")
_LATIN_RE = re.compile(r"[a-zA-Z]")

# ---------------------------------------------------------------------------
# Manglish signal words
# Common words that appear when Malayalam is written in Latin script.
# ---------------------------------------------------------------------------

_MANGLISH_SIGNALS: frozenset[str] = frozenset({
    # existence / negation
    "und", "undo", "undu", "undakkum", "undakki",
    "illa", "ille", "illayo",
    # action verbs
    "varunilla", "varunundo", "pokunilla", "cheyyunilla", "kittunnilla",
    "thalli", "kalliyittu", "kondupoi", "thakarnnu", "veennu",
    # pronouns / discourse words
    "njan", "njangal", "avide", "evidey", "ingane", "enganey",
    "enthu", "enthanu", "enthokke",
    # connectors / question markers
    "alle", "alledo", "ano", "anu", "aanu", "veno", "venda",
    # civic-complaint Manglish fragments
    "kuzhi", "chori", "kazhivu", "mala", "vellam", "kambhi",
    "thadangi", "mudangi", "block", "nokkiyille",
    # copulas / tense suffixes
    "annu", "aayirunnilla", "aayi", "aayittu",
    # Additional past-tense / state suffixes common in civic Manglish
    "potti", "adangi", "kavinju", "nikkunnu", "oodunnu",
    "thirinju", "keduthu", "poyyi", "veennu",
    # Water / pipe / drainage Manglish fragments
    "kuzhal", "odha", "chala", "theruva", "malam", "septic",
    "vilakku", "thirikku", "odunnu",
})

# ---------------------------------------------------------------------------
# Issue classification keywords
# Flat tuple per category — English + Manglish + Malayalam Unicode.
# ---------------------------------------------------------------------------

_ISSUE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "road_damage": (
        "pothole", "road damage", "road broken", "road repair",
        "damaged road", "road cave", "road crack", "asphalt", "tar broken",
        "manhole", "road sinking", "road dug",
        "road kuzhi", "vazhiyil kuzhi", "kuzhi und", "kuzhi undu",
        "road thorannu", "road baadhayundu",
        "റോഡ്", "കുഴി", "ടാർ", "ആസ്ഫാൽറ്റ്",
    ),
    "waste_management": (
        "garbage", "waste", "trash", "dump", "litter", "rubbish",
        "refuse", "bin overflow", "garbage pile", "solid waste",
        "mala", "malineyam", "cheti", "mala thalli", "mala nirakki",
        "mala edukkunilla",
        "മാലിന്യം", "ചവർ", "കൂടം",
    ),
    "water_supply": (
        "water supply", "pipe burst", "water leak", "no water",
        "water shortage", "pipe broken", "tap water", "drinking water",
        "water contamination", "muddy water", "water cut",
        "vellam varunilla", "vellam illa", "vellam chori", "pipe pottannu",
        "vellam mudangi", "vellam malinambaayi",
        "വെള്ളം", "പൈപ്പ്", "ജലം", "കുടിവെള്ളം",
        "വെള്ളം വരുന്നില്ല",
    ),
    "street_light": (
        "street light", "streetlight", "lamp post", "light not working",
        "no light", "dark road", "lamp broken", "light failure",
        "street light illa", "light illa", "lamp thakarnnu", "andharam",
        "തെരുവ് വിളക്ക്", "വിളക്ക്", "ലൈറ്റ് ഇല്ല",
    ),
    "drainage": (
        "drainage", "drain blocked", "drain overflow", "clogged drain",
        "waterlogging", "water stagnation", "flooded street", "blocked drain",
        "kazhivu thilayunnu", "kazhivu block", "vellam nikkunnu",
        "thada undu", "kazhivu nikki",
        "ഓടകൾ", "ഓട", "ഡ്രെയിൻ", "വെള്ളക്കെട്ട്",
    ),
    "tree_fall": (
        "tree fallen", "fallen tree", "tree fall", "tree blocking road",
        "broken branch", "tree uprooted", "tree collapsed",
        "maram veennu", "maram thadangi", "mara veenu", "mara kuti",
        "മരം", "മരം വീണു",
    ),
    "illegal_construction": (
        "illegal construction", "unauthorised building", "encroachment",
        "illegal building", "building violation", "unauthorized structure",
        "permit illate", "anumati illaathe", "kudiyerupu",
        "അനധികൃത നിർമ്മാണം", "കൈയ്യേറ്റം",
    ),
    "electrical_hazard": (
        "electric wire", "live wire", "fallen wire", "electric shock",
        "exposed wire", "transformer fault", "high voltage", "sparking wire",
        "electric hazard", "current leakage",
        "live wire und", "kambhi veennu", "current pidikunnu",
        "shock adikkunnu", "transformer kalikkunnu", "high tension kambhi",
        "വൈദ്യുത", "കമ്പി", "ഷോക്ക്",
    ),
    "sewage_issue": (
        "sewage", "sewage overflow", "sewer", "sewer blocked", "sewage leak",
        "open sewage", "sewage smell", "sewage pipe",
        "sewage problem", "theruvi malam", "mala oozhunnu",
        "മലജലം", "സ്യൂവേജ്",
    ),
}

# ---------------------------------------------------------------------------
# Category → department routing
# Department codes are hints for the service layer — not FK lookups here.
# ---------------------------------------------------------------------------

_CATEGORY_TO_DEPT: dict[str, str] = {
    "road_damage":           "roads_and_drainage",
    "drainage":              "roads_and_drainage",
    "waste_management":      "sanitation",
    "water_supply":          "water_authority",
    "street_light":          "street_lighting",
    "tree_fall":             "parks_and_environment",
    "illegal_construction":  "building_permit_office",
    "electrical_hazard":     "electrical_engineering",
    "sewage_issue":          "sanitation",
}

# ---------------------------------------------------------------------------
# Priority signal phrases (checked in order: urgent → high → low → category)
# ---------------------------------------------------------------------------

_PRIORITY_URGENT: tuple[str, ...] = (
    "fallen wire", "fallen electric wire", "live wire",
    "live current", "kambhi veennu", "high tension wire",
    "electric shock", "electrocution", "current pidikunnu",
    "transformer blast", "transformer fire", "sparking wire",
    "tree fallen blocking", "tree blocking road", "maram thadangi road",
    "flash flood", "flooding road", "contamination",
    "sewage overflow school", "sewage overflow hospital",
)

_PRIORITY_HIGH: tuple[str, ...] = (
    "major pothole", "large pothole", "deep pothole", "road cave",
    "tree fallen", "maram veennu",
    "drain overflow", "sewage overflow",
    "water contamination", "muddy water supply", "pipe burst",
    "power failure area", "transformer fault",
)

_PRIORITY_LOW: tuple[str, ...] = (
    "street light", "street light illa", "light not working", "lamp broken",
    "minor repair", "small pothole", "minor issue",
)

# Category-level base priority used when no signal phrase fires.
_CATEGORY_BASE_PRIORITY: dict[str, str] = {
    "electrical_hazard":     "urgent",
    "tree_fall":             "high",
    "sewage_issue":          "high",
    "water_supply":          "medium",
    "drainage":              "medium",
    "road_damage":           "medium",
    "illegal_construction":  "medium",
    "waste_management":      "low",
    "street_light":          "low",
}

# ---------------------------------------------------------------------------
# Spam signals
# ---------------------------------------------------------------------------

_SPAM_PHRASES: frozenset[str] = frozenset({
    "test", "testing", "hello", "hi there", "bye", "asdf", "qwerty",
    "1234", "abcd", "lorem ipsum", "dummy", "sample", "xyz", "aaa",
    "nothing", "no issue", "just checking", "please ignore",
})

# ---------------------------------------------------------------------------
# Landmark alias dictionary
# Format : alias_lowercase → (canonical_display_name, ward_code)
#
# Sources : TVMC official ward registry (tvm_001–tvm_101, 2025 delimitation) +
#           local knowledge of Thiruvananthapuram landmarks.
#
# Covers  : all 101 ward names, common alternate spellings, Malayalam script,
#           Manglish variants, abbreviations, named hospitals/temples/junctions.
#
# DO NOT add entries without a verified real-world reference.
# ---------------------------------------------------------------------------

_LANDMARK_ALIASES: dict[str, tuple[str, str]] = {

    # ── All 101 ward names (exact + common alt spellings) ────────────────
    "kazhakkoottam":      ("Kazhakkoottam",    "tvm_001"),
    "kazhakoottam":       ("Kazhakkoottam",    "tvm_001"),
    "sainika school":     ("Sainika School",   "tvm_002"),
    "sainik school":      ("Sainika School",   "tvm_002"),
    "chanthavila":        ("Chanthavila",      "tvm_003"),
    "kattaikonam":        ("Kattaikonam",      "tvm_004"),
    "njandoorkonam":      ("Njandoorkonam",    "tvm_005"),
    "powdikonam":         ("Powdikonam",       "tvm_006"),
    "chenkottukonam":     ("Chenkottukonam",   "tvm_007"),
    "chengottukonam":     ("Chenkottukonam",   "tvm_007"),
    "chempazhanthy":      ("Chempazhanthy",    "tvm_008"),
    "kariavattom":        ("Kariavattom",      "tvm_009"),
    "karyavattom":        ("Kariavattom",      "tvm_009"),
    "pangappara":         ("Pangappara",       "tvm_010"),
    "sreekariyam":        ("Sreekariyam",      "tvm_011"),
    "sreekaryam":         ("Sreekariyam",      "tvm_011"),
    "chellamangalam":     ("Chellamangalam",   "tvm_012"),
    "mannanthala":        ("Mannanthala",      "tvm_013"),
    "pathirapalli":       ("Pathirapalli",     "tvm_014"),
    "ambalamukku":        ("Ambalamukku",      "tvm_015"),
    "kudappanakunnu":     ("Kudappanakunnu",   "tvm_016"),
    "thuruthummoola":     ("Thuruthummoola",   "tvm_017"),
    "nettayam":           ("Nettayam",         "tvm_018"),
    "kachani":            ("Kachani",          "tvm_019"),
    "vazhottukonam":      ("Vazhottukonam",    "tvm_020"),
    "kodunganoor":        ("Kodunganoor",      "tvm_021"),
    "vattiyoorkavu":      ("Vattiyoorkavu",    "tvm_022"),
    "vattiyorkavu":       ("Vattiyoorkavu",    "tvm_022"),
    "kanjirampara":       ("Kanjirampara",     "tvm_023"),
    "peroorkada":         ("Peroorkada",       "tvm_024"),
    "kowdiar":            ("Kowdiar",          "tvm_025"),
    "kuravankonam":       ("Kuravankonam",     "tvm_026"),
    "muttada":            ("Muttada",          "tvm_027"),
    "chettivilakam":      ("Chettivilakam",    "tvm_028"),
    "kinavoor":           ("Kinavoor",         "tvm_029"),
    "nalanchira":         ("Nalanchira",       "tvm_030"),
    "edavakode":          ("Edavakode",        "tvm_031"),
    "ulloor":             ("Ulloor",           "tvm_032"),
    "medical college":    ("Medical College Hospital", "tvm_033"),
    "pattom":             ("Pattom",           "tvm_034"),
    "kesavadasapuram":    ("Kesavadasapuram",  "tvm_035"),
    "kdp":                ("Kesavadasapuram",  "tvm_035"),
    "gowreeshapattom":    ("Gowreeshapattom",  "tvm_036"),
    "kunnukuzhy":         ("Kunnukuzhy",       "tvm_037"),
    "nanthancode":        ("Nanthancode",      "tvm_038"),
    "palayam":            ("Palayam",          "tvm_039"),
    "vazhuthacaud":       ("Vazhuthacaud",     "tvm_040"),
    "vazhuthacad":        ("Vazhuthacaud",     "tvm_040"),
    "sasthamangalam":     ("Sasthamangalam",   "tvm_041"),
    "pangode":            ("Pangode",          "tvm_042"),
    "thirumala":          ("Thirumala",        "tvm_043"),
    "valiyavila":         ("Valiyavila",       "tvm_044"),
    "thrikkannapuram":    ("Thrikkannapuram",  "tvm_045"),
    "punnakkamugal":      ("Punnakkamugal",    "tvm_046"),
    "poojappura":         ("Poojappura",       "tvm_047"),
    "jagathy":            ("Jagathy",          "tvm_048"),
    "thycaud":            ("Thycaud",          "tvm_049"),
    "valiyasala":         ("Valiyasala",       "tvm_050"),
    "arannoor":           ("Arannoor",         "tvm_051"),
    "mudavanmugal":       ("Mudavanmugal",     "tvm_052"),
    "nemom":              ("Nemom",            "tvm_054"),
    "ponnumangalam":      ("Ponnumangalam",    "tvm_055"),
    "melamcode":          ("Melamcode",        "tvm_056"),
    "pappanamcode":       ("Pappanamcode",     "tvm_057"),
    "karamana":           ("Karamana",         "tvm_058"),
    "nedumcaud":          ("Nedumcaud",        "tvm_059"),
    "nedumcad":           ("Nedumcaud",        "tvm_059"),
    "kaladi":             ("Kaladi",           "tvm_060"),
    "karumom":            ("Karumom",          "tvm_061"),
    "punchakkari":        ("Punchakkari",      "tvm_062"),
    "poonkulam":          ("Poonkulam",        "tvm_063"),
    "venganoor":          ("Venganoor",        "tvm_064"),
    "vizhinjam":          ("Vizhinjam",        "tvm_066"),
    "vellar":             ("Vellar",           "tvm_068"),
    "thiruvallam":        ("Thiruvallam",      "tvm_069"),
    "poonthura":          ("Poonthura",        "tvm_070"),
    "puthenppalli":       ("Puthenppalli",     "tvm_071"),
    "ambalathara":        ("Ambalathara",      "tvm_072"),
    "attukal":            ("Attukal",          "tvm_073"),
    "kalippankulam":      ("Kalippankulam",    "tvm_074"),
    "kamaleswaram":       ("Kamaleswaram",     "tvm_075"),
    "beemapalli":         ("Beemapalli",       "tvm_076"),
    "valiyathura":        ("Valiyathura",      "tvm_077"),
    "vallakkadavu":       ("Vallakkadavu",     "tvm_078"),
    "sreevaraham":        ("Sreevaraham",      "tvm_079"),
    "manacaud":           ("Manacaud",         "tvm_080"),
    "manakad":            ("Manacaud",         "tvm_080"),
    "chalai":             ("Chalai",           "tvm_081"),
    "perunthanni":        ("Perunthanni",      "tvm_083"),
    "sreekanteswaram":    ("Sreekanteswaram",  "tvm_084"),
    "thampanoor":         ("Thampanoor",       "tvm_085"),
    "vanchiyoor":         ("Vanchiyoor",       "tvm_086"),
    "kannammoola":        ("Kannammoola",      "tvm_087"),
    "pettah":             ("Pettah",           "tvm_088"),
    "chackai":            ("Chackai",          "tvm_089"),
    "vettukadu":          ("Vettukadu",        "tvm_090"),
    "karikkakam":         ("Karikkakam",       "tvm_091"),
    "kadakampally":       ("Kadakampally",     "tvm_092"),
    "anamugham":          ("Anamugham",        "tvm_093"),
    "akkulam":            ("Akkulam",          "tvm_094"),
    "cheruvaikkal":       ("Cheruvaikkal",     "tvm_095"),
    "alathara":           ("Alathara",         "tvm_096"),
    "kuzhivila":          ("Kuzhivila",        "tvm_097"),
    "poundkadavu":        ("Poundkadavu",      "tvm_098"),
    "kulathoor":          ("Kulathoor",        "tvm_099"),
    "attipra":            ("Attipra",          "tvm_100"),
    "pallithura":         ("Pallithura",       "tvm_101"),

    # ── Transport hubs ───────────────────────────────────────────────────
    "central station":          ("Thiruvananthapuram Central",           "tvm_085"),
    "trivandrum central":       ("Thiruvananthapuram Central",           "tvm_085"),
    "tvc station":              ("Thiruvananthapuram Central",           "tvm_085"),
    "railway station":          ("Thiruvananthapuram Central",           "tvm_085"),
    "ksrtc bus stand":          ("KSRTC Bus Station Thampanoor",         "tvm_085"),
    "ksrtc buststand":          ("KSRTC Bus Station Thampanoor",         "tvm_085"),
    "ksrtc":                    ("KSRTC Bus Station Thampanoor",         "tvm_085"),
    "bus stand thampanoor":     ("KSRTC Bus Station Thampanoor",         "tvm_085"),
    "airport":                  ("Thiruvananthapuram International Airport", "tvm_089"),
    "tia":                      ("Thiruvananthapuram International Airport", "tvm_089"),
    "trivandrum airport":       ("Thiruvananthapuram International Airport", "tvm_089"),
    "international airport":    ("Thiruvananthapuram International Airport", "tvm_089"),

    # ── Medical / Healthcare ─────────────────────────────────────────────
    "mch":                      ("Medical College Hospital",    "tvm_033"),
    "govt medical college":     ("Medical College Hospital",    "tvm_033"),
    "government medical college": ("Medical College Hospital",  "tvm_033"),
    "gmc trivandrum":           ("Medical College Hospital",    "tvm_033"),
    "medical college hospital": ("Medical College Hospital",    "tvm_033"),
    "sat hospital":             ("SAT Hospital",                "tvm_033"),
    "general hospital":         ("General Hospital Palayam",    "tvm_039"),
    "govt hospital palayam":    ("General Hospital Palayam",    "tvm_039"),
    "ims hospital":             ("IMS Hospital Ulloor",         "tvm_032"),
    "sut hospital":             ("SUT Hospital Pattom",         "tvm_034"),
    "prs hospital":             ("PRS Hospital",                "tvm_035"),

    # ── Educational ──────────────────────────────────────────────────────
    "kerala university":        ("Kerala University",                    "tvm_009"),
    "university of kerala":     ("Kerala University",                    "tvm_009"),
    "kariavattom campus":       ("Kerala University Kariavattom Campus", "tvm_009"),
    "university college":       ("University College Trivandrum",        "tvm_039"),
    "cet":                      ("College of Engineering Trivandrum",    "tvm_039"),
    "college of engineering":   ("College of Engineering Trivandrum",    "tvm_039"),
    "engineering college tvm":  ("College of Engineering Trivandrum",    "tvm_039"),
    "womens college":           ("Government Women's College",           "tvm_040"),

    # ── Government offices ───────────────────────────────────────────────
    "secretariat":              ("Kerala Secretariat",           "tvm_039"),
    "kerala secretariat":       ("Kerala Secretariat",           "tvm_039"),
    "assembly":                 ("Kerala Legislative Assembly",  "tvm_039"),
    "legislature":              ("Kerala Legislative Assembly",  "tvm_039"),
    "high court":               ("Kerala High Court",            "tvm_040"),
    "raj bhavan":               ("Raj Bhavan",                   "tvm_025"),
    "governor house":           ("Raj Bhavan",                   "tvm_025"),
    "corporation office":       ("TVMC Corporation Office",      "tvm_085"),
    "collectorate":             ("District Collectorate",        "tvm_039"),
    "tvmc office":              ("TVMC Corporation Office",      "tvm_085"),

    # ── Temples / Religious ──────────────────────────────────────────────
    "padmanabhaswamy temple":   ("Padmanabhaswamy Temple",       "tvm_082"),
    "sree padmanabhaswamy":     ("Padmanabhaswamy Temple",       "tvm_082"),
    "padmanabha temple":        ("Padmanabhaswamy Temple",       "tvm_082"),
    "east fort":                ("East Fort",                    "tvm_082"),
    "west fort":                ("Fort Area",                    "tvm_082"),
    "fort area":                ("Fort Area",                    "tvm_082"),
    "attukal temple":           ("Attukal Bhagavathy Temple",    "tvm_073"),
    "attukal bhagavathy":       ("Attukal Bhagavathy Temple",    "tvm_073"),
    "beemapalli mosque":        ("Beemapalli Mosque",            "tvm_076"),
    "beema palli":              ("Beemapalli Mosque",            "tvm_076"),
    "thycaud church":           ("Thycaud Church",               "tvm_049"),
    "sreekanteswaram temple":   ("Sreekanteswaram Temple",       "tvm_084"),

    # ── IT / Business parks ──────────────────────────────────────────────
    "technopark":               ("Technopark",                   "tvm_001"),
    "techno park":              ("Technopark",                   "tvm_001"),
    "technopark phase 1":       ("Technopark Phase 1",           "tvm_001"),
    "technopark phase 3":       ("Technopark Phase 3",           "tvm_001"),
    "infosys technopark":       ("Infosys Technopark",           "tvm_001"),
    "tcs technopark":           ("TCS Technopark",               "tvm_001"),

    # ── Beaches / Recreation ─────────────────────────────────────────────
    "vizhinjam harbour":        ("Vizhinjam Harbour",            "tvm_067"),
    "vizhinjam port":           ("Vizhinjam Port",               "tvm_065"),
    "shanghumugham":            ("Shanghumugham Beach",          "tvm_093"),
    "shanmugham beach":         ("Shanghumugham Beach",          "tvm_093"),
    "thiruvallam bridge":       ("Thiruvallam Bridge",           "tvm_069"),
    "kovalam":                  ("Kovalam",                      "tvm_064"),
    "kovalam beach":            ("Kovalam Beach",                "tvm_064"),
    "akkulam lake":             ("Akkulam Lake",                 "tvm_094"),
    "veli lake":                ("Veli Lake",                    "tvm_094"),
    "veli tourist village":     ("Veli Tourist Village",         "tvm_094"),
    "poovar":                   ("Poovar",                       "tvm_101"),
    "napier museum":            ("Napier Museum",                "tvm_039"),
    "zoo":                      ("Trivandrum Zoo",               "tvm_039"),
    "trivandrum zoo":           ("Trivandrum Zoo",               "tvm_039"),
    "kanakakunnu":              ("Kanakakunnu Palace",           "tvm_039"),

    # ── Junctions / named roads / landmarks ─────────────────────────────
    "statue junction":          ("Statue Junction",              "tvm_039"),
    "palayam market":           ("Palayam Market",               "tvm_039"),
    "connemara market":         ("Connemara Market",             "tvm_039"),
    "mg road":                  ("MG Road",                      "tvm_039"),
    "chalai market":            ("Chalai Market",                "tvm_081"),
    "chalai bazaar":            ("Chalai Market",                "tvm_081"),
    "karamana bridge":          ("Karamana Bridge",              "tvm_058"),
    "jagathy bridge":           ("Jagathy Bridge",               "tvm_048"),
    "poojappura jail":          ("Central Prison Poojappura",    "tvm_047"),
    "central prison":           ("Central Prison Poojappura",    "tvm_047"),
    "central jail":             ("Central Prison Poojappura",    "tvm_047"),
    "pangode military":         ("Pangode Military Camp",        "tvm_042"),
    "tvm overbridge":           ("Thampanoor Overbridge",        "tvm_085"),
    "nh 66":                    ("NH 66",                        "tvm_001"),
    "nh66":                     ("NH 66",                        "tvm_001"),
    "vellayambalam":            ("Vellayambalam",                "tvm_025"),
    "bypass road":              ("Thiruvananthapuram Bypass",    "tvm_001"),

    # ── City-level aliases ───────────────────────────────────────────────
    "trivandrum":               ("Thiruvananthapuram",           "tvm_085"),
    "tvm":                      ("Thiruvananthapuram",           "tvm_085"),
    "thiruvananthapuram":       ("Thiruvananthapuram",           "tvm_085"),

    # ── Malayalam script aliases ─────────────────────────────────────────
    "മെഡിക്കൽ കോളേജ്":        ("Medical College Hospital",    "tvm_033"),
    "ഗവ. മെഡിക്കൽ കോളേജ്":   ("Medical College Hospital",    "tvm_033"),
    "പദ്മനാഭ ക്ഷേത്രം":       ("Padmanabhaswamy Temple",      "tvm_082"),
    "ശ്രീ പദ്മനാഭസ്വാമി":     ("Padmanabhaswamy Temple",      "tvm_082"),
    "അട്ടുകൽ ക്ഷേത്രം":      ("Attukal Bhagavathy Temple",   "tvm_073"),
    "ആറ്റുകൽ ക്ഷേത്രം":      ("Attukal Bhagavathy Temple",   "tvm_073"),
    "ചാലൈ":                    ("Chalai",                      "tvm_081"),
    "ഫോർട്ട്":                 ("Fort",                        "tvm_082"),
    "കഴക്കൂട്ടം":             ("Kazhakkoottam",               "tvm_001"),
    "ടെക്നോ പാർക്ക്":         ("Technopark",                  "tvm_001"),
    "ടെക്നോപാർക്ക്":          ("Technopark",                  "tvm_001"),
    "ശ്രീകാര്യം":              ("Sreekariyam",                 "tvm_011"),
    "കരമന":                    ("Karamana",                    "tvm_058"),
    "കൗഡിയാർ":                 ("Kowdiar",                     "tvm_025"),
    "പട്ടം":                   ("Pattom",                      "tvm_034"),
    "കേരള സെക്രട്ടേറിയേറ്റ്": ("Kerala Secretariat",          "tvm_039"),
    "ഹൈക്കോടതി":               ("Kerala High Court",           "tvm_040"),
    "തമ്പാനൂർ":                ("Thampanoor",                  "tvm_085"),
    "തമ്പനൂർ":                 ("Thampanoor",                  "tvm_085"),
    "പേട്ട":                   ("Pettah",                      "tvm_088"),
    "വഞ്ചിയൂർ":                ("Vanchiyoor",                  "tvm_086"),
    "മാനക്കാട്":               ("Manacaud",                    "tvm_080"),
    "ജഗതി":                    ("Jagathy",                     "tvm_048"),
    "വെള്ളയമ്പലം":             ("Vellayambalam",               "tvm_025"),
    "ഉള്ളൂർ":                  ("Ulloor",                      "tvm_032"),
    "നളഞ്ചിര":                 ("Nalanchira",                  "tvm_030"),
    "നന്തൻകോട്":               ("Nanthancode",                 "tvm_038"),
    "തൈക്കാട്":                ("Thycaud",                     "tvm_049"),
    "കരമന പാലം":               ("Karamana Bridge",             "tvm_058"),
    "വിഴിഞ്ഞം":                ("Vizhinjam",                   "tvm_066"),
    "ആറ്റുകൽ":                 ("Attukal",                     "tvm_073"),
    "ബീമാപള്ളി":               ("Beemapalli",                  "tvm_076"),
    "തിരുവല്ലം":               ("Thiruvallam",                 "tvm_069"),
    "ശംഖുമുഖം":                ("Shanghumugham Beach",          "tvm_093"),
    "മ്യൂസിയം":                ("Napier Museum",               "tvm_039"),
    "കേന്ദ്ര ജയിൽ":            ("Central Prison",              "tvm_047"),
    "തിരുവനന്തപുരം":           ("Thiruvananthapuram",          "tvm_085"),
    "ശ്രീകണ്ഠേശ്വരം":          ("Sreekanteswaram",             "tvm_084"),
    "കോടതി":                   ("Kerala High Court",           "tvm_040"),
    "ബൈപ്പാസ്":                ("Thiruvananthapuram Bypass",   "tvm_001"),
    "ബ്ലൂ ഫ്ലാഗ് ബീച്ച്":     ("Shanghumugham Beach",          "tvm_093"),
    "ഗ്രീൻ ഫീൽഡ്":             ("Greenfield Stadium",          "tvm_001"),
    "ഗ്രീൻഫീൽഡ് സ്റ്റേഡിയം":  ("Greenfield Stadium",          "tvm_001"),
    "ഗ്രൗണ്ട്":                ("Greenfield Stadium",          "tvm_001"),
}

# ---------------------------------------------------------------------------
# Tokenisation helper (shared by classify_issue, detect_possible_duplicate)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-zA-Zഀ-ൿ]+")


def _tokenize(text: str) -> frozenset[str]:
    """Return lower-cased word tokens including Malayalam Unicode."""
    return frozenset(_TOKEN_RE.findall(text.lower()))


# ===========================================================================
# Public pure functions
# ===========================================================================


def detect_language(text: str) -> dict[str, object]:
    """Identify the script and language of the input text.

    Returns::

        {
            "language": "malayalam" | "english" | "manglish" | "mixed" | "unknown",
            "script":   "malayalam" | "latin" | "mixed" | "unknown",
            "confidence": float,   # 0.0–1.0
        }
    """
    stripped = text.strip()
    if not stripped:
        return {"language": "unknown", "script": "unknown", "confidence": 0.0}

    non_space = [c for c in stripped if not c.isspace()]
    total = len(non_space)
    if total == 0:
        return {"language": "unknown", "script": "unknown", "confidence": 0.0}

    ml_count = len(_MALAYALAM_RE.findall(stripped))
    latin_count = len(_LATIN_RE.findall(stripped))
    ml_ratio = ml_count / total
    latin_ratio = latin_count / total

    # Dominant Malayalam script
    if ml_ratio > 0.5:
        return {
            "language": "malayalam",
            "script": "malayalam",
            "confidence": round(min(ml_ratio + 0.15, 1.0), 3),
        }

    # Mixed scripts present
    if ml_ratio > 0.08 and latin_ratio > 0.1:
        return {"language": "mixed", "script": "mixed", "confidence": 0.75}

    # Latin — distinguish English from Manglish
    if latin_ratio > 0.4:
        words = frozenset(_TOKEN_RE.findall(stripped.lower()))
        manglish_hits = words & _MANGLISH_SIGNALS
        if manglish_hits:
            conf = round(min(0.55 + len(manglish_hits) * 0.08, 0.95), 3)
            return {"language": "manglish", "script": "latin", "confidence": conf}
        return {
            "language": "english",
            "script": "latin",
            "confidence": round(min(latin_ratio + 0.1, 1.0), 3),
        }

    return {"language": "unknown", "script": "unknown", "confidence": 0.0}


def normalize_text(text: str) -> str:
    """Return a clean, NFKC-normalised single-line string truncated to 240 chars."""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized[:240]


def classify_issue(text: str) -> dict[str, object]:
    """Return the best-matching civic category for the complaint text.

    Scoring: count keyword hits per category; pick the one with the most hits.

    Returns::

        {
            "category_code": str,   # e.g. "road_damage" or "" if no match
            "confidence": float,    # 0.0–1.0
        }
    """
    lowered = text.lower()
    best_category = ""
    best_hits = 0

    for category, keywords in _ISSUE_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in lowered)
        if hits > best_hits:
            best_hits = hits
            best_category = category

    if best_hits == 0:
        return {"category_code": "", "confidence": 0.0}

    # 1 hit → 0.50,  2 → 0.65,  3 → 0.78, 4+ → capped at 0.92
    confidence = round(min(0.40 + best_hits * 0.13, 0.92), 3)
    return {"category_code": best_category, "confidence": confidence}


def detect_department(category_code: str) -> str:
    """Return a department-code hint for the given category code.

    Returns an empty string if the category is unknown.  The service layer
    resolves this hint to a Department FK — this function only produces the
    string code.
    """
    return _CATEGORY_TO_DEPT.get(category_code, "")


def extract_landmarks(text: str) -> dict[str, object]:
    """Search the text for known TVMC landmarks and return ward hints.

    Matching strategy:
    - Multi-word aliases: substring match (case-insensitive).
    - Single-word aliases: word-boundary match to reduce false positives
      (e.g. "port" should not fire inside "transport").

    Results are deduplicated by ward_code — the longest alias match wins
    for each ward.

    Returns::

        {
            "landmarks": list[dict],  # [{name, ward_code, alias_matched}, ...]
            "ward_hint": str | None,  # ward code of the first/best match
            "confidence": float,      # 0.0–1.0
        }
    """
    lowered = text.lower()

    # Longest alias first so more-specific matches shadow short ones.
    sorted_aliases = sorted(_LANDMARK_ALIASES.items(), key=lambda kv: -len(kv[0]))

    raw_matches: list[dict] = []
    for alias, (canonical, ward_code) in sorted_aliases:
        if " " in alias:
            matched = alias in lowered
        else:
            # Word-boundary check using negative look-around for word chars.
            matched = bool(
                re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", lowered)
            )
        if matched:
            raw_matches.append(
                {"name": canonical, "ward_code": ward_code, "alias_matched": alias}
            )

    # Deduplicate by ward_code, keeping the longest alias match per ward.
    best_by_ward: dict[str, dict] = {}
    for m in raw_matches:
        wc = m["ward_code"]
        if wc not in best_by_ward or len(m["alias_matched"]) > len(
            best_by_ward[wc]["alias_matched"]
        ):
            best_by_ward[wc] = m

    landmarks = list(best_by_ward.values())
    ward_hint = landmarks[0]["ward_code"] if landmarks else None
    confidence = round(min(0.5 + len(landmarks) * 0.15, 0.95), 3) if landmarks else 0.0

    return {"landmarks": landmarks, "ward_hint": ward_hint, "confidence": confidence}


def predict_priority(text: str, category_code: str = "") -> str:
    """Infer complaint severity from signal phrases and category.

    Priority ladder (checked in order):
      urgent   → life-safety signal phrases (live wire, flood, etc.)
      high     → major infrastructure failure
      category → per-category default (electrical_hazard → urgent, etc.)
      low      → minor / cosmetic signal phrases
      medium   → fallback

    Returns one of: ``"low"`` | ``"medium"`` | ``"high"`` | ``"urgent"`` | ``"critical"``
    """
    lowered = text.lower()

    for phrase in _PRIORITY_URGENT:
        if phrase in lowered:
            return "urgent"

    for phrase in _PRIORITY_HIGH:
        if phrase in lowered:
            return "high"

    if category_code:
        base = _CATEGORY_BASE_PRIORITY.get(category_code)
        if base in ("urgent", "high"):
            return base

    for phrase in _PRIORITY_LOW:
        if phrase in lowered:
            return "low"

    if category_code:
        return _CATEGORY_BASE_PRIORITY.get(category_code, "medium")

    return "medium"


def detect_spam(text: str) -> dict[str, object]:
    """Detect gibberish, repetition, and meaningless submissions.

    Returns::

        {
            "is_spam": bool,
            "spam_score": float,   # 0.0–1.0
            "spam_reason": str,    # human-readable reason or ""
        }
    """
    stripped = text.strip()

    if not stripped:
        return {"is_spam": True, "spam_score": 1.0, "spam_reason": "empty submission"}

    if len(stripped) < 8:
        return {"is_spam": True, "spam_score": 0.90, "spam_reason": "too short to be a valid complaint"}

    lowered = stripped.lower()

    if lowered in _SPAM_PHRASES:
        return {"is_spam": True, "spam_score": 0.95, "spam_reason": "known test or spam phrase"}

    # High word repetition (e.g. "road road road road")
    words = lowered.split()
    if len(words) >= 3:
        most_common_count = Counter(words).most_common(1)[0][1]
        repetition_ratio = most_common_count / len(words)
        if repetition_ratio >= 0.65:
            return {
                "is_spam": True,
                "spam_score": round(0.6 + repetition_ratio * 0.3, 3),
                "spam_reason": "excessive word repetition",
            }

    # Mostly non-alphabetic characters
    alpha = sum(1 for c in stripped if c.isalpha())
    alpha_ratio = alpha / len(stripped)
    if alpha_ratio < 0.35:
        return {
            "is_spam": True,
            "spam_score": round(0.5 + (0.35 - alpha_ratio) * 2, 3),
            "spam_reason": "mostly non-alphabetic content",
        }

    return {"is_spam": False, "spam_score": 0.0, "spam_reason": ""}


def detect_possible_duplicate(
    text: str,
    recent_texts: Sequence[str] = (),
) -> dict[str, object]:
    """Detect whether the complaint is a near-duplicate of a recent submission.

    Uses Jaccard similarity over Unicode word tokens.  Threshold: 0.55.

    Returns::

        {
            "is_duplicate": bool,
            "similarity_score": float,   # 0.0–1.0 (highest match)
            "matching_text": str | None, # the most similar recent text, or None
        }
    """
    if not recent_texts or not text.strip():
        return {"is_duplicate": False, "similarity_score": 0.0, "matching_text": None}

    tokens = _tokenize(text)
    if not tokens:
        return {"is_duplicate": False, "similarity_score": 0.0, "matching_text": None}

    best_score = 0.0
    best_text: str | None = None

    for recent in recent_texts:
        recent_tokens = _tokenize(recent)
        if not recent_tokens:
            continue
        union_size = len(tokens | recent_tokens)
        if union_size == 0:
            continue
        score = len(tokens & recent_tokens) / union_size
        if score > best_score:
            best_score = score
            best_text = recent

    _DUPLICATE_THRESHOLD = 0.55
    is_dup = best_score >= _DUPLICATE_THRESHOLD
    return {
        "is_duplicate": is_dup,
        "similarity_score": round(best_score, 3),
        "matching_text": best_text if is_dup else None,
    }


def score_analysis(
    *,
    language_result: dict,
    category_result: dict,
    landmark_result: dict,
    spam_result: dict,
    duplicate_result: dict,
) -> float:
    """Combine individual signal scores into an overall confidence value.

    Weights:
      - Category match : 0.45  (primary signal — is the issue identifiable?)
      - Landmark match : 0.25  (is there a locatable address?)
      - Language detect: 0.15  (is the text interpretable?)
      - Base floor     : 0.15  (any non-empty, non-spam submission gets credit)

    Penalties:
      - Spam score reduces overall confidence by up to 80 %.
      - Confirmed duplicate applies a 30 % soft penalty.

    Returns a float in [0.0, 1.0].
    """
    base = 0.15
    cat_score = float(category_result.get("confidence", 0.0)) * 0.45
    lm_score = float(landmark_result.get("confidence", 0.0)) * 0.25
    lang_score = float(language_result.get("confidence", 0.0)) * 0.15

    raw = base + cat_score + lm_score + lang_score

    # Spam penalty: spam_score=1.0 → multiply by 0.2 (80 % reduction)
    spam_score = float(spam_result.get("spam_score", 0.0))
    raw = raw * (1.0 - spam_score * 0.80)

    # Duplicate soft penalty
    if duplicate_result.get("is_duplicate"):
        raw *= 0.70

    return round(max(0.0, min(1.0, raw)), 3)


# ===========================================================================
# ML-backed inference helpers (primary prediction path)
# ===========================================================================

def _try_ml_category(text: str) -> dict[str, object] | None:
    """Run ML category prediction.  Returns None if ML is unavailable.

    Populates ``ml_tier`` in the result dict so ``_fuse_category`` can set
    ``inference_source`` to ``"transformer"`` | ``"tfidf"`` correctly.
    """
    try:
        from apps.ml import ml_inference  # noqa: PLC0415
        result = ml_inference.predict_category(text)
        # Read which tier answered AFTER the call
        tier = ml_inference.active_tier()
        # Treat spam / no_category predictions from ML as empty category
        if result.label in {"spam", "no_category"}:
            return {
                "category_code": "",
                "confidence":    0.0,
                "ml_label":      result.label,
                "ml_confidence": result.confidence,
                "ml_tier":       tier,
            }
        return {
            "category_code": result.label,
            "confidence":    result.confidence,
            "ml_label":      result.label,
            "ml_confidence": result.confidence,
            "ml_tier":       tier,
        }
    except Exception as exc:  # noqa: BLE001
        logger.debug("ML category unavailable: %s", exc)
        return None


def _try_ml_priority(text: str, category_code: str = "") -> str | None:
    """Run ML priority prediction.  Returns None if ML is unavailable."""
    try:
        from apps.ml.ml_inference import ModelUnavailable, predict_priority as ml_prio  # noqa: PLC0415
        result = ml_prio(text)
        if result.confidence >= _ML_BLEND_THRESHOLD:
            return result.label
    except (Exception,):  # noqa: BLE001
        pass
    return None


def _try_ml_spam(text: str) -> dict[str, object] | None:
    """Run ML spam detection.  Returns None if ML is unavailable."""
    try:
        from apps.ml.ml_inference import ModelUnavailable, predict_spam as ml_spam  # noqa: PLC0415
        result = ml_spam(text)
        return {
            "is_spam":    result.is_spam,
            "spam_score": result.spam_score,
            "spam_reason": "ml_spam_detector" if result.is_spam else "",
        }
    except (Exception,):  # noqa: BLE001
        return None


def _try_ml_language(text: str) -> dict[str, object] | None:
    """Run ML language detection.  Returns None if ML is unavailable."""
    try:
        from apps.ml.ml_inference import ModelUnavailable, predict_language as ml_lang  # noqa: PLC0415
        result = ml_lang(text)
        if result.confidence >= _ML_BLEND_THRESHOLD:
            # Map ML labels to the rule-engine vocabulary
            lang_map = {"en": "english", "ml": "malayalam", "manglish": "manglish", "mixed": "mixed"}
            return {
                "language":   lang_map.get(result.language, result.language),
                "script":     "malayalam" if result.language == "ml" else "latin",
                "confidence": result.confidence,
            }
    except (Exception,):  # noqa: BLE001
        pass
    return None


def _fuse_category(
    rule_result: dict[str, object],
    ml_result: dict[str, object] | None,
) -> dict[str, object]:
    """Blend rule-engine and ML category predictions.

    Fusion strategy
    ---------------
    ML confidence >= _ML_PRIMARY_THRESHOLD (0.55)
        ML wins outright.  ``source`` = ml_tier ("transformer" | "tfidf").
    ML confidence in [_ML_BLEND_THRESHOLD, 0.55)
        Blend: if both agree → average confidence, source = "<tier>_fusion";
               if disagree → higher-confidence source wins × 0.85 discount.
    ML confidence < _ML_BLEND_THRESHOLD
        Rule engine result returned as-is (source not set → "rule").
    ML unavailable
        Rule engine result returned as-is.
    """
    if ml_result is None:
        return rule_result

    ml_conf   = float(ml_result.get("ml_confidence", 0.0))
    rule_conf = float(rule_result.get("confidence", 0.0))
    ml_cat    = str(ml_result.get("ml_label", ""))
    rule_cat  = str(rule_result.get("category_code", ""))
    tier      = str(ml_result.get("ml_tier", "ml"))  # "transformer" | "tfidf"

    if ml_conf >= _ML_PRIMARY_THRESHOLD:
        cat = str(ml_result.get("category_code", ""))
        return {
            "category_code": cat,
            "confidence":    round(ml_conf, 3) if cat else 0.0,
            "source":        tier,           # "transformer" or "tfidf"
        }

    if ml_conf >= _ML_BLEND_THRESHOLD:
        if ml_cat == rule_cat and ml_cat:
            blended = round((ml_conf + rule_conf) / 2, 3)
            return {
                "category_code": ml_cat,
                "confidence":    blended,
                "source":        f"{tier}_fusion",
            }
        if ml_conf >= rule_conf:
            return {
                "category_code": str(ml_result["category_code"]),
                "confidence":    round(ml_conf * 0.85, 3),
                "source":        f"{tier}_over_rule",
            }
        return {
            "category_code": rule_cat,
            "confidence":    round(rule_conf * 0.85, 3),
            "source":        f"rule_over_{tier}",
        }

    # ML confidence too low — trust rule engine
    return rule_result


def _try_transformer_location(text: str) -> list[dict] | None:
    """Use transformer-based ward candidate ranking.

    Returns a list of {name, ward_code, score} dicts (top-5), or None if
    transformer is not available or similarity is too low.
    """
    try:
        from apps.ml.transformer_inference import get_transformer_engine  # noqa: PLC0415
        engine = get_transformer_engine()
        if not engine.is_ready:
            return None
        loc_result = engine.find_ward_candidates(text, top_k=5)
        if loc_result.top_score < 0.40:
            return None
        candidates = []
        for name, score in loc_result.candidates:
            if score >= 0.35:
                ward_code = _LANDMARK_ALIASES.get(name.lower(), (name, "unknown"))[1]
                candidates.append({
                    "name":          name,
                    "ward_code":     ward_code,
                    "alias_matched": name,
                    "score":         round(float(score), 3),
                })
        return candidates if candidates else None
    except Exception as exc:  # noqa: BLE001
        logger.debug("Transformer location failed: %s", exc)
        return None


def _try_ml_duplicate(
    text: str,
    recent_texts: Sequence[str],
) -> dict[str, object] | None:
    """Semantic / TF-IDF cosine duplicate detection via ml_inference tier chain."""
    if not recent_texts or not text.strip():
        return None
    try:
        from apps.ml import ml_inference  # noqa: PLC0415
        best_score = 0.0
        best_text: str | None = None
        for recent in recent_texts:
            sim = ml_inference.compute_duplicate_similarity(text, recent)
            if sim > best_score:
                best_score = sim
                best_text = recent
        tier = ml_inference.active_tier()
        _DUPLICATE_THRESHOLD = 0.55
        is_dup = best_score >= _DUPLICATE_THRESHOLD
        return {
            "is_duplicate":     is_dup,
            "similarity_score": round(best_score, 3),
            "matching_text":    best_text if is_dup else None,
            "method":           tier,   # "transformer" | "tfidf"
        }
    except Exception as exc:  # noqa: BLE001
        logger.debug("ML duplicate check failed: %s", exc)
        return None


# ===========================================================================
# Main orchestrator (ML primary + rule fallback + fusion)
# ===========================================================================


def analyze_complaint(
    text: str,
    *,
    recent_texts: Sequence[str] | None = None,
    language_hint: str | None = None,
    image_input: object = None,
) -> dict[str, object]:
    """Orchestrate all analysis steps and return the complete intelligence payload.

    Pipeline
    --------
    PRIMARY:  ML-backed inference (TF-IDF + LogisticRegression, trained models).
    FALLBACK: Rule-based inference (keyword matching, heuristics).
    FUSION:   When both ML and rule engine produce results, confidence-weighted
              blending is applied (see _fuse_category).

    Parameters
    ----------
    text
        Raw complaint text — any language / script combination.
    recent_texts
        Optional list of recent complaint texts for duplicate detection.
    language_hint
        Optional caller-supplied language hint (overrides auto-detection for
        the ``language`` key; detection still runs internally for confidence).
    image_input
        Optional image evidence (str path, bytes, or PIL Image).  When
        provided, Phase B image intelligence is applied:  confidence is
        adjusted and additional human-review flags may be raised.

    Returns
    -------
    dict with keys:
        language, language_confidence, normalized_text,
        category_code, category_confidence, department_code,
        landmarks, ward_hint, landmark_confidence,
        priority, spam, duplicate,
        needs_human_review, review_reasons, confidence,
        image_analysis  (None when no image provided),
        decision,
        inference_source  ("transformer" | "tfidf" | "rule" | "<tier>_fusion" — category source)
    """
    # ── Phase 1: language (ML primary, rule fallback) ────────────────────
    ml_lang = _try_ml_language(text)
    if ml_lang is not None:
        lang_result = ml_lang
    else:
        lang_result = detect_language(text)
    effective_language = language_hint or str(lang_result["language"])

    # ── Phase 2: normalise ───────────────────────────────────────────────
    normalized = normalize_text(text)

    # ── Phase 3: spam (ML primary, rule fallback) ────────────────────────
    ml_spam_res = _try_ml_spam(text)
    rule_spam   = detect_spam(text)
    if ml_spam_res is not None:
        # Use ML spam score but keep rule reason when rule fires strongly
        if float(ml_spam_res["spam_score"]) >= _ML_BLEND_THRESHOLD:
            spam_result = ml_spam_res
        else:
            # Blend: take the higher spam score (conservative)
            combined_score = max(
                float(ml_spam_res["spam_score"]),
                float(rule_spam.get("spam_score", 0.0)),
            )
            spam_result = {
                "is_spam":    combined_score >= 0.50,
                "spam_score": round(combined_score, 3),
                "spam_reason": rule_spam.get("spam_reason", ""),
            }
    else:
        spam_result = rule_spam

    # ── Phase 4: category (ML primary, rule fallback, fusion) ────────────
    ml_cat_res   = _try_ml_category(text)
    rule_cat_res = classify_issue(text)
    fused_cat    = _fuse_category(rule_cat_res, ml_cat_res)
    category_result  = fused_cat
    inference_source = str(fused_cat.get("source", "rule"))

    # ── Phase 5: department (ML-aware: prefer ML dept, else rule mapping) ─
    dept_code = detect_department(str(category_result["category_code"]))
    # If rule mapping returns empty but we have a category, try ML dept prediction
    if not dept_code and str(category_result.get("category_code", "")):
        try:
            from apps.ml.ml_inference import predict_department  # noqa: PLC0415
            dept_pred = predict_department(text)
            if dept_pred.confidence >= _ML_BLEND_THRESHOLD and dept_pred.label != "none":
                dept_code = dept_pred.label
        except Exception:  # noqa: BLE001
            pass

    # ── Phase 6: landmarks (rule-based + transformer hybrid) ─────────────
    landmark_result = extract_landmarks(text)
    # Supplement with transformer location intelligence when available.
    # If dictionary-based extraction found no landmarks, try embedding lookup.
    # If both find landmarks, merge them (dictionary result takes priority).
    if not landmark_result["landmarks"]:
        transformer_locs = _try_transformer_location(text)
        if transformer_locs:
            landmark_result = {
                "landmarks":  transformer_locs,
                "ward_hint":  transformer_locs[0]["ward_code"],
                "confidence": round(
                    min(0.3 + transformer_locs[0].get("score", 0.5) * 0.5, 0.75), 3
                ),
            }

    # ── Phase 7: priority (ML primary, rule fallback) ────────────────────
    ml_priority = _try_ml_priority(text, str(category_result["category_code"]))
    if ml_priority is not None:
        priority = ml_priority
    else:
        priority = predict_priority(text, str(category_result["category_code"]))

    # ── Phase 8: duplicate check (ML cosine primary, Jaccard fallback) ───
    ml_dup = _try_ml_duplicate(text, recent_texts or [])
    if ml_dup is not None:
        dup_result = ml_dup
    else:
        dup_result = detect_possible_duplicate(text, recent_texts or [])

    # ── Phase 9: overall confidence ──────────────────────────────────────
    overall_confidence = score_analysis(
        language_result=lang_result,
        category_result=category_result,
        landmark_result=landmark_result,
        spam_result=spam_result,
        duplicate_result=dup_result,
    )

    # ── Phase 10: human-review flags ─────────────────────────────────────
    review_reasons: list[str] = []
    if spam_result["is_spam"] or float(spam_result.get("spam_score", 0.0)) > 0.4:
        review_reasons.append("spam_suspicion")
    if not category_result["category_code"]:
        review_reasons.append("no_category_detected")
    if not landmark_result["landmarks"]:
        review_reasons.append("no_landmark_detected")
    if dup_result["is_duplicate"]:
        review_reasons.append("possible_duplicate")
    if overall_confidence < 0.35 and not spam_result["is_spam"]:
        review_reasons.append("low_confidence")
    if category_result["category_code"] and float(lang_result["confidence"]) < 0.25:
        review_reasons.append("language_uncertain")

    # ── Phase 11: image evidence (optional) ─────────────────────────────
    image_analysis: dict | None = None
    if image_input is not None:
        # Import lazily so analyzer.py has no hard Pillow dependency.
        from apps.ml.image_analyzer import analyze_image  # noqa: PLC0415
        image_analysis = analyze_image(
            image_input,
            text_category=str(category_result["category_code"]),
        )

        # Apply confidence penalties and collect review flags.
        if not image_analysis["is_valid"]:
            overall_confidence = round(overall_confidence * 0.85, 3)
            review_reasons.append("image_invalid")
        elif not image_analysis["usable"]:
            overall_confidence = round(overall_confidence * 0.90, 3)
            review_reasons.append("image_poor_quality")
        elif image_analysis["is_irrelevant"]:
            overall_confidence = round(overall_confidence * 0.85, 3)
            review_reasons.append("image_irrelevant")

        if not image_analysis["is_consistent"]:
            overall_confidence = round(overall_confidence * 0.80, 3)
            if "image_contradicts_complaint" not in review_reasons:
                review_reasons.append("image_contradicts_complaint")

    # ── Phase 12: decision intelligence ─────────────────────────────────
    from apps.ml.decision_engine import make_final_decision  # noqa: PLC0415

    _decision_input: dict[str, object] = {
        "spam":                spam_result,
        "duplicate":           dup_result,
        "image_analysis":      image_analysis,
        "category_confidence": float(category_result["confidence"]),
        "language_confidence": float(lang_result["confidence"]),
        "department_code":     dept_code,
        "ward_hint":           landmark_result["ward_hint"],
        "priority":            priority,
        "category_code":       str(category_result["category_code"]),
        "review_reasons":      review_reasons,
    }
    decision = make_final_decision(_decision_input)

    # Phase C may extend review_reasons (deduplicated inside make_final_decision).
    review_reasons = list(decision["review_reasons"])

    return {
        "language":            effective_language,
        "language_confidence": lang_result["confidence"],
        "normalized_text":     normalized,
        "category_code":       str(category_result["category_code"]),
        "category_confidence": float(category_result["confidence"]),
        "department_code":     dept_code,
        "landmarks":           landmark_result["landmarks"],
        "ward_hint":           landmark_result["ward_hint"],
        "landmark_confidence": float(landmark_result["confidence"]),
        "priority":            priority,
        "spam":                spam_result,
        "duplicate":           dup_result,
        "needs_human_review":  bool(review_reasons),
        "review_reasons":      review_reasons,
        "confidence":          overall_confidence,
        "image_analysis":      image_analysis,
        "decision":            decision,
        "inference_source":    inference_source,
    }
