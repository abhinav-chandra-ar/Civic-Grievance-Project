"""apps/ml/training/corpus_data_v3.py

Bias-repair additions to the TVMC ML training corpus.

Rationale
---------
Audit run (2026-05) surfaced four weaknesses:
  1. Malayalam subgroup F1 gap = 0.368 — model trained on too few native-script
     samples for drainage / sewage_issue / illegal_construction.
  2. Drainage vs sewage confusion — confusion matrix showed 4 drainage predicted
     as sewage_issue; both share "overflow", "manhole", "smell" vocabulary.
  3. Priority inflation — 4/12 emotional-language pairs inflated to urgent;
     model conflates frustrated register with urgency signal.
  4. Location intelligence weak — TVM_LOCATIONS only has single ward names;
     real complaints use compound landmark phrases.

This file provides:
  DRAINAGE_ML_ADDITIONS         — 38 Malayalam seeds (native script + Manglish)
  SEWAGE_ML_ADDITIONS           — 38 Malayalam seeds (native script + Manglish)
  ILLEGAL_CONSTRUCTION_ML_ADDITIONS — 38 Malayalam seeds
  DRAINAGE_SEWAGE_CONTRASTIVE   — 55 contrastive English/Manglish/Malayalam pairs
  PRIORITY_EMOTIONAL_ANCHORS    — 44 frustrated/angry texts with CORRECT priority
  TVM_LOCATION_ALIASES          — 52 compound landmark phrases for TransformerEngine

generate_corpus_v2.py imports and merges these into ALL_SAMPLES automatically.
train_transformer.py uses TVM_LOCATION_ALIASES for landmark pre-encoding.
"""
from __future__ import annotations

from apps.ml.training.corpus_data_v2 import TrainingSample, TVM_LOCATIONS

# ===========================================================================
# FIX 1-A: Malayalam additions — DRAINAGE (38 seeds)
# Vocabulary anchors: mazha (rain), vellam (water), oda/channal (channel),
# thadayu (block), nirachuvazhukal (overflow), vellakkettu (flooding),
# vattu/kuzhi (sinkhole), mazhavellam (rainwater/stormwater)
# Register diversity: formal petition, angry WhatsApp, vague, typo/informal
# ===========================================================================

DRAINAGE_ML_ADDITIONS: list[TrainingSample] = [
    # ── formal report style ─────────────────────────────────────────────
    (
        "ഞങ്ങളുടെ കോളനിയിലെ മഴവെള്ള ഓട ഗ്രാവൽ, പ്ലാസ്റ്റിക് മാലിന്യം കൊണ്ട് പൂർണ്ണമായി അടഞ്ഞ് കിടക്കുന്നു. "
        "ഒരു മഴ പെയ്താൽ ഞങ്ങളുടെ മൂന്ന് തെരുവുകൾ കൂടി വെള്ളക്കെട്ടാകുന്നു. ഉടൻ ഇടപെടൽ ആവശ്യം.",
        "drainage", "high", "drainage",
    ),
    (
        "പ്രധാന ചാൽ കനാൽ ഒരു മാസത്തേറെയായി ശുദ്ധി ചെയ്തിട്ടില്ല. "
        "ഓണക്കാലം ആകുന്നതോടെ ഇത് ഗുരുതരമായ വെള്ളക്കെട്ടിന് ഇടയാക്കും. "
        "ഈ വിഷയത്തിൽ കോർപ്പറേഷനിൽ നിന്ന് ഉടൻ നടപടി പ്രതീക്ഷിക്കുന്നു.",
        "drainage", "high", "drainage",
    ),
    (
        "റോഡ് അരികിലുള്ള ഓടചാൽ ഇളകി, ഒരു ഭാഗം തകർന്നുവീണ് ഭൂഗർഭ ഗർത്തം ഉണ്ടായി. "
        "ദ്വിചക്ര വാഹനങ്ങൾ ഇതിൽ പതിക്കാൻ സാധ്യത ഏറെ. ഉടൻ അറ്റകുറ്റപ്പണി വേണം.",
        "drainage", "urgent", "drainage",
    ),
    (
        "ബ്ലോക്ക് ആയ ഓടചാൽ ഒഴുകാൻ ഇടമില്ലാതെ, ഞങ്ങളുടെ ഭവനസമുച്ചയ ഗ്രൗണ്ടിലേക്ക് "
        "മഴ ദിവസം മഴവെള്ളം കയറുന്നു. ഗ്രൗണ്ടിലെ ഫ്ലോർ ഒരടി ജലത്തിൽ ആകുന്നു.",
        "drainage", "urgent", "drainage",
    ),
    (
        "ഒഴുക്ക് ചാൽ കൊമ്പ്, ഇലകൾ, ഭൂഗർഭ ഗ്രാവൽ കൊണ്ട് അടഞ്ഞ് ഒഴുക്ക് തടഞ്ഞ് "
        "മഴ വെള്ളം റോഡ് ക്രോസ്സ് ചെയ്ത് ഗ്രൗണ്ടിൽ കെട്ടി. "
        "ഇതിൽ ഡെങ്കി ഉൾപ്പടെ രോഗ ഭീഷണി ഉണ്ട്.",
        "drainage", "medium", "drainage",
    ),

    # ── angry / frustrated register ─────────────────────────────────────
    (
        "ഇതൊക്കെ എന്ന് ശരിയാകും? ഓടചാൽ ബ്ലോക്ക് ആണ് മൂന്ന് മാസം ആയി! "
        "ഒരു ഫോൺ ചെയ്‌താൽ 'ചെക്ക് ചെയ്യാം' പറഞ്ഞ് ആരും വരുന്നില്ല. "
        "ഇനിയും മഴ പെയ്‌താൽ ഞങ്ങളുടെ വീട്ടിൽ വെള്ളം കയറും.",
        "drainage", "high", "drainage",
    ),
    (
        "ഒരു മഴ പെയ്‌താൽ ഓടചാൽ നിറഞ്ഞ് തെരുവ് ഒക്കെ ഒഴുകുന്നു. "
        "ഇക്കാര്യം ഒരു വർഷം ആയി പറഞ്ഞ് കൊണ്ടിരിക്കുന്നു, ഒന്നും ചെയ്തില്ല. "
        "ആർക്ക് ഇക്കാര്യം ഒന്ന് നോക്കിയാൽ?",
        "drainage", "high", "drainage",
    ),
    (
        "ഡ്രെയ്നേജ് ക്ലിയർ ചെയ്യാൻ ആർക്കും സമയമില്ല, മഴ ആകുമ്പോൾ ആർക്കും ഉത്തരം ഇല്ല! "
        "വർഷം ഒന്ന് ആകും ഇതൊരേ കഥ. ഉദ്യോഗസ്ഥർ ഒന്ന് ശ്രദ്ധിക്ക്.",
        "drainage", "high", "drainage",
    ),

    # ── short / vague / incomplete ───────────────────────────────────────
    (
        "ഓടചാൽ ബ്ലോക്ക്, മഴ വന്നാൽ പ്രശ്‌നം",
        "drainage", "high", "drainage",
    ),
    (
        "ചാൽ ഒഴുകുന്നില്ല, റോഡ് ഒക്കെ വെള്ളം",
        "drainage", "high", "drainage",
    ),
    (
        "ഡ്രെയ്ൻ കവർ ഇല്ല, അപകടം",
        "drainage", "high", "drainage",
    ),
    (
        "വെള്ളക്കെട്ട് ഉണ്ട്, ഓട തടഞ്ഞ്",
        "drainage", "medium", "drainage",
    ),

    # ── long / detailed / formal petition ───────────────────────────────
    (
        "ഞങ്ങളുടെ ഭൂഖണ്ഡത്തിൽ കഴിഞ്ഞ അഞ്ച് വർഷമായി ഒരു ശരിയായ ഡ്രെയ്നേജ് ചാൽ ഇല്ല. "
        "ഓരോ മൺസൂൺ കാലത്തും ഞങ്ങൾ വെള്ളക്കെട്ടിൽ ജീവിക്കേണ്ടി വരുന്നു. "
        "ഇക്കഴിഞ്ഞ ഇരുപത് ദിവസം ഞങ്ങളുടെ ഗ്രൗണ്ടിൽ ജലം ഒഴിഞ്ഞ് പോയിട്ടില്ല. "
        "വൈദ്യ ഉദ്യോഗസ്ഥർ ഡെങ്കി ഭീഷണി ഉണ്ടെന്ന് അറിയിക്കുന്നു. "
        "കോർപ്പറേഷൻ ഒരു ശാശ്വത പ്രതിവിധി കൊണ്ട് വരണം.",
        "drainage", "urgent", "drainage",
    ),
    (
        "പ്രധാന ഡ്രെയ്നേജ് ചാൽ നഗരത്തിന്റെ ഈ ഭാഗം ക്ലിയർ ചെയ്‌തില്ലെങ്കിൽ ഈ മൺസൂണിൽ "
        "വൻ ദുരന്തം ഉണ്ടാകും. ഭൂഗർഭ ജലം ഉയർന്ന് ഒഴുക്ക് ചാൽ വഴി "
        "ആറ് ഫ്ലാറ്റ് കോം‌പ്ലക്‌സുകൾ ഡൂൺ ആകും. ഉടൻ ഇടപെടൽ ആവശ്യം.",
        "drainage", "urgent", "drainage",
    ),

    # ── typo / WhatsApp informal ─────────────────────────────────────────
    (
        "channal block aanu 3 masam aai help vendo urgent plz",
        "drainage", "high", "drainage",
    ),
    (
        "oda nirachuvazhukannu rain vandal road muzhuvan vellam",
        "drainage", "high", "drainage",
    ),
    (
        "mazha vandal oru vaaram aayittu vellam road il nikunnu drain clear cheyyende",
        "drainage", "high", "drainage",
    ),

    # ── mixed formal/informal Malayalam ─────────────────────────────────
    (
        "Drain channel block aayittu mazha kaalam aayi, flood aakum ennu parayunnu — "
        "ഇക്കാര്യം ഉടൻ ശ്രദ്ധിക്കണം, ദയവായി.",
        "drainage", "high", "drainage",
    ),
    (
        "ഞങ്ങളുടെ road side drain clear ചെയ്‌തിട്ട് ഒരു year ആയി — "
        "monsoon ku mump oru thadava clean cheyyanam please.",
        "drainage", "medium", "drainage",
    ),
    (
        "Stormwater channel block aanu — ഈ ഭാഗം flood risk ഉണ്ട്, ഉടൻ ക്ലീൻ ചെയ്യണം.",
        "drainage", "high", "drainage",
    ),

    # ── additional varied Malayalam seeds ───────────────────────────────
    (
        "ഞങ്ങളുടെ ഇടവഴിയിൽ മഴ ദിവസം ഡ്രെയ്ൻ കൂടി ഒഴുകി കൊതുക് ഉൽഭവ കേന്ദ്രം ആകുന്നു. "
        "ഇക്കഴിഞ്ഞ ആഴ്ചയിൽ നിരവധി കുട്ടികൾ ഡെങ്കി ജ്വരം ബാധിച്ച് ചികിത്സ തേടി.",
        "drainage", "medium", "drainage",
    ),
    (
        "ഒഴുക്ക് ചാൽ കുറ്റി ആകൽ കൊണ്ട് ഈ ഭൂഖണ്ഡം ഒട്ടും മഴ വെള്ളം ഒഴുകി പോകാൻ ഇടം ഇല്ലാതായി. "
        "ഭൂഗർഭ ഗർത്തം ഉണ്ടാകാൻ ഇത് കാരണമാകും.",
        "drainage", "urgent", "drainage",
    ),
    (
        "ഓടചാൽ ഗ്രേറ്റിംഗ് ഇളകി, ദ്വിചക്ര വാഹനം ഒടിക്കുന്നവർ ഉള്ളിൽ ആഴ്‌ന്ന് ദ്വന്ദ്വ അർഥ "
        "ആവുന്നു. ഗ്രേറ്റ് ഇൻ‌സ്‌റ്റോൾ ചെയ്‌ത് ഗ്രേറ്റ് ബ്ലോക്ക് ഒഴിവാക്കണം.",
        "drainage", "high", "drainage",
    ),
    (
        "ഓടചാൽ level road level ൽ നിന്ന് കുറഞ്ഞ് ഒഴുകി, ഓരോ വർഷവും ഒരേ "
        "വെള്ളക്കെട്ട് ഉണ്ടാകുന്നു. ഇതൊരു ദീർഘ‌കാല പ്രശ്‌നം.",
        "drainage", "high", "drainage",
    ),
    (
        "ഓടചാൽ drain outlet canal ൽ തടഞ്ഞ്, ഒരു ദീർഘ ഭാഗം road ൽ ജലം കെട്ടിക്കിടക്കുന്നു.",
        "drainage", "high", "drainage",
    ),
    (
        "mazha kaalam aayi, drain clear cheyyaathe, road muzhuvan vellathinadiyil aanu",
        "drainage", "high", "drainage",
    ),
    (
        "oda cover illathe, rathri kazhinjal aaraanenkilum veezhum, urgent",
        "drainage", "high", "drainage",
    ),
    (
        "sinkhol undaayi, manhole kazhinjal, drain keezhil pipe pottannu veganam nokkanam",
        "drainage", "urgent", "drainage",
    ),
    (
        "ഒഴുക്ക് ചാൽ ക്ലിയർ ചെയ്‌തിട്ട് ഒരു വർഷം ആയി, ഈ monsoon ൽ "
        "ഒഴുക്ക് ഇല്ലെങ്കിൽ ഞങ്ങൾ flood ആകും",
        "drainage", "high", "drainage",
    ),
    (
        "drain cover missing bus stop kazhijal, night il kittunnilla, very dangerous",
        "drainage", "high", "drainage",
    ),
    (
        "road side channal oru maasam block, corporation annu vannu check cheythu, "
        "oru muppattu divasam aayi clear aayittilla",
        "drainage", "medium", "drainage",
    ),
    (
        "Drain channel full of leaves plastic, monsoon coming, flood sure aakum, "
        "ഉടൻ clear ചെയ്‌ത് ഞങ്ങളെ സഹായിക്കണം",
        "drainage", "high", "drainage",
    ),
]


# ===========================================================================
# FIX 1-B: Malayalam additions — SEWAGE ISSUE (38 seeds)
# Vocabulary anchors: malinya jalam (sewage/waste water), sewer pipe, septic tank,
# toilet mazhalappu (toilet waste), azhukku (filth), durgandham (foul smell),
# malinam (contaminated), kazhappu (excreta)
# Critical distinction: sewage = TOILET WASTE / SEWER SYSTEM (not stormwater)
# ===========================================================================

SEWAGE_ML_ADDITIONS: list[TrainingSample] = [
    # ── formal report style ─────────────────────────────────────────────
    (
        "ഞങ്ങളുടെ ഇടവഴിയിലെ സ്യൂവേജ് മാൻ‌ഹോൾ ഒഴുകി, ടോയ്‌ലറ്റ് മലിന ജലം "
        "റോഡിലേക്ക് ഒഴുകി. ബസ് സ്‌റ്റോപ്പ് പ്രദേശം ദുർഗന്ധം കൊണ്ട് "
        "ഉപയോഗ‌ശൂന്യം ആയിരിക്കുന്നു.",
        "sewage_issue", "urgent", "sewage",
    ),
    (
        "ഭൂഗർഭ സ്യൂവേജ് ലൈൻ ഭൂഗർഭ ഭഭഒടിഞ്ഞ്, "
        "ടോയ്‌ലറ്റ് മലിന ജലം ഒരു ആഴ്ചയായി ഒഴുകി. "
        "ഗൗരവ ആരോഗ്യ ഭീഷണി ഇടയ്‌ക്കിടെ ഉണ്ടാകുന്നതിനാൽ ഉടൻ ഇടപെടൽ ആവശ്യം.",
        "sewage_issue", "high", "sewage",
    ),
    (
        "ഞങ്ങളുടെ ഗ്രൗണ്ട് ഫ്ലോർ ഫ്ലാറ്റിൽ ബാത്ത്‌റൂം ഉപയോഗിക്കാൻ കഴിയുന്നില്ല. "
        "ടോയ്‌ലറ്റ് waste ഒഴുകി ഒരു ദിവസം ആയി. ഇവിടം ജീവിക്കാൻ ഇടമല്ല.",
        "sewage_issue", "urgent", "sewage",
    ),
    (
        "സ്‌കൂളിന് സമീപം ഒഴുക്കി ടോയ്‌ലറ്റ് മലിന ജലം. "
        "കുട്ടികൾ കളിക്കുന്ന ഇടത്ത് ഈ അഴുക്ക് ഒഴുകി. "
        "ഇത് ഗുരുതരമായ ആരോഗ്യ ഭീഷണിയാണ്, ഉടൻ ശ്രദ്ധ ആവശ്യം.",
        "sewage_issue", "urgent", "sewage",
    ),
    (
        "ഒരു അടുത്ത ഫ്ലാറ്റ് സ്‌യൂവേജ് ലൈൻ ഭൂഗർഭ കനൽ ലൈനിൽ കൂട്ടി. "
        "ഈ ഭൂഗർഭ ലൈൻ ഇനി ഞങ്ങളുടെ കുടി‌വെള്ളം supply ചെയ്യുന്ന ലൈനിൽ "
        "ഭേദം ഉണ്ടാക്കും. ഇത് ഒരു ഗൗരവ ഭീഷണി.",
        "sewage_issue", "high", "sewage",
    ),
    (
        "ഒഴുക്ക് ലൈൻ തകർന്ന് ടോയ്‌ലറ്റ് waste ഒഴുകി, ഞങ്ങളുടെ ഫ്ലാറ്റ് "
        "ഇടനാഴി ഒക്കെ ദുർഗന്ധം ഉണ്ട്, ബാത്ത്‌റൂം use ചെയ്യൻ ഇടം ഇല്ല.",
        "sewage_issue", "urgent", "sewage",
    ),

    # ── angry / frustrated register ─────────────────────────────────────
    (
        "ടോയ്‌ലറ്റ് waste ഒഴുകി വരുന്നത് ഒരു ആഴ്ചയായി! "
        "ആർക്കും ഇക്കാര്യം ഗൗരവം ഇല്ലേ? "
        "ഇനിയും കാത്ത് നിൽക്കേണ്ടി വരുമോ?",
        "sewage_issue", "urgent", "sewage",
    ),
    (
        "Sewage smell ഒരു മാസം ആയി! ഒന്നും ചെയ്‌തില്ല! "
        "ആർക്കോ ഇക്കാര്യം ഒന്ന് ശ്രദ്ധിക്കണ്ടേ? "
        "ഞങ്ങൾ ഇവിടം ജീവിക്കണം.",
        "sewage_issue", "high", "sewage",
    ),
    (
        "ടോയ്‌ലറ്റ് ഒഴുക്ക് ഒരു ആഴ്ചയായി ശരിയായില്ല, landlord ഒന്നും ചെയ്‌തില്ല. "
        "ഇനി ഞങ്ങൾ corporation ൽ complaint ഇടേണ്ടി വരും.",
        "sewage_issue", "urgent", "sewage",
    ),

    # ── short / vague / incomplete ───────────────────────────────────────
    (
        "ടോയ്‌ലറ്റ് waste ഒഴുകുന്നു, ദുർഗന്ധം",
        "sewage_issue", "urgent", "sewage",
    ),
    (
        "sewer block, bathroom use cheyyaan patilla",
        "sewage_issue", "urgent", "sewage",
    ),
    (
        "manhole open, toilet smell, road il",
        "sewage_issue", "high", "sewage",
    ),
    (
        "septic tank nirachu, ithil ninnu oranzh varunnu",
        "sewage_issue", "urgent", "sewage",
    ),

    # ── long / detailed ──────────────────────────────────────────────────
    (
        "ഞങ്ങളുടെ ലൈനിൽ ടോയ്‌ലറ്റ് waste ഒഴുകി, ഇടവഴി ഒക്കെ ദുർഗന്ധം ഉണ്ട്. "
        "ഇക്കഴിഞ്ഞ ആഴ്ചയിൽ നിരവധി കുട്ടികൾ ആരോഗ്യ പ്രശ്‌നം ഉണ്ടായി. "
        "ആശുപത്രി ഡോക്ടർ ഇത് sewage contamination ആണ് കൊണ്ടന്ന് "
        "ഉറപ്പ് ഇട്ടു. ഇക്കാര്യത്തിൽ ഉടൻ ഇടപെടൽ ആവശ്യം.",
        "sewage_issue", "urgent", "sewage",
    ),
    (
        "ഞങ്ങളുടെ കോളനിയിൽ ഭൂഗർഭ sewage ലൈൻ ഒടിഞ്ഞ്, ടോയ്‌ലറ്റ് waste ഭൂഗർഭ "
        "ജലത്തിൽ ഇറക്കുന്നു. ഇത് ഞങ്ങളുടെ കിണർ ജലം ദൂഷണം ആക്കും. "
        "ഒരു ആഴ്ചയായി ഈ പ്രശ്‌നം ഉണ്ട്, ഒന്നും ചെയ്‌തില്ല.",
        "sewage_issue", "high", "sewage",
    ),

    # ── Manglish ─────────────────────────────────────────────────────────
    (
        "toilet waste oru vaaram aayittu road il vannu, corporation annu vannu "
        "check cheythu, oru thadava aayi ithum aayittilla",
        "sewage_issue", "urgent", "sewage",
    ),
    (
        "sewer line pottiyannu, toilet smell veetil niranju, bathroom use patilla",
        "sewage_issue", "urgent", "sewage",
    ),
    (
        "manhole nirachu toilet waste road il varunnu, school kazhijal, children risk",
        "sewage_issue", "urgent", "sewage",
    ),
    (
        "septic tank overflow aayittu lane il toilet waste nikunnu, very serious",
        "sewage_issue", "urgent", "sewage",
    ),
    (
        "sewage pump breakdown, tank nirachu, overflow aakaan neram aayi",
        "sewage_issue", "urgent", "sewage",
    ),

    # ── mixed formal/informal ────────────────────────────────────────────
    (
        "Sewage line blocked aayittu — ഞങ്ങളുടെ bathroom ഉപയോഗിക്കാൻ ഇടം ഇല്ല, "
        "ടോയ്‌ലറ്റ് waste ഒഴുകി വരുന്നു, urgent help needed.",
        "sewage_issue", "urgent", "sewage",
    ),
    (
        "Sewer pipe cracked aanu — ഭൂഗർഭ waste ഒഴുകി, road soft ആയി, smell awful.",
        "sewage_issue", "high", "sewage",
    ),
    (
        "Septic tank full aayittu — ഇടനാഴിയിൽ ടോയ്‌ലറ്റ് ദ്രാവകം ഒഴുകുന്നു, "
        "urgent ആയി empty ചെയ്യണം.",
        "sewage_issue", "urgent", "sewage",
    ),

    # ── additional varied seeds ──────────────────────────────────────────
    (
        "ടോയ്‌ലറ്റ് ഒഴുക്ക് ലൈൻ ഒഴുകി, sewage ഒഴുകി, കനൽ ലൈൻ ദൂഷണം ആകുന്നു.",
        "sewage_issue", "critical", "sewage",
    ),
    (
        "toilet kazhijal sewage back up aayittu ഗ്രൗണ്ട് ഫ്ലോർ ഫ്ലാറ്റ് useless",
        "sewage_issue", "urgent", "sewage",
    ),
    (
        "ഭൂഗർഭ sewer ലൈൻ ഒടിഞ്ഞ്, ടോയ്‌ലറ്റ് waste ഒഴുകി, ദുർഗന്ധം "
        "ഒരു ആഴ്ചയായി. ഒരു ഇടപെടൽ ആവശ്യം.",
        "sewage_issue", "high", "sewage",
    ),
    (
        "STP overflow aayittu canal il toilet waste vannu, water polluted aakum",
        "sewage_issue", "critical", "sewage",
    ),
    (
        "open manhole toilet waste smell, road il, night il veezhum aaraanenkilum",
        "sewage_issue", "high", "sewage",
    ),
    (
        "sewage back up aayittu flat il, ഗ്രൗണ്ട് ഫ്ലോർ bathroom use cheyyaan pattilla",
        "sewage_issue", "urgent", "sewage",
    ),
    (
        "sewer pipe pottiyannu, road soft spot undayi, toilet waste underground leak",
        "sewage_issue", "high", "sewage",
    ),
    (
        "commercial building grease trap toilet waste municipal sewer il ittaanu, illegal",
        "sewage_issue", "high", "sewage",
    ),
    (
        "ടോയ്‌ലറ്റ് ദ്രാവകം ഒഴുകി market kazhijal, flies full, health hazard",
        "sewage_issue", "urgent", "sewage",
    ),
    (
        "ഭൂഗർഭ sewer ലൈൻ ബ്ലോക്ക്, ഒഴുക്ക് ഇല്ല, ദ്രുത ഇടപെടൽ ആവശ്യം",
        "sewage_issue", "high", "sewage",
    ),
    (
        "toilet pipe leak cheythu veetinu ullil, smell veetil niranju, "
        "ഒരു ആഴ്ചയായി ഇക്കാര്യം ശ്രദ്ധ ഇല്ല",
        "sewage_issue", "high", "sewage",
    ),
]


# ===========================================================================
# FIX 1-C: Malayalam additions — ILLEGAL CONSTRUCTION (38 seeds)
# Vocabulary anchors: anumathi illaathe (without permission), pothubhumi (public land),
# atikrama nirmmanam (encroachment construction), plan langanganam (plan violation),
# floor koodi (extra floor), setback illa (no setback), CRZ
# ===========================================================================

ILLEGAL_CONSTRUCTION_ML_ADDITIONS: list[TrainingSample] = [
    # ── formal report style ─────────────────────────────────────────────
    (
        "ഞങ്ങളുടെ ഇടവഴി സമീപം പൊതുഭൂമിയിൽ അനുമതി ഇല്ലാതെ ഒരു മൂന്ന് നില "
        "കെട്ടിടം പണി ചെയ്‌ത് കൊണ്ടിരിക്കുന്നു. ഒരു ബോർഡ് പോലും ദൃശ്യമല്ല. "
        "ഉടൻ ഇടപെടൽ ആവശ്യം.",
        "illegal_construction", "high", "planning",
    ),
    (
        "ഞങ്ങളുടെ കോളനിയിൽ ഒരു ഭൂഖണ്ഡ ഉടമ, അനുമതി ഇല്ലാതെ ഒരു "
        "രണ്ടടി ഉയരമുള്ള compound wall, ഇടവഴിയിൽ കെട്ടി. "
        "Wheelchair users ഇനി ഇടവഴിയിലൂടെ കടക്കാൻ ഇടം ഇല്ല.",
        "illegal_construction", "medium", "planning",
    ),
    (
        "ഒരു commercial building കടൽ തീരത്ത് CRZ zone ൽ നിർമ്മാണം ആരംഭിച്ചു. "
        "ഇത് coastal norms ലംഘനം ആണ്. ഉടൻ stop work notice ആവശ്യം.",
        "illegal_construction", "urgent", "planning",
    ),
    (
        "ഒരു apartment complex, approved plan ൽ ഉള്ള floor ൽ നിന്ന് "
        "കൂടുതൽ ഒരു floor കൂടി ചേർക്കൽ ആരംഭിച്ചിരിക്കുന്നു. "
        "ഈ plan violation ഉടൻ ശ്രദ്ധ ആകർഷിക്കണം.",
        "illegal_construction", "high", "planning",
    ),
    (
        "ഒരു കൃഷ്ണ ശ്യാമ, ഭൂഗർഭ ഒഴുക്ക് ചാൽ ഇടുങ്ങി, ഒരു "
        "കെട്ടിടം പണിഞ്ഞ്. ഇതിൽ വർഷം ഒന്ന് ആകുമ്പോൾ ഈ ഒഴുക്ക് "
        "ചാൽ ഒഴുകാൻ ഇടം ഇല്ലാതെ flood ഉണ്ടാക്കും.",
        "illegal_construction", "high", "planning",
    ),

    # ── angry / frustrated register ─────────────────────────────────────
    (
        "ഇത് ഒരു കൊല്ലം ആയി ആ permit ഇല്ലാ building permit ഇല്ലാ "
        "നിർമ്മാണം കണ്ടിട്ടും ആർക്കും ഒന്നും ചെയ്‌തില്ല! "
        "Corporation officer ഒന്ന് ഇക്കാര്യം ശ്രദ്ധിക്കണ്ടേ?",
        "illegal_construction", "high", "planning",
    ),
    (
        "ഒരു neighbouring compound wall public road ൽ ഉണ്ടാക്കി, "
        "ഇനി traffic ഒരു lane ൽ ആണ്. ഒരു കൊല്ലം ആയി ഇക്കാര്യം "
        "ഒരു government official ഉം ശ്രദ്ധിച്ചില്ല!",
        "illegal_construction", "high", "planning",
    ),
    (
        "Midnight ൽ construction ആരംഭിച്ചു, permit ഇല്ല, "
        "ഒച്ചപ്പാട് കൊണ്ട് ഉറങ്ങൻ ഇടം ഇല്ല! "
        "ഇത് ഒന്ന് ശ്രദ്ധ ആകർഷിക്കണം.",
        "illegal_construction", "medium", "planning",
    ),

    # ── short / vague / incomplete ───────────────────────────────────────
    (
        "permit ഇല്ലാ construction, public land",
        "illegal_construction", "high", "planning",
    ),
    (
        "extra floor anumathi illatha, violation",
        "illegal_construction", "high", "planning",
    ),
    (
        "CRZ zone nirmmanam, urgent",
        "illegal_construction", "urgent", "planning",
    ),
    (
        "compound wall road il, encroachment",
        "illegal_construction", "high", "planning",
    ),

    # ── long / detailed petition ─────────────────────────────────────────
    (
        "ഞങ്ങളുടെ residential colony ൽ ഒരു industrial unit, "
        "green belt ൽ anumathi ഇല്ലാതെ ആരംഭിച്ചു. ഇത് ഒരു "
        "ആഴ്ചയായി നിർമ്മാണം. ദുർഗന്ധം, ശബ്ദ ദൂഷണം ഉണ്ടാകുന്നു. "
        "TVM Corporation ഇക്കാര്യം ശ്രദ്ധ ആകർഷിക്കണം.",
        "illegal_construction", "high", "planning",
    ),
    (
        "ഒരു Poultry farm, residential zone ൽ anumathi ഇല്ലാതെ "
        "ആരംഭിച്ചിരിക്കുന്നു. ഈ ഭൂഖണ്ഡം ഒക്കെ ദുർഗന്ധം "
        "ഉണ്ടാകുന്നു. ഇക്കാര്യത്തിൽ ഒരു action ആവശ്യം.",
        "illegal_construction", "medium", "planning",
    ),

    # ── Manglish ─────────────────────────────────────────────────────────
    (
        "permit illatha construction government land il, multi storey, oru board illakill",
        "illegal_construction", "high", "planning",
    ),
    (
        "neighbour compound wall road il, encroachment, traffic narrowed down",
        "illegal_construction", "high", "planning",
    ),
    (
        "CRZ violation near beach, urgent stop order venam",
        "illegal_construction", "urgent", "planning",
    ),
    (
        "extra floor anumathi illatha, corporation officer oru action venam",
        "illegal_construction", "high", "planning",
    ),
    (
        "shop footpath il build cheythu, pedestrians road il irangani, encroachment",
        "illegal_construction", "medium", "planning",
    ),
    (
        "government land enclosed, private construction started, illegal",
        "illegal_construction", "high", "planning",
    ),
    (
        "green belt il commercial complex, town plan violation, urgent",
        "illegal_construction", "urgent", "planning",
    ),

    # ── mixed formal/informal ────────────────────────────────────────────
    (
        "Permit illatha construction pothubhoomiyil — ഉടൻ ഇടപെടൽ ആവശ്യം.",
        "illegal_construction", "high", "planning",
    ),
    (
        "CRZ zone il construction — ഈ coastal violation ഒരു official ഉം "
        "ശ്രദ്ധ ഇട്ടില്ല.",
        "illegal_construction", "urgent", "planning",
    ),
    (
        "Drainage path block cheythu building — ഇതിൽ flood ഉണ്ടാകും.",
        "illegal_construction", "high", "planning",
    ),
    (
        "Corporation land enclose cheythu — private construction started, "
        "ഒരു stop notice ആവശ്യം.",
        "illegal_construction", "high", "planning",
    ),

    # ── additional varied seeds ──────────────────────────────────────────
    (
        "ഒരു retaining wall, natural drainage path block ചെയ്‌ത്, "
        "ഞങ്ങളുടെ compound ൽ flooding ഉണ്ടാക്കുന്നു.",
        "illegal_construction", "high", "planning",
    ),
    (
        "access road block cheythu, ഒരു new construction ആരംഭിച്ചു, "
        "ഇനി ഞങ്ങൾ colony ൽ ഇറങ്ങൻ ഇടം ഇല്ലാതാകും.",
        "illegal_construction", "high", "planning",
    ),
    (
        "ഒരു ഭൂഖണ്ഡ ഉടമ, ഭൂഗർഭ drainage channel ൽ building "
        "ഇട്ട്, ഒഴുക്ക് ചാൽ block ചെയ്‌ത്. Rain il flooding ഉണ്ടാകുന്നു.",
        "illegal_construction", "high", "planning",
    ),
    (
        "New construction ഇടവഴിക്ക് setback ഇല്ലാതെ, wall ഞങ്ങളുടെ "
        "compound wall ൽ ചേർന്ന്. ഇക്കാര്യം ഉടൻ ഇടപെടൽ ആവശ്യം.",
        "illegal_construction", "high", "planning",
    ),
    (
        "ഒരു commercial shop footpath ൽ extend ചെയ്‌ത്, "
        "pedestrians block ആകുന്നു. ഉടൻ encroachment നടപടി.",
        "illegal_construction", "medium", "planning",
    ),
    (
        "construction midnight il aarambhichu permit illakill, "
        "noise kaaran uyangaan pattunilla, residents upset",
        "illegal_construction", "medium", "planning",
    ),
    (
        "ഒരു land, corporation ഉടൻ action ആവശ്യം enclose cheythu "
        "private construction, permit board ഇല്ല",
        "illegal_construction", "high", "planning",
    ),
    (
        "multi storey building pothubhoomiyil, oru permit board illakill, "
        "ഉടൻ ഇടപെടൽ ആവശ്യം",
        "illegal_construction", "high", "planning",
    ),
    (
        "annexe build cheythu without permission, floor count exceed, violation",
        "illegal_construction", "high", "planning",
    ),
]


# ===========================================================================
# FIX 2: DRAINAGE vs SEWAGE CONTRASTIVE PAIRS
#
# Model currently confuses these two categories because they share vocabulary
# (overflow, manhole, smell). These pairs make the semantic distinction
# crystal-clear during training:
#   DRAINAGE  = stormwater / rainwater / road flooding / channel blockage
#   SEWAGE    = toilet waste / sewer pipe / septic tank / fecal smell
#
# Each pair is two separate TrainingSamples — one drainage, one sewage_issue.
# ===========================================================================

DRAINAGE_SEWAGE_CONTRASTIVE: list[TrainingSample] = [
    # ── English contrastive pairs ────────────────────────────────────────
    # Pair 1
    ("Rainwater drain beside the road is blocked. After every shower the road floods knee-deep.", "drainage", "high", "drainage"),
    ("Sewer line beside the road is cracked. Toilet waste is seeping up through the road surface.", "sewage_issue", "high", "sewage"),
    # Pair 2
    ("Stormwater channel near the school has plastic waste clogging it. During rain the schoolyard floods.", "drainage", "high", "drainage"),
    ("Sewage pipe near the school is broken. Toilet effluent is flowing into the schoolyard. Children at risk.", "sewage_issue", "urgent", "sewage"),
    # Pair 3
    ("The side drain that carries rainwater from the road to the canal is completely choked with silt.", "drainage", "medium", "drainage"),
    ("The sewer pipe that carries toilet waste from houses to the main line is completely blocked. Bathrooms are unusable.", "sewage_issue", "urgent", "sewage"),
    # Pair 4
    ("Open stormwater channel in front of our house overflows every monsoon and floods our compound.", "drainage", "urgent", "drainage"),
    ("Open sewer channel in front of our house has raw toilet waste flowing in it. Flies and unbearable smell.", "sewage_issue", "urgent", "sewage"),
    # Pair 5
    ("Drain grating near the bus stop is broken. Motorists' front wheels fall in during night.", "drainage", "high", "drainage"),
    ("Manhole near the bus stop has no cover. Raw sewage smell and waste visible inside.", "sewage_issue", "high", "sewage"),
    # Pair 6
    ("Road floods every time it rains because the stormwater outlet to the canal is blocked.", "drainage", "high", "drainage"),
    ("Road has a damp patch because the underground sewer pipe is leaking toilet waste.", "sewage_issue", "high", "sewage"),
    # Pair 7
    ("Leaves and gravel from the slope have choked the rain drain. The whole lane stands under water.", "drainage", "high", "drainage"),
    ("Septic tank of the adjacent building overflowed and toilet sludge spread across our lane.", "sewage_issue", "urgent", "sewage"),
    # Pair 8
    ("Sinkhole forming near the road junction because the underground stormwater pipe has collapsed.", "drainage", "urgent", "drainage"),
    ("Sinkhole forming near the road junction because the underground sewer pipe has collapsed and leaked.", "sewage_issue", "urgent", "sewage"),
    # Pair 9
    ("Mosquitoes breeding in the stagnant rainwater in the blocked side drain near the park.", "drainage", "medium", "drainage"),
    ("Flies and mosquitoes breeding around the open manhole with overflowing toilet waste near the park.", "sewage_issue", "urgent", "sewage"),
    # Pair 10
    ("After the road was relaid the stormwater drain level dropped. Rain now floods the ground floor flats.", "drainage", "high", "drainage"),
    ("After the road was relaid the sewer pipe connection was broken. Toilet waste is now backing up into flats.", "sewage_issue", "urgent", "sewage"),

    # ── Manglish contrastive pairs ───────────────────────────────────────
    # Pair 11
    ("mazhavellam road il nikunnu, channal block aanu, drain ila veganam clear cheyyenam", "drainage", "high", "drainage"),
    ("toilet waste road il varunnu, sewer pipe pottannu, durgandham, urgent", "sewage_issue", "urgent", "sewage"),
    # Pair 12
    ("rain vandal oda nirachuvazhukannu, oru muppattu adiyulla vellam road il", "drainage", "urgent", "drainage"),
    ("bathroom use cheyyaan pattunilla, toilet waste back up aayittu, sewer block", "sewage_issue", "urgent", "sewage"),
    # Pair 13
    ("stormwater outlet block aayittu, road muzhuvan vellathinadiyil, urgent clear venam", "drainage", "high", "drainage"),
    ("sewage pump thakarnu, tank nirachu, toilet waste overflow aakaan neram aayi", "sewage_issue", "urgent", "sewage"),
    # Pair 14
    ("oda channal gravel kondu block, mazha vandal veedu flood aakum", "drainage", "high", "drainage"),
    ("septic tank nirachu ozhukunu, lane il toilet water ozhukunu", "sewage_issue", "urgent", "sewage"),
    # Pair 15
    ("drain cover missing junction kazhijal, vehicle wheel veezhum, very dangerous", "drainage", "high", "drainage"),
    ("manhole open junction kazhijal, toilet waste smell, road safety issue", "sewage_issue", "high", "sewage"),

    # ── Malayalam contrastive pairs ──────────────────────────────────────
    # Pair 16
    ("ഒഴുക്ക് ചാൽ ബ്ലോക്ക്. മഴ വന്നാൽ ഞങ്ങളുടെ തെരുവ് വെള്ളക്കെട്ടാകും.", "drainage", "high", "drainage"),
    ("ടോയ്‌ലറ്റ് ദ്രാവകം ഒഴുകി, ഞങ്ങളുടെ ഇടവഴി ദുർഗന്ധം കൊണ്ട് ഉപയോഗ‌ശൂന്യം.", "sewage_issue", "urgent", "sewage"),
    # Pair 17
    ("മഴ ദിവസം ഓടചാൽ നിറഞ്ഞ് ഒഴുകി, ഞങ്ങളുടെ ഗ്രൗണ്ടിൽ വെള്ളം കയറി.", "drainage", "urgent", "drainage"),
    ("ടോയ്‌ലറ്റ് waste ഒഴുകി ഞങ്ങളുടെ ഗ്രൗണ്ടിൽ, ഗ്രൗണ്ട് ഫ്ലോർ ഫ്ലാറ്റ് ഉപയോഗ‌ശൂന്യം.", "sewage_issue", "urgent", "sewage"),
    # Pair 18
    ("stormwater channel school kazhijal block, rain vandal schoolyard flood aakum", "drainage", "high", "drainage"),
    ("toilet waste school kazhijal ola ozhukunu, കുട്ടികൾ ആരോഗ്യ ഭീഷണി നേരിടുന്നു", "sewage_issue", "urgent", "sewage"),
    # Pair 19
    ("ഓടചാൽ ഗ്രേറ്റ് ഇളകി, വാഹനം ആഴ്ന്ന് കിടക്കുന്നു, അപകടകരം", "drainage", "high", "drainage"),
    ("മാൻ‌ഹോൾ കവർ ഇല്ല, ടോയ്‌ലറ്റ് smell, road safety issue", "sewage_issue", "high", "sewage"),
    # Pair 20
    ("sinkhole forming, mazhavellam pipe pottannu, urgent check", "drainage", "urgent", "drainage"),
    ("sinkhole forming, sewer pipe pottannu, toilet waste underground nikunnu, urgent", "sewage_issue", "urgent", "sewage"),

    # ── Explicitly label-anchoring single samples ────────────────────────
    # (add keyword anchors that make the semantic domain unambiguous)
    ("The stormwater drain that takes rainwater from uphill is completely clogged. No toilet or sewage connection — purely a rain drainage issue.", "drainage", "high", "drainage"),
    ("The main sewer line carrying toilet waste and bathroom water is backed up. Nothing to do with rainwater — pure sewage blockage.", "sewage_issue", "urgent", "sewage"),
    ("Side drain meant for road surface water is overflowing. No smell of sewage, just rainwater flooding.", "drainage", "medium", "drainage"),
    ("Sewage smell is everywhere near our house. Sewer pipe leaking toilet waste, not rainwater.", "sewage_issue", "high", "sewage"),
    ("Channel blocked with leaves after rain. The channel carries only stormwater, not sewage.", "drainage", "medium", "drainage"),
    ("Manhole overflowing with solid sewage and toilet waste. Has nothing to do with rain.", "sewage_issue", "urgent", "sewage"),
    ("After heavy rain the road floods because the stormwater outlet to the nullah is blocked.", "drainage", "high", "drainage"),
    ("Sewer treatment plant discharged raw fecal matter into the canal. Water supply at risk.", "sewage_issue", "critical", "sewage"),
    ("Rainwater from the upslope hillside floods our lane because the culvert is too small.", "drainage", "high", "drainage"),
    ("Toilet waste and human excreta overflowing from the septic tank onto the road.", "sewage_issue", "urgent", "sewage"),
]


# ===========================================================================
# FIX 3: PRIORITY EMOTIONAL ANCHORS
#
# Problem: 33% of emotional-language complaints were inflated to urgent.
# The model conflates frustrated register with urgency level.
#
# These samples pair emotional wording with CORRECT (non-inflated) priority
# labels. Medium-severity issues expressed with all-caps/excessive punctuation
# must still map to medium, not urgent.
# ===========================================================================

PRIORITY_EMOTIONAL_ANCHORS: list[TrainingSample] = [
    # ── Street light: medium severity, emotional wording ─────────────────
    ("ABSOLUTELY DISGUSTING!! The street light in front of our house has been broken for WEEKS and nobody cares at all!!", "street_light", "medium", "electricity"),
    ("I am SO FRUSTRATED. The pole light at the bus stop is dead for the 5th time this month. FIX IT PERMANENTLY please!!", "street_light", "medium", "electricity"),
    ("This is UNACCEPTABLE!! Why does the corporation not repair the street light near the school?? It's been months!!!", "street_light", "medium", "electricity"),
    ("Three times I have complained about this broken street light. THREE TIMES. And still nothing. Disgusting service.", "street_light", "medium", "electricity"),
    ("No one is listening!!! The road light near junction has been off for 2 weeks. Is this how corporation works???", "street_light", "medium", "electricity"),
    ("veri frustrated!! light illa road kazhijal, oru year aayittu complain cheythu, oru response illakill", "street_light", "medium", "electricity"),
    ("ഇത് ആർക്കും ശ്രദ്ധ ഇല്ലേ??? Street light ഒരു മാസം ആയി broken. ഇനിയും കാക്കണോ???", "street_light", "medium", "electricity"),

    # ── Road damage: medium severity, emotional wording ──────────────────
    ("HOW MANY MORE ACCIDENTS BEFORE SOMEONE FIXES THE POTHOLE?? Already 3 bikes fell there!!", "road_damage", "high", "roads"),
    ("My patience has completely run out. The road in our lane is terrible but nobody in the corporation seems to care!!", "road_damage", "medium", "roads"),
    ("PATHETIC roads! Every rainy season same story. Year after year. Does the corporation even exist?!?", "road_damage", "medium", "roads"),
    ("I am beyond furious. The road damage near the school was reported 6 months ago. NOTHING DONE. Shameful.", "road_damage", "medium", "roads"),
    ("road muzhuvan kuzhi, corporation annu nokkan vannu oru action illakill, VERY BAD SERVICE", "road_damage", "medium", "roads"),
    ("ഈ road ഇനിയും ശരിയാക്കില്ലേ?? ഒരു കൊല്ലം ആയി complaint ഇടുന്നു!!", "road_damage", "medium", "roads"),

    # ── Garbage: medium severity, emotional wording ──────────────────────
    ("GARBAGE HAS NOT BEEN COLLECTED FOR 10 DAYS! The entire street smells horrible! Is this a civilised city or not??", "solid_waste", "medium", "sanitation"),
    ("This is absolutely shameful. Rotting garbage everywhere and no one from the corporation even acknowledges our calls!!", "solid_waste", "medium", "sanitation"),
    ("I am totally fed up!! The waste bin has been overflowing for a week. The smell is unbearable. Please act NOW!!!", "solid_waste", "medium", "sanitation"),
    ("garbage truck oru thadavayum varunnilla, oru week aayi, VERY FRUSTRATING!!!", "solid_waste", "medium", "sanitation"),
    ("ഓരോ ദിവസവും ഒരേ കാഴ്ച! Garbage overflowing!! ഇനി ആർക്ക് complaint ഇടണം??", "solid_waste", "medium", "sanitation"),

    # ── Water supply: medium severity, emotional wording ─────────────────
    ("Water has been off for 2 days and the corporation helpline just rings and rings!! HOW ARE WE SUPPOSED TO LIVE??", "water_supply", "high", "water"),
    ("ZERO water pressure for a week now. We have complained three times. Not a single response. Absolutely useless!!!", "water_supply", "medium", "water"),
    ("I'm completely at my wit's end. The water pipe outside is dripping and the meter is running. NOBODY CARES.", "water_supply", "medium", "water"),
    ("Water supply only 20 minutes every morning. HOW do you expect a family of 6 to manage?? Completely unacceptable!!", "water_supply", "medium", "water"),
    ("vellam karunna illakill oru vaaram, helpline ring cheythu no response, VERY UPSET", "water_supply", "high", "water"),

    # ── Drainage: medium severity, emotional wording ─────────────────────
    ("Every SINGLE rain the drain backs up and our road floods. We've been saying this for YEARS! Why won't you act??", "drainage", "high", "drainage"),
    ("I AM SO ANGRY. The drain cover near the junction is missing for 3 months. Someone will die before it's fixed!!", "drainage", "high", "drainage"),
    ("ഈ ഡ്രെയ്നേജ് ശരിയാക്കുന്നത് എന്നാണ്?? ഓരോ മഴ ദിവസവും ഒരേ കഥ! Completely fed up!!", "drainage", "high", "drainage"),
    ("drain block, rain vandal flood, oru 10 thadava complaint cheythu, ZERO RESPONSE, SO FRUSTRATED", "drainage", "high", "drainage"),

    # ── Illegal construction: medium severity, emotional wording ─────────
    ("Is the corporation BLIND?? There's an illegal building going up on public land and NOBODY is stopping it!!", "illegal_construction", "high", "planning"),
    ("I've been reporting this encroachment for 6 months. SIX MONTHS. The compound wall is still there. What is this??", "illegal_construction", "medium", "planning"),
    ("permit illatha construction 1 year aayittu, corporation oru action illakill, VERY DISGUSTING", "illegal_construction", "high", "planning"),

    # ── Sewage: high severity even with calm wording (control group) ──────
    # (these confirm that calm+urgent correctly maps to urgent)
    ("Sewage overflowing from manhole onto road near school. Children present. Immediate action required.", "sewage_issue", "urgent", "sewage"),
    ("Toilet waste backing up into ground floor flat. Bathrooms unusable. This is an urgent health emergency.", "sewage_issue", "urgent", "sewage"),

    # ── Cross-category emotional samples with low/medium priority ────────
    ("DISGUSTING!! The park bench is broken for a year! Corporation doesn't care about citizens at all!!", "no_category", "low", "none"),
    ("When will the footpath near our house be repaired?? It's been like this for so long. Very disappointing service.", "road_damage", "medium", "roads"),
    ("I am extremely upset about the state of the public toilet in our ward. Complete lack of cleanliness. Shame on corporation!!", "solid_waste", "medium", "sanitation"),
    ("WHY IS THERE NO STREET LIGHT FOR 1 KM STRETCH ON OUR ROAD?? This has been the case for 3 months. Totally irresponsible!!", "street_light", "medium", "electricity"),
    ("completely fed up!! tree hanging over road, been telling for months, no action. What if it falls??", "tree_fall", "medium", "parks"),
    ("It is really SHAMEFUL that the road in our colony has not been tarred for 5 years. Total neglect by the authorities!!", "road_damage", "medium", "roads"),
    ("ഇതൊന്നും ശരിയാകില്ലേ?? Street light ഒരു month broken, corporation ഒന്ന് respond ചെയ്‌തില്ല, very bad!!", "street_light", "medium", "electricity"),
    ("totally frustrated with municipal service!! drain not cleaned before monsoon, 3rd year same problem!!", "drainage", "medium", "drainage"),
    ("why nobody listens!! garbage not collected 1 week, bin overflowing, smell very bad, response zero!!", "solid_waste", "medium", "sanitation"),
    ("HOW LONG must we wait?? The pothole at junction has been there for 4 months!! Bikes keep falling!!", "road_damage", "high", "roads"),
]


# ===========================================================================
# FIX 4: EXPANDED LOCATION ALIASES
#
# The current TVM_LOCATIONS list only has 43 single ward/area names.
# The TransformerEngine encodes these for cosine-similarity location matching.
# When a complaint uses compound descriptions like "Pattom junction SUT hospital"
# the single-word embedding of "Pattom" is too far in embedding space to match.
#
# These compound phrases are real Thiruvananthapuram civic landmarks used in
# actual complaint descriptions.  Imported by train_transformer.py and appended
# to TVM_LOCATIONS before landmark pre-encoding.
# ===========================================================================

TVM_LOCATION_ALIASES: list[str] = [
    # Pattom area
    "Pattom junction",
    "Pattom junction SUT hospital",
    "Pattom SUT Medical College",
    "Pattom palace road",
    "Pattom main road",
    "near Pattom overbridge",

    # Kazhakkoottam / Technopark
    "Kazhakkoottam Technopark bypass",
    "Kazhakkoottam junction Technopark",
    "Technopark phase 1 main gate",
    "Technopark phase 3 bypass",
    "Kazhakkoottam railway station",
    "Kazhakkoottam main road",

    # Karamana area
    "Karamana bridge junction",
    "Karamana river bridge",
    "Karamana Kaliyikkavila road",
    "Karamana Attakulangara road",
    "near Karamana police station",

    # Medical College area
    "Medical College hospital campus",
    "Medical College junction",
    "Government Medical College Thiruvananthapuram",
    "Medical College children's hospital",
    "near SAT hospital Medical College",

    # East Fort / Padmanabhaswamy
    "East Fort Padmanabhaswamy temple",
    "East Fort junction",
    "East Fort market",
    "near Padmanabhaswamy temple",
    "Sree Padmanabhaswamy temple road",

    # Kowdiar area
    "Kowdiar palace junction",
    "Kowdiar main road",
    "Kowdiar Nanthancode junction",
    "near Raj Bhavan Kowdiar",

    # Palayam area
    "Palayam market junction",
    "Palayam bus station",
    "Palayam Connemara market",
    "Palayam Unity Church junction",

    # Secretariat / Central
    "Secretariat Thiruvananthapuram",
    "Secretariat junction",
    "near Secretariat main gate",
    "Secretariat Statue junction",
    "Statue junction Thiruvananthapuram",

    # Nanthancode area
    "Nanthancode Vellayambalam junction",
    "Nanthancode main road",
    "Vellayambalam junction",
    "near AG office Vellayambalam",

    # Kesavadasapuram area
    "Kesavadasapuram junction",
    "Kesavadasapuram main road",
    "near Kesavadasapuram post office",

    # Other major junctions and landmarks
    "Thampanoor railway station road",
    "Thampanoor bus stand",
    "Central railway station Thiruvananthapuram",
    "Sreekaryam Kariavattom road",
    "Sreekaryam University campus",
    "Kariavattom University bypass",
    "Ulloor Akkulam road",
    "Ulloor Medical College road junction",
    "Kochulloor Ambalamukku junction",
    "Ambalamukku Poojappura road",
    "Poojappura jail junction",
    "Vizhinjam fishing harbour",
    "Vizhinjam port area",
    "Shanghumugham beach road",
    "Vattiyoorkavu bypass road",
    "Peroorkada junction",
    "Vazhuthacaud Kowdiar road",
    "Vazhuthacaud overbridge",
    "Jagathy bus stop junction",
    "Chalai market road",
    "Thycaud hospital junction",
    "Beemapalli mosque junction",
    "Attukal Kunnukuzhy road",
    "Mudavanmugal junction",
    "Pangode military station road",
    "Thirumala hilltop road",
    "Sasthamangalam bypass",
    "Mannanthala Chackai road",
    "Muttada junction",
    "Kachani Sreekariyam road",
    "Thiruvallam backwaters road",
]

# Merged extended location list for use by train_transformer.py
TVM_LOCATIONS_EXTENDED: list[str] = TVM_LOCATIONS + TVM_LOCATION_ALIASES
