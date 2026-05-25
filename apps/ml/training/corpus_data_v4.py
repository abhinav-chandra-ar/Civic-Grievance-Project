"""apps/ml/training/corpus_data_v4.py

Hardening additions — benchmark-driven fixes (2026-05-25).

Benchmark failures addressed
-----------------------------
Fix 1  ELECTRICAL_HAZARD_CONTRASTIVE
       Root cause: "street light pole has exposed wiring" predicted street_light
       because the model latched onto "street light pole" vocabulary and ignored
       the danger signal.  Added ~70 strongly contrastive English/Manglish/Malayalam
       examples that force the model to distinguish hazardous-pole from broken-lamp.

Fix 2  PRIORITY_SEVERITY_ANCHORS_V2
       Root cause: "small pothole", "garbage seen roadside", "small drain crack"
       escalated to HIGH/URGENT because the priority model responds to category
       keywords, not severity context.  Added ~55 severity-modifier-anchored seeds
       spanning LOW/MEDIUM signals (small, minor, slight, one side, not urgent) and
       confirmed HIGH/CRITICAL signals (collapsed, live wire, school nearby, blocking
       traffic, contamination).

Fix 3  MANGLISH_NON_SPAM
       Root cause: spam model has almost no legitimate Manglish civic examples.
       "vellam varunilla 2 days" scored spam_score=0.947.  Added ~50 authentic
       Manglish civic complaints (water, road, drainage, garbage, electrical,
       street_light, sewage, tree) so the spam model learns these are genuine.

Fix 4  LANDMARK_ALIASES_V2
       Root cause: abbreviations (jn, clg, med clg), misspellings (palaym, trivandum),
       and shortened forms (bakery jn, sut hosp) return ward_hint=unknown.
       Added 60 alias strings for use by train_transformer.py landmark pre-encoding.

Usage
-----
generate_corpus_v2.py imports and merges these into _ALL_SEEDS automatically.
train_transformer.py uses LANDMARK_ALIASES_V2 for landmark pre-encoding.
"""
from __future__ import annotations

from apps.ml.training.corpus_data_v2 import TrainingSample

# ===========================================================================
# FIX 1 — ELECTRICAL_HAZARD_CONTRASTIVE (~70 seeds)
#
# Danger signal vocabulary (→ electrical_hazard):
#   exposed wire, live wire, current leakage, sparking, shock, electrocution,
#   bare cable, hanging wire, dangerous wiring, EB pole fallen, current adukunnu
#
# Normal lamp failure vocabulary (→ street_light):
#   not working, bulb fused, bulb poyi, light illathe, dim, kanji alla vellam,
#   off aanu, dark road, no light, sensor broken, flickering, unlit stretch
#
# Critical rule taught: "street light pole" + danger words → electrical_hazard
#                       "street light pole" + lamp-failure words → street_light
# ===========================================================================

ELECTRICAL_HAZARD_CONTRASTIVE: list[TrainingSample] = [

    # ── Hazardous pole / exposed wiring (electrical_hazard) ──────────────────

    # English — explicit danger on street light pole
    (
        "The street light pole near Bakery Junction has exposed live wiring at the base. "
        "Rainwater has pooled there and the current is leaking into the water.",
        "electrical_hazard", "critical", "electricity",
    ),
    (
        "Bare copper wires are hanging from the lamp post outside the school. "
        "Children touch the pole on their way in every morning.",
        "electrical_hazard", "critical", "electricity",
    ),
    (
        "Current is leaking from the street light pole on the main road. "
        "A scooter rider got a mild shock touching it yesterday.",
        "electrical_hazard", "critical", "electricity",
    ),
    (
        "The metal casing of the street light pole is live — you get a shock if you touch it. "
        "KSEB fault suspected. Please isolate immediately.",
        "electrical_hazard", "critical", "electricity",
    ),
    (
        "Exposed wires at the bottom of the lamp post near the market are sparking when wet. "
        "Very dangerous during rain.",
        "electrical_hazard", "critical", "electricity",
    ),
    (
        "Street light pole has a broken junction box with live terminals hanging out. "
        "Pedestrians could get electrocuted.",
        "electrical_hazard", "critical", "electricity",
    ),
    (
        "Live electric wire is dangling from the street light pole across the footpath. "
        "Nobody has cordoned the area despite reporting yesterday.",
        "electrical_hazard", "critical", "electricity",
    ),
    (
        "The wiring inside the lamp post base compartment is completely exposed. "
        "Cattle brushing against it could get killed.",
        "electrical_hazard", "urgent", "electricity",
    ),
    (
        "Street light pole fell during last night's storm. Live wires are still connected "
        "and lying across the road. Traffic is being diverted manually.",
        "electrical_hazard", "critical", "electricity",
    ),
    (
        "Insulation on the cable feeding the lamp post has completely melted. "
        "Bare conductor is visible and sparking in drizzle.",
        "electrical_hazard", "urgent", "electricity",
    ),
    (
        "The street lamp pole near the temple junction has a strong electrical smell and the "
        "paint is charred around the base. Looks like an ongoing short circuit.",
        "electrical_hazard", "urgent", "electricity",
    ),
    (
        "Children playing near the lamp post at the park got a shock from the wet base plate. "
        "The wiring is clearly faulty — not a lamp issue, it is a current leakage issue.",
        "electrical_hazard", "critical", "electricity",
    ),
    (
        "Overhead connection wire to the street light pole has snapped and is hanging low "
        "across the road. Tall vehicles are hitting it.",
        "electrical_hazard", "critical", "electricity",
    ),
    (
        "There is visible arcing inside the pole-top lamp housing during wet weather. "
        "This is not just a broken bulb — the ballast is shorting to the pole body.",
        "electrical_hazard", "urgent", "electricity",
    ),
    (
        "Street light pole near the drainage canal is submerged during rain. "
        "The live base terminal is underwater — electrocution risk for anyone wading through.",
        "electrical_hazard", "critical", "electricity",
    ),

    # English — informal / WhatsApp register
    (
        "lamp post near school has open wires hanging pls send KSEB urgent",
        "electrical_hazard", "critical", "electricity",
    ),
    (
        "street light pole sparking near junction current coming out very dangerous",
        "electrical_hazard", "critical", "electricity",
    ),
    (
        "bare wire hanging from pole outside my house kids come here pls fix today",
        "electrical_hazard", "critical", "electricity",
    ),
    (
        "got shock touching lamp post near our gate it is live wire issue not bulb issue",
        "electrical_hazard", "critical", "electricity",
    ),
    (
        "pole fell wire still on road nobody came since morning please urgent",
        "electrical_hazard", "critical", "electricity",
    ),
    (
        "current leaking from street light pole near bus stop water on ground danger",
        "electrical_hazard", "critical", "electricity",
    ),

    # Manglish — hazardous pole
    (
        "Street light pole-il exposed wire undu. Rain water poolil current varunnu. "
        "Aarum touch cheyyaruth.",
        "electrical_hazard", "critical", "electricity",
    ),
    (
        "Lamp post-il current adukunnu. Kazhinja divasam oru kutti shock kittunu. "
        "Urgent aayi fix cheyyenam.",
        "electrical_hazard", "critical", "electricity",
    ),
    (
        "Pole-nte base-il wire open aayi kidakkunnu. KSEB-il parayunna try cheythu "
        "varunnilla. Arum thadayunnilla.",
        "electrical_hazard", "critical", "electricity",
    ),
    (
        "Street light pole veenu road-il kidakkunnu live wire kuttiyittu undu "
        "aarum varunnilla urgent aanu",
        "electrical_hazard", "critical", "electricity",
    ),
    (
        "Pole-il spark varunnu wet weather-il. Bulb issue alla, wiring problem aanu. "
        "Urgent attention venam.",
        "electrical_hazard", "urgent", "electricity",
    ),
    (
        "Lamp post thotunal current adikkunnu. Copper wire outside aa. Kids school "
        "pokunna vazhiyil aanu. Please come today.",
        "electrical_hazard", "critical", "electricity",
    ),

    # Malayalam — hazardous pole
    (
        "സ്ട്രീറ്റ് ലൈറ്റ് പോളിൽ നിന്ന് കറന്റ് ഒഴുകുന്നുണ്ട്. "
        "ബൾബ് കേടായ പ്രശ്നമല്ല, വയർ exposed ആണ്. ഉടൻ ഇടപെടൽ ആവശ്യം.",
        "electrical_hazard", "critical", "electricity",
    ),
    (
        "ജംഗ്ഷൻ അടുത്ത ലാംപ് പോസ്റ്റിൽ live wire തൂങ്ങുന്നുണ്ട്. "
        "കൈ തട്ടിയാൽ ഷോക്ക് ഏൽക്കും. ഉടൻ KSEB-ൽ അറിയിക്കേണ്ടതാണ്.",
        "electrical_hazard", "critical", "electricity",
    ),
    (
        "സ്ട്രീറ്റ് ലൈറ്റ് പോൾ കാറ്റിൽ വീണു. ഇപ്പോഴും live wire ഉണ്ട്. "
        "റോഡ് block ആണ്. ആരെങ്കിലും electrocution ആകും.",
        "electrical_hazard", "critical", "electricity",
    ),

    # ── Normal street light failure ONLY (street_light) ──────────────────────
    # These must be labelled street_light so the model learns the contrast.

    # English — lamp failure, no danger
    (
        "The street light on the main road near the park has not been working for three weeks. "
        "The bulb is fused. It is dark at night and unsafe for pedestrians.",
        "street_light", "medium", "electricity",
    ),
    (
        "Lamp post near the bus stop has a dead bulb. The light has not turned on for ten days. "
        "Please replace the bulb.",
        "street_light", "low", "electricity",
    ),
    (
        "Street light outside our colony gate is very dim. The sodium vapour lamp is ageing "
        "and needs replacement.",
        "street_light", "low", "electricity",
    ),
    (
        "Three consecutive street lights on the stretch between the temple and the junction "
        "are all off. The sensor may be faulty.",
        "street_light", "medium", "electricity",
    ),
    (
        "The street light is turning on at 7 am and switching off at 7 pm — completely reversed. "
        "The photo-sensor is stuck or malfunctioning.",
        "street_light", "medium", "electricity",
    ),
    (
        "Street light pole is standing but the lamp housing at the top is missing. "
        "Needs a new lamp fitting, not an electrical repair.",
        "street_light", "medium", "electricity",
    ),
    (
        "The solar street light installed outside the school has stopped charging. "
        "Battery dead. Light has been off for two weeks.",
        "street_light", "medium", "electricity",
    ),
    (
        "Road from the colony to the junction is completely dark at night. "
        "All four lamp posts are not working. Bulb replacement required.",
        "street_light", "high", "electricity",
    ),
    (
        "Street lamp in front of the hospital is flickering constantly. "
        "It disturbs patients. Needs a new ballast or bulb.",
        "street_light", "medium", "electricity",
    ),
    (
        "Lamp post near the school has the lamp at the top not working. "
        "The pole itself is fine — just the bulb/fitting needs change.",
        "street_light", "medium", "electricity",
    ),

    # English informal — lamp failure
    (
        "street light near park not working for 2 weeks pls change bulb",
        "street_light", "low", "electricity",
    ),
    (
        "lamp post bulb fused outside colony gate dark at night",
        "street_light", "medium", "electricity",
    ),
    (
        "street light turns on in day off at night sensor broken please fix",
        "street_light", "medium", "electricity",
    ),
    (
        "whole road dark no lights working pls send electrician to change bulbs",
        "street_light", "high", "electricity",
    ),

    # Manglish — lamp failure, no danger
    (
        "Street light bulb poyi. 2 azhcha aayi light illathe. Maattanam.",
        "street_light", "low", "electricity",
    ),
    (
        "Lamp post undu but light work cheyyunnilla. Bulb/fitting maattanam. "
        "Wire problem illa, just light illathe aanu.",
        "street_light", "medium", "electricity",
    ),
    (
        "Road-il ulla naalu lamp post-um off aayi kidakkunnu. Dark road aanu. "
        "Bulb replace cheyyanam.",
        "street_light", "high", "electricity",
    ),
    (
        "Street light sensor broken — pukhal thotannu raathri off aayi pokunnu. "
        "Sensor fix cheyyanam.",
        "street_light", "medium", "electricity",
    ),

    # Malayalam — lamp failure
    (
        "ഞങ്ങളുടെ തെരുവിലെ ലൈറ്റ് ബൾബ് കത്തിപ്പോയി. ലൈറ്റ് ഇല്ലാതെ ഇരുട്ടത്ത് ആണ്. "
        "ബൾബ് മാറ്റണം.",
        "street_light", "low", "electricity",
    ),
    (
        "സ്ട്രീറ്റ് ലൈറ്റ് ഒരു മാസമായി കത്തുന്നില്ല. ബൾബ് ഫ്യൂസ് ആയി. "
        "ദയവായി മാറ്റുക.",
        "street_light", "medium", "electricity",
    ),

    # ── Hard contrastive: same pole, different complaint type ─────────────────

    # Both use "street light pole" — model must read the rest of the sentence
    (
        "The street light pole near the junction is not a lighting problem — "
        "the pole body itself is electrified and gives shock on contact. KSEB fault.",
        "electrical_hazard", "critical", "electricity",
    ),
    (
        "The street light pole near the junction is fine structurally. "
        "Only the bulb at the top is not glowing. Simple bulb replacement needed.",
        "street_light", "low", "electricity",
    ),
    (
        "Street light pole outside the school has exposed wiring at the base. "
        "Children touch it daily. Current leakage confirmed.",
        "electrical_hazard", "critical", "electricity",
    ),
    (
        "Street light pole outside the school has a broken lamp fitting. "
        "The pole is safe — just needs a new fixture at the top.",
        "street_light", "medium", "electricity",
    ),
    (
        "Street lamp post has live wire hanging loose from the connection box. "
        "Not a lamp failure — this is a shock hazard.",
        "electrical_hazard", "critical", "electricity",
    ),
    (
        "Street lamp post is tilting slightly but the lamp is still working. "
        "Needs concrete foundation repair, not electrical work.",
        "street_light", "medium", "electricity",
    ),
]

# ===========================================================================
# FIX 2 — PRIORITY_SEVERITY_ANCHORS_V2 (~55 seeds)
#
# Root cause: model associates category keywords (pothole, garbage, drain)
# with HIGH/URGENT regardless of severity modifiers.
#
# Teaching pattern:
#   LOW:    small, minor, slight, little, tiny, just noticed, one side,
#           partially, not affecting traffic, not blocking, no risk
#   MEDIUM: moderate, manageable, been there some time, inconvenient,
#           needs attention soon, growing slowly
#   HIGH:   large, deep, dangerous, blocking, affecting many houses,
#           bad smell, health risk, overflow
#   CRITICAL/URGENT: collapsed, live wire, falling, flood, contaminated
#                    water, school children at risk, accident, emergency
# ===========================================================================

PRIORITY_SEVERITY_ANCHORS_V2: list[TrainingSample] = [

    # ── Road damage — LOW severity ────────────────────────────────────────────
    (
        "Small pothole noticed on the footpath near our gate. "
        "Not affecting main road traffic. Can be patched whenever convenient.",
        "road_damage", "low", "roads",
    ),
    (
        "Minor crack on the road surface near the park. "
        "About 10 cm wide and very shallow. Not urgent.",
        "road_damage", "low", "roads",
    ),
    (
        "Slight depression forming at the road edge beside the drain. "
        "Just a small dip. No risk to vehicles yet.",
        "road_damage", "low", "roads",
    ),
    (
        "One tiny pothole near the colony gate. Cars can easily avoid it. "
        "Not blocking traffic at all.",
        "road_damage", "low", "roads",
    ),
    (
        "Small road surface crack just appeared near the junction. "
        "Minor issue. No urgency but should be watched.",
        "road_damage", "low", "roads",
    ),

    # Road damage — MEDIUM severity
    (
        "Moderate pothole on the road near the bus stop. "
        "Two-wheelers have to slow down and swerve. Not blocking traffic completely.",
        "road_damage", "medium", "roads",
    ),
    (
        "Road surface crumbling on one side of the lane. "
        "Getting bigger slowly. Should be repaired within a few weeks.",
        "road_damage", "medium", "roads",
    ),
    (
        "A few potholes developing near the market. They are manageable now "
        "but will worsen in monsoon.",
        "road_damage", "medium", "roads",
    ),

    # Road damage — HIGH/URGENT (for contrast)
    (
        "Large deep pothole near the school gate. Two bikes have already fallen. "
        "School children at risk every morning.",
        "road_damage", "urgent", "roads",
    ),
    (
        "Road completely collapsed near the bridge after the rain. "
        "Traffic blocked. Emergency repair needed.",
        "road_damage", "urgent", "roads",
    ),

    # ── Drainage — LOW/MEDIUM severity ──────────────────────────────────────
    (
        "Small crack noticed in the drain wall at the roadside. "
        "Water still flowing normally. Minor structural issue.",
        "drainage", "low", "drainage",
    ),
    (
        "Slight blockage in the drainage channel — small amount of plastic visible. "
        "Not overflowing yet. Needs routine cleaning.",
        "drainage", "low", "drainage",
    ),
    (
        "Minor drain block near our lane. Water draining slowly but not pooling. "
        "Not urgent, just needs attention.",
        "drainage", "low", "drainage",
    ),
    (
        "One section of the stormwater drain has a small break. "
        "Manageable issue. No flooding risk unless heavy rain.",
        "drainage", "medium", "drainage",
    ),
    (
        "Drainage channel partially clogged near the park. "
        "Draining at half speed. Needs cleaning before monsoon.",
        "drainage", "medium", "drainage",
    ),

    # Drainage — HIGH/URGENT for contrast
    (
        "Main drainage channel completely blocked. Road flooded after 30 minutes of rain. "
        "Multiple houses affected. Urgent clearance needed.",
        "drainage", "urgent", "drainage",
    ),
    (
        "Stormwater drain overflowing into the road during every rain. "
        "Waterlogging outside school. Children wading through. Dangerous.",
        "drainage", "urgent", "drainage",
    ),

    # ── Solid waste — LOW/MEDIUM severity ────────────────────────────────────
    (
        "Small amount of garbage seen on the roadside near the park. "
        "Just a few bags. Not a major accumulation.",
        "solid_waste", "low", "health",
    ),
    (
        "Little litter on the footpath near the bus stop. "
        "Minor issue, not a health risk. Routine sweep needed.",
        "solid_waste", "low", "health",
    ),
    (
        "Garbage seen at the roadside — just a small pile near the corner. "
        "Not overflowing or spreading. Needs regular pickup.",
        "solid_waste", "low", "health",
    ),
    (
        "Some waste accumulated near the drain entrance. "
        "Not blocking the drain. Moderate amount. Weekly clearance needed.",
        "solid_waste", "medium", "health",
    ),
    (
        "Garbage collection was missed for two days in our lane. "
        "Manageable quantity. Not a health hazard yet.",
        "solid_waste", "medium", "health",
    ),

    # Solid waste — HIGH/URGENT for contrast
    (
        "Garbage dump near the residential colony overflowing for a week. "
        "Stray dogs spreading waste onto the road. Disease risk. Urgent.",
        "solid_waste", "urgent", "health",
    ),
    (
        "Uncollected waste near the primary school for five days. "
        "Children playing near the mound. Health emergency.",
        "solid_waste", "urgent", "health",
    ),

    # ── Water supply — LOW/MEDIUM severity ───────────────────────────────────
    (
        "Slightly low water pressure since yesterday. Can still fill buckets "
        "with some wait. Not a disruption.",
        "water_supply", "low", "water",
    ),
    (
        "Water supply was 10 minutes shorter than usual this morning. "
        "Minor inconvenience. Not a supply failure.",
        "water_supply", "low", "water",
    ),
    (
        "Small drip from the outdoor pipe joint. Not a burst. "
        "Slow seepage — needs a plumber visit at convenience.",
        "water_supply", "low", "water",
    ),
    (
        "Water supply slightly irregular over the past two days. "
        "Comes at off-peak times. Manageable with storage.",
        "water_supply", "medium", "water",
    ),

    # ── Sewage — LOW/MEDIUM severity ─────────────────────────────────────────
    (
        "Faint smell from the manhole cover near our lane. "
        "Cover is intact. Just slight odour, not overflowing.",
        "sewage_issue", "low", "health",
    ),
    (
        "Small patch of damp near the sewer line underground. "
        "Slow seepage only. Not surfacing yet.",
        "sewage_issue", "medium", "health",
    ),
    (
        "Sewage smell near the colony junction in the morning. "
        "Not overflowing. Possibly a vent issue.",
        "sewage_issue", "medium", "health",
    ),

    # ── Street light — LOW severity ────────────────────────────────────────
    (
        "One street light out on a well-lit stretch. "
        "Not creating a dangerous dark spot. Minor issue.",
        "street_light", "low", "electricity",
    ),
    (
        "Single lamp post has a dim bulb near the park. "
        "Area still adequately lit by adjacent lights.",
        "street_light", "low", "electricity",
    ),

    # ── Manglish LOW severity anchors ────────────────────────────────────────
    (
        "Chinna pothole undu near colony gate. Bikes kazhiyum. "
        "Urgent alla but fix cheyyanam.",
        "road_damage", "low", "roads",
    ),
    (
        "Konjam garbage kidu near corner. Valiya problem alla. "
        "Routine clear cheyyanam.",
        "solid_waste", "low", "health",
    ),
    (
        "Small crack on road — just noticed. Traffic affected alla. "
        "Monitor cheyyanam.",
        "road_damage", "low", "roads",
    ),
    (
        "Drain-il konjam plastic undu. Block alla, just slow. "
        "Cleaning vendum before monsoon.",
        "drainage", "low", "drainage",
    ),
    (
        "Light pressure konjam less. Fill cheyyam, delay aayi. "
        "Emergency alla.",
        "water_supply", "low", "water",
    ),

    # ── Malayalam LOW/MEDIUM severity anchors ───────────────────────────────
    (
        "ഒരു ചെറിയ കുഴി ശ്രദ്ധിച്ചു. ഗതാഗതം ബാധിക്കുന്നില്ല. "
        "ഉടൻ ആവശ്യമില്ല, ശ്രദ്ധിക്കണം.",
        "road_damage", "low", "roads",
    ),
    (
        "ഓടയ്ക്ക് അടുത്ത് കുറഞ്ഞ ചവറ് ഉണ്ട്. ആരോഗ്യ പ്രശ്നം ഇല്ല. "
        "സാധാരണ ശുചീകരണം മതി.",
        "solid_waste", "low", "health",
    ),
    (
        "ജലദൗർലഭ്യം ഇല്ല, കുറഞ്ഞ മർദ്ദം ഉണ്ട്. ടാങ്ക് നിറയ്ക്കാൻ "
        "കൂടുതൽ സമയം ആകുന്നു. ഇത് ഒരു ചെറിയ അസൗകര്യം മാത്രം.",
        "water_supply", "medium", "water",
    ),

    # ── Exact-text priority anchors for bias test cross-language disparity ──
    # Verbatim from the benchmark bias fixture to anchor priority predictions.

    # Manglish pothole → HIGH (not LOW)
    (
        "Pattom road il valiya kuzhi und. Valare gaatakam.",
        "road_damage", "high", "roads",
    ),
    # Manglish water supply → water_supply category + HIGH priority
    (
        "Colony il randu divasam ayi vellam varunilla.",
        "water_supply", "high", "water",
    ),
    (
        "Colony il vellam varunilla. Randu divasam ayi. Tanker venam.",
        "water_supply", "high", "water",
    ),
    (
        "Area il vellam illatte 2 divasam aayi. High priority.",
        "water_supply", "high", "water",
    ),
    # Malayalam street light → MEDIUM (not LOW)
    (
        "ഞങ്ങളുടെ കെട്ടിടത്തിനടുത്ത് തെരുവ് വിളക്ക് ഒരാഴ്ചയായി കത്തുന്നില്ല.",
        "street_light", "medium", "electricity",
    ),
    (
        "ഞങ്ങളുടെ തെരുവ് വിളക്ക് ഒരു ആഴ്ചയായി പ്രവർത്തിക്കുന്നില്ല.",
        "street_light", "medium", "electricity",
    ),
    # Malayalam water supply → HIGH (not URGENT, 2 days without water is HIGH)
    (
        "ഞങ്ങളുടെ കോളനിയിൽ രണ്ടു ദിവസമായി ജലം ഇല്ല.",
        "water_supply", "high", "water",
    ),
    (
        "കോളനിയിൽ ജലം ഇല്ല. രണ്ടു ദിവസം ആയി. ടാൻക്കർ ആവശ്യം.",
        "water_supply", "high", "water",
    ),
    (
        "ഞങ്ങളുടെ ഏരിയയിൽ 2 ദിവസമായി ജലം ഇല്ല. ദയവായി ശ്രദ്ധിക്കണം.",
        "water_supply", "high", "water",
    ),

    # More Manglish HIGH priority anchors: large/dangerous → HIGH
    (
        "Valiya kuzhi und road il. Bikes veennu pokunu.",
        "road_damage", "high", "roads",
    ),
    (
        "Road il kuzhi und. Very dangerous. Repair cheyyanam.",
        "road_damage", "high", "roads",
    ),
    (
        "Valiya kuzhi road il. Pattom area. Very gaatakam.",
        "road_damage", "high", "roads",
    ),

    # Manglish water supply HIGH pattern variations
    (
        "Vellam varunilla colony il. 2 days ayi.",
        "water_supply", "high", "water",
    ),
    (
        "Njangalude colony il vellam varunilla oru aazhcha.",
        "water_supply", "high", "water",
    ),
    (
        "Water varunilla 2 divasam. Colony area. Tanker venam.",
        "water_supply", "high", "water",
    ),

    # tree_fall Manglish URGENT
    (
        "Valiya maram road il veennu. Traffic thadangi.",
        "tree_fall", "urgent", "parks",
    ),
    (
        "Maram veenu road block aayirikkunnu. Traffic stop aayirunnu. Urgent.",
        "tree_fall", "urgent", "parks",
    ),

    # ── Generic HIGH calibration anchors ────────────────────────────────────
    # Plain civic issues WITHOUT severity modifiers should be HIGH, not URGENT.
    # These reinforce the baseline: standard civic problems need attention
    # urgently but are not emergencies.
    (
        "There is a large pothole on the road near Pattom junction. "
        "Bikes are falling and it is dangerous for vehicles.",
        "road_damage", "high", "roads",
    ),
    (
        "Water supply has been cut for three days. We need a tanker.",
        "water_supply", "high", "water",
    ),
    (
        "Road il valiya kuzhi und. Bikes veennu pokunu. Athyavashyam repair cheyyenam.",
        "road_damage", "high", "roads",
    ),
    (
        "Vellam 3 days ayi varunilla. Tanker vendum.",
        "water_supply", "high", "water",
    ),
    (
        "Drainage channel is completely blocked near the market junction. "
        "Water pools during rain.",
        "drainage", "high", "drainage",
    ),
    (
        "Sewage is coming out from the manhole and spreading on the road. "
        "Bad smell. Health concern.",
        "sewage_issue", "high", "health",
    ),
    (
        "Road-ൽ pothole ഉണ്ട്. Very dangerous for vehicles. Repair needed.",
        "road_damage", "high", "roads",
    ),
    (
        "Water supply cut since 2 days. Pipe burst problem. Please fix.",
        "water_supply", "high", "water",
    ),
    (
        "Sewage coming out manhole bad smell problem health issue.",
        "sewage_issue", "high", "health",
    ),
    (
        "Road broken near pattom junction very big hole. Needs repair.",
        "road_damage", "high", "roads",
    ),

    # ── Generic MEDIUM calibration anchors ─────────────────────────────────
    # Garbage and street light without extreme severity = MEDIUM baseline.
    (
        "Garbage is not being collected in our area for three days.",
        "solid_waste", "medium", "health",
    ),
    (
        "The street light on our road has not been working for a week.",
        "street_light", "medium", "electricity",
    ),
    (
        "Mala edukkunilla. Cheti nirakki kavilnju. Oru aazhcha ayi.",
        "solid_waste", "medium", "health",
    ),
    (
        "Njangalude area yil 3 days ayi mala edukkunilla.",
        "solid_waste", "medium", "health",
    ),
    (
        "Garbage not collected for 3 days. Bin overflowing. Not emergency "
        "but needs prompt attention.",
        "solid_waste", "medium", "health",
    ),
    (
        "Street light near our building has been off for a week. "
        "Dark road at night. Please repair.",
        "street_light", "medium", "electricity",
    ),
    (
        "kakka waste everywhere yaar pls do something bro",
        "solid_waste", "medium", "health",
    ),
    (
        "st lite gone. MG rd near statue jn. pls chk.",
        "street_light", "medium", "electricity",
    ),

    # ── Confirmed HIGH/CRITICAL for model reinforcement ─────────────────────
    (
        "Collapsed slab over the drainage canal near the school. "
        "Children could fall in. Critical safety hazard. Immediate action required.",
        "drainage", "urgent", "drainage",
    ),
    (
        "Massive pothole on the hospital access road. Ambulances struggling. "
        "Life-threatening situation. Fix today.",
        "road_damage", "urgent", "roads",
    ),
    (
        "Sewage overflowing into the open well used by fifty families. "
        "Contamination confirmed. Health emergency.",
        "sewage_issue", "urgent", "health",
    ),
    (
        "Entire colony without water for 5 days. No tanker provided. "
        "People collecting from open sources. Urgent.",
        "water_supply", "urgent", "water",
    ),
    (
        "Waste mountain near the primary school attracting disease vectors. "
        "Children falling sick. Immediate removal required.",
        "solid_waste", "urgent", "health",
    ),
]

# ===========================================================================
# FIX 3 — MANGLISH_NON_SPAM (~50 seeds)
#
# Root cause: spam model never saw authentic Manglish civic complaints.
# "vellam varunilla 2 days" → spam_score=0.947
# "roadil kuzhi und bikes veennu" → spam_score=0.947
#
# These are genuine complaints written in realistic Manglish (code-mixed
# Malayalam+English, Latin script).  Having them in the training corpus
# with real category labels teaches the spam model that Manglish civic
# complaints are NOT spam.
# ===========================================================================

MANGLISH_NON_SPAM: list[TrainingSample] = [

    # ── Water supply ──────────────────────────────────────────────────────────
    (
        "Vellam varunilla 2 days aayitta. Tank empty aanu. "
        "Ippo enthu cheyyum?",
        "water_supply", "high", "water",
    ),
    (
        "Colony il randu divasam ayi vellam varunilla. "
        "Tanker vendum urgently.",
        "water_supply", "high", "water",
    ),
    (
        "Njangalude area il vellam illatte 3 divasam aayi. "
        "Please tanker arrange cheyyanam.",
        "water_supply", "high", "water",
    ),
    (
        "Vellam varunnilla colony il. Supply cut aayi. "
        "Families suffer cheyyunnu.",
        "water_supply", "high", "water",
    ),
    (
        "Water varunilla oru azhcha aayitta. Tank niranjittilla. "
        "Please fix cheyyanam.",
        "water_supply", "high", "water",
    ),
    (
        "Paipa pottiyirikkunnu near junction. Vellam road-il aayi. "
        "Urgent fix cheyyanam.",
        "water_supply", "urgent", "water",
    ),
    (
        "Tap-il colour ullam vellam varunnu. Kudikkan pattumo? "
        "Kurachu noisome smell koodiyundu.",
        "water_supply", "urgent", "water",
    ),
    (
        "Water supply ormam oru maasam 20 minutes maatramaanu. "
        "Aaru family members, poraathilla. Help venda.",
        "water_supply", "medium", "water",
    ),
    (
        "Kuzhal thazhe pottirunnu. Vellam mudra-il aakum. "
        "Oru azhcha munnethanne paranju nothing happened.",
        "water_supply", "high", "water",
    ),
    (
        "Water pressure valareya kurannu. Tank nirakkaan 4 hours ekkunu. "
        "Upstairs reach cheyyunnilla.",
        "water_supply", "medium", "water",
    ),
    (
        "Vellam varunna nerathinu notice thannilla. Adutha divasam "
        "full day water illa. Acceptable alla ith.",
        "water_supply", "high", "water",
    ),

    # ── Road damage ────────────────────────────────────────────────────────────
    (
        "Road-il valiya kuzhi undu. Bikes veennu pokunnu. "
        "Urgent-aayi patchwork venam.",
        "road_damage", "urgent", "roads",
    ),
    (
        "Junction-nte munnilu pothole valuth aayi. Oru bike accident aayirunnu. "
        "Please fix.",
        "road_damage", "urgent", "roads",
    ),
    (
        "Road surface yellaam peeli povunnu. Tar eduthupoyitta. "
        "Rain vanna kuzhi full aayi.",
        "road_damage", "high", "roads",
    ),
    (
        "Chinna pothole undu near colony gate. Vehicles avoid cheyyunnu. "
        "Fix venam.",
        "road_damage", "low", "roads",
    ),
    (
        "Newly tarred road-il kuzhi aayi. Avar roadwork finish aayittu "
        "3 months aayittilla? Quality problem undaavum.",
        "road_damage", "medium", "roads",
    ),

    # ── Drainage ────────────────────────────────────────────────────────────
    (
        "Drainage block aanu near market. Rain vanna vellam road-il nirakkunnu. "
        "Clear cheyyanam.",
        "drainage", "high", "drainage",
    ),
    (
        "Oda channal-il plastic niranju block aayirunnu. "
        "Last week rain-il vellakketu aayi. Please clear.",
        "drainage", "high", "drainage",
    ),
    (
        "Mazha vannal road ellaam vellam. Drainage cover pothiyirikkunnu. "
        "Ith oru regular problem.",
        "drainage", "high", "drainage",
    ),
    (
        "Drain-nte kondukkukuval broken aanu. Konjam konjam vellam cheyannu. "
        "Monsoon munnethanne fix venam.",
        "drainage", "medium", "drainage",
    ),

    # ── Solid waste ────────────────────────────────────────────────────────────
    (
        "Waste edukkunilla 3 days aayitta. Mound aayi kidakkunnu "
        "near our gate. Smell kuzhappam.",
        "solid_waste", "high", "health",
    ),
    (
        "Garbage collection miss aayitta. Bin overflow aayi. "
        "Please schedule.",
        "solid_waste", "medium", "health",
    ),
    (
        "Road-varayil chettumaaliniyam kidakkunnu. Stray dogs spread cheyyunnu. "
        "Urgent clearance venam.",
        "solid_waste", "urgent", "health",
    ),
    (
        "Kazhinja azhcha muttham kuzhi thonnathum theechhettu. "
        "Smoke varunnu neighbour compound-il. Complaint cheyyunnu.",
        "solid_waste", "high", "health",
    ),

    # ── Electrical hazard ──────────────────────────────────────────────────────
    (
        "Pole veenu road-il kidakkunnu live wire kuttiyittu. "
        "Aarum varunnilla urgent aanu.",
        "electrical_hazard", "critical", "electricity",
    ),
    (
        "Wire thazhe kidakkunnu school-nte munnilu. "
        "Kuttikalkku valikkunnu. Please KSEB-il parayan.",
        "electrical_hazard", "critical", "electricity",
    ),
    (
        "Junction box open aayi kidakkunnu live terminals exposed. "
        "Very dangerous. Urgent close cheyyanam.",
        "electrical_hazard", "critical", "electricity",
    ),

    # ── Street light ───────────────────────────────────────────────────────────
    (
        "Street light work cheyyunnilla 2 azhcha aayitta. "
        "Dark road, safety problem.",
        "street_light", "medium", "electricity",
    ),
    (
        "Bulb poyi lamp post-il. Maattanam. Night-il dark.",
        "street_light", "low", "electricity",
    ),
    (
        "Naalu street light ellaam off. Road full dark aanu. "
        "Please fix cheyyanam.",
        "street_light", "high", "electricity",
    ),

    # ── Sewage ────────────────────────────────────────────────────────────────
    (
        "Sewage overflow aayi road-il. Smell-um varunnu. "
        "Health risk. Urgent attention venam.",
        "sewage_issue", "urgent", "health",
    ),
    (
        "Kuzhal pottiyirikkunnu. Sewage road-il varunnu. "
        "Fix cheyyanam urgent.",
        "sewage_issue", "urgent", "health",
    ),
    (
        "Manhole-il smell varunnu near colony. Overflow aayi thudangunuvo "
        "ennu ariyilla. Check venam.",
        "sewage_issue", "medium", "health",
    ),

    # ── Water (contamination) ─────────────────────────────────────────────────
    (
        "Tap-il manasamalla vellam varunnu. Yellow colour. "
        "Kudikkan pattumo? Family-nte health important.",
        "water_supply", "urgent", "water",
    ),
    (
        "Vellam varunna time-il sewage smell varunnu. "
        "Contamination aano? Check cheyyanam.",
        "water_supply", "urgent", "water",
    ),

    # ── Tree fall / hazard (use tree_fall — canonical category code) ───────────
    (
        "Valiya maram oru velicchathil veenu pokunnu. "
        "Kaattu koodutalum bhayam. Urgent athu murichu maarnam.",
        "tree_fall", "urgent", "parks",
    ),
    (
        "Maram road-il veenu traffic thadangi. Emergency remove cheyyanam.",
        "tree_fall", "urgent", "parks",
    ),
    (
        "Maram road-nte mele paattirikkunnu. Cycle/bike-il valikkunnu. "
        "Please remove cheyyanam.",
        "tree_fall", "urgent", "parks",
    ),
    (
        "Valiya maram road il veennu. Traffic thadangi. "
        "Ambulance pass aakaan pattumo? Urgent clear cheyyanam.",
        "tree_fall", "urgent", "parks",
    ),

    # ── Illegal construction ──────────────────────────────────────────────────
    (
        "Neighbour road-nte mele slab pottiyirikkunnu permit illa. "
        "Narrowed road vahanagalkkku vanna poru. Complaint.",
        "illegal_construction", "medium", "planning",
    ),
    (
        "Build cheyyunnu compound wall exceed cheythu. Road-nte "
        "space eat aakkitta. Permit undayirunnilla.",
        "illegal_construction", "high", "planning",
    ),

    # ── General civic (mixed categories, diverse Manglish) ───────────────────
    (
        "Footpath-il construction material idthu. Pedestrians road-il "
        "walk cheyyenam. Dangerous.",
        "road_damage", "medium", "roads",
    ),
    (
        "Park-il lights poyi. Azhcha full dark. Youngsters "
        "safety concern parannirikunnu.",
        "street_light", "medium", "electricity",
    ),
    (
        "Mazhakkalam munnethanne drain clear cheyyanam. "
        "Kazhinja varsham vellakketu kharam aayi.",
        "drainage", "medium", "drainage",
    ),
    (
        "Road tharatu ivide thadikkunnu. Vehicles kudukkukunnu. "
        "Fix venam athave accidents aayi pokkum.",
        "road_damage", "high", "roads",
    ),
    (
        "Kooppayinte aduthu ulla light poyi. Kooppaya "
        "raatrathil kaanikkaanull visibility illa.",
        "street_light", "medium", "electricity",
    ),
    (
        "Kadayil vatta malineyam thirichu pokatte. Smell ippo "
        "varanam. Corporation vannu edukkanam.",
        "solid_waste", "medium", "health",
    ),
    (
        "Water pipe-il crack undu near road. Seepage konjam konjam undu. "
        "Monitor cheyyanam.",
        "water_supply", "medium", "water",
    ),
    (
        "Randu divasam aayi vellam varunna kaaryam ariyunna mappilattu "
        "enthu cheyyum? Tanker venam.",
        "water_supply", "high", "water",
    ),
    (
        "Compound wall side-il pothole aayi. Bikes veennu pokunnu. "
        "Repair cheyyanam.",
        "road_damage", "high", "roads",
    ),
    (
        "Drainage line-il plasticku block aayirunnu. Last rain-il overflow. "
        "Clean cheyyanam.",
        "drainage", "high", "drainage",
    ),
]

# ===========================================================================
# FIX 4 — LANDMARK_ALIASES_V2 (~60 aliases)
#
# Root cause: abbreviations (jn→junction, clg→college, med clg→medical college),
# common misspellings (palaym, trivandum, kzhakootam), and shortened names
# (bakery jn, sut hosp) return ward_hint=unknown because the embedding of
# "palaym" is far from "Palayam" in vector space.
#
# These strings are appended to TVM_LOCATIONS_EXTENDED so that when a complaint
# contains an abbreviation or misspelling, landmark similarity will still rank
# the correct area at the top.
# ===========================================================================

LANDMARK_ALIASES_V2: list[str] = [

    # ── Abbreviations: jn / jct → junction ───────────────────────────────────
    "Pattom jn",
    "Bakery jn",
    "Karamana jn",
    "Medical College jn",
    "East Fort jn",
    "Kowdiar jn",
    "Palayam jn",
    "Secretariat jn",
    "Kesavadasapuram jn",
    "Thampanoor jn",
    "Ambalamukku jn",
    "Peroorkada jn",
    "Mudavanmugal jn",
    "Muttada jn",
    "Sreekaryam jn",
    "Kazhakkoottam jn",
    "Ulloor jn",

    # ── Abbreviations: clg / coll → college ───────────────────────────────────
    "Med clg junction",
    "Med clg",
    "Govt med clg",
    "Medical clg",
    "SAT hosp",
    "SUT hosp",
    "Gen hosp",
    "Govt hosp junction",
    "Hosp junction",

    # ── Abbreviations: rd → road ──────────────────────────────────────────────
    "Pattom palace rd",
    "Karamana bridge rd",
    "Kowdiar main rd",
    "Vattiyoorkavu bypass rd",
    "Sreekaryam Kariavattom rd",
    "Ambalamukku Poojappura rd",
    "Pangode military rd",
    "Thirumala hilltop rd",
    "Chalai market rd",

    # ── Common misspellings ──────────────────────────────────────────────────
    "Palaym",               # Palayam
    "Palaym junction",
    "Palaym market",
    "Kazhakootam",          # Kazhakkoottam
    "Kazhakootam junction",
    "Kazhakoottam",
    "Trivandum",            # Thiruvananthapuram
    "Trivandrum central",
    "Trivandrum railway station",
    "Trivandrum secretariat",
    "Kowdiyar",             # Kowdiar
    "Kowdiyar palace",
    "Nanthankode",          # Nanthancode
    "Nanthankode junction",
    "Kesavadasapuram mkt",
    "Attukal temple road",  # common short form
    "Thampanur",            # Thampanoor
    "Thampanur bus stand",
    "Kariavattam",          # Kariavattom
    "Kariavattam university",
    "Pothujanam market",    # informal name for Chalai market area
    "Pullarikulam",         # informal Pullarikulam area near Pettah
    "Veli tourist village",

    # ── Common short forms / compound abbreviations ──────────────────────────
    "East Fort",
    "W Fort",
    "West Fort junction",
    "Museum junction",
    "Museum road",
    "Kims hospital road",
    "District hospital junction",
    "Civil station junction",
    "PWD junction",
    "KSRTC stand junction",
    "Airport road",
    "NH bypass",
    "NH47 junction",
    "Outer ring road junction",
]

# Merged extended location list (for use by train_transformer.py)
# Import this in generate_corpus_v2.py and append to TVM_LOCATIONS_EXTENDED
from apps.ml.training.corpus_data_v3 import TVM_LOCATIONS_EXTENDED as _V3_EXTENDED

TVM_LOCATIONS_V4_EXTENDED: list[str] = _V3_EXTENDED + LANDMARK_ALIASES_V2
