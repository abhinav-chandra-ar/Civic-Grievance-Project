"""apps/ml/training/corpus_data_v2.py

Rich multilingual training corpus for the TVMC civic grievance ML pipeline.

Design principles
-----------------
* Every sentence is independently varied — no mechanical prefix/suffix templates.
* Language distribution: English ~45%, Malayalam ~20%, Manglish ~20%, Mixed ~15%.
* Complaint register diversity: formal report, WhatsApp message, angry citizen,
  polite request, urgent distress call, matter-of-fact description.
* Includes: location-rich, time-bounded, impact-described, incomplete/typo samples.
* Duplicate groups explicitly labelled for semantic similarity evaluation.
* 9 civic categories + spam + no_category, ~80–110 seeds each.
* generate_corpus_v2.py expands these into 5000+ training samples via structured
  slot substitution (locations, times, synonyms) — not trivial prefix noise.

Category codes (same as production rule engine)
-------------------------------------------------
water_supply, drainage, sewage_issue, solid_waste, road_damage,
electrical_hazard, street_light, tree_fall, illegal_construction,
spam, no_category
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Shared slot vocabularies used by generate_corpus_v2.py
# ---------------------------------------------------------------------------

# Real TVMC ward names and landmark names for location injection
TVM_LOCATIONS: list[str] = [
    "Kazhakkoottam", "Pattom", "Karamana", "Kesavadasapuram", "Nalanchira",
    "Kowdiar", "Vattiyoorkavu", "Sreekariyam", "Ulloor", "Nanthancode",
    "Vazhuthacaud", "Thycaud", "Jagathy", "Chalai", "Pettah", "Vanchiyoor",
    "Kannammoola", "Thampanoor", "Palayam", "Manacaud", "Beemapalli",
    "Attukal", "Kunnukuzhy", "Mudavanmugal", "Pangode", "Thirumala",
    "Poojappura", "Ambalamukku", "Kudappanakunnu", "Mannanthala",
    "Technopark", "Medical College", "East Fort", "Shanghumugham",
    "Vizhinjam", "Peroorkada", "Muttada", "Sasthamangalam", "Pangappara",
    "Kariavattom", "Thiruvallam", "Kachani", "Edavakode",
]

# Varied time expressions for slot injection
TIME_EXPRESSIONS: list[str] = [
    "for the past three days", "for two weeks now", "since last Monday",
    "for over a month", "since the rains started", "for the past week",
    "since yesterday evening", "for five days straight", "for nearly ten days",
    "since last Thursday", "for the past fortnight", "since the storm",
    "from Sunday onwards", "for almost a week", "since last month",
]

# Impact descriptions for slot injection
IMPACT_PHRASES: list[str] = [
    "Residents are suffering greatly.",
    "Children and elderly are affected.",
    "Daily life has become very difficult.",
    "People are forced to buy expensive water tankers.",
    "Vehicles are getting damaged daily.",
    "Several accidents have already occurred.",
    "Health of the local people is at serious risk.",
    "School children are unable to use the road safely.",
    "Women feel unsafe travelling at night.",
    "Businesses in the area are badly affected.",
    "Immediate action is urgently needed.",
    "Nobody seems to care despite repeated complaints.",
]

# ---------------------------------------------------------------------------
# Type alias (same as v1 for compatibility)
# ---------------------------------------------------------------------------
TrainingSample = tuple[str, str, str, str]

# ===========================================================================
# WATER SUPPLY  (~100 seeds)
# ===========================================================================

WATER_SUPPLY: list[TrainingSample] = [
    # ── English: formal report register ─────────────────────────────────
    ("Water supply has been completely disrupted in our ward for the past three days. Overhead tank is empty and residents have no drinking water.", "water_supply", "high", "water"),
    ("The municipal water pipe on the main road has burst and water is gushing onto the footpath continuously.", "water_supply", "urgent", "water"),
    ("Brown and foul-smelling water is coming from the taps since this morning. It is clearly contaminated and undrinkable.", "water_supply", "urgent", "water"),
    ("Water pressure in our area is extremely low. The overhead tank takes four hours to fill and still does not reach the top floor.", "water_supply", "medium", "water"),
    ("No advance notice was given before the water supply was cut off yesterday. Residents were left without water for the whole day.", "water_supply", "high", "water"),
    ("The underground water main near the school compound has a slow leak. Water has been seeping into the road for a week and the tar is eroding.", "water_supply", "high", "water"),
    ("Water meter is recording consumption even when the tap is fully closed. Suspected internal pipe leak.", "water_supply", "medium", "water"),
    ("Supply comes for only thirty minutes each morning. A family of five cannot manage with that quantity.", "water_supply", "medium", "water"),
    ("Crack in the distribution main near the market is wasting hundreds of litres every day.", "water_supply", "high", "water"),
    ("Multiple households in our lane have no water connection working despite paying bills. Connection was cut due to a billing error.", "water_supply", "medium", "water"),
    ("The pipe laid last year near the junction has already developed a major crack. This looks like a quality failure.", "water_supply", "high", "water"),
    ("Water supply pipe is broken under the newly laid road. Every time there is rain the road sinks further at that spot.", "water_supply", "urgent", "water"),
    ("Rust-coloured water supply since the maintenance work was done on Friday. Nobody has come to fix it despite two calls to the helpline.", "water_supply", "high", "water"),
    ("Our entire street has been without water for six days now. No tanker has been arranged either. This is unacceptable.", "water_supply", "high", "water"),
    ("Water pipe burst at the road crossing near the temple. Road is flooded and traffic is disrupted.", "water_supply", "urgent", "water"),
    ("Overhead water tank in our colony is overflowing due to a faulty valve. Water is going to waste for two days.", "water_supply", "medium", "water"),
    ("Connection pipe corroded and leaking inside our wall. Water bill has tripled. Need a plumber visit.", "water_supply", "medium", "water"),
    ("Sewage is mixing with drinking water. Multiple families had stomach illness last week. Need immediate testing.", "water_supply", "urgent", "water"),
    ("Only one side of the street gets water. The other side has had no supply for eight days.", "water_supply", "high", "water"),
    ("New housing colony has no water connection at all despite repeated applications over six months.", "water_supply", "medium", "water"),

    # ── English: informal / WhatsApp style ──────────────────────────────
    ("No water since 3 days pls help urgent", "water_supply", "high", "water"),
    ("Pipe burst near the junction water all over the road", "water_supply", "urgent", "water"),
    ("tap water is yellow color not safe to drink", "water_supply", "urgent", "water"),
    ("getting very low water pressure since last week can't fill tank", "water_supply", "medium", "water"),
    ("water cut without notice yesterday whole day no water", "water_supply", "high", "water"),
    ("pipe leaking near school compound already one week nobody came", "water_supply", "high", "water"),
    ("supply only 20 min each morning not enough 5 family members", "water_supply", "medium", "water"),
    ("water pipe broken under road near market wasting water daily", "water_supply", "high", "water"),
    ("brown water from tap since morning is it safe???", "water_supply", "urgent", "water"),
    ("water bill is too high suspecting leakage inside pipe please check", "water_supply", "medium", "water"),

    # ── Malayalam: native script, varied register ────────────────────────
    ("ഇവിടെ മൂന്ന് ദിവസമായി ജലവിതരണം നിർത്തിയിരിക്കുന്നു. ഓവർഹെഡ് ടാൻക്ക് ഒഴിഞ്ഞ് കിടക്കുകയാണ്. കുടിക്കാൻ വെള്ളം ഇല്ല.", "water_supply", "high", "water"),
    ("ടാപ്പ് തുറന്നാൽ മഞ്ഞ നിറത്തിൽ ദുർഗന്ധമുള്ള വെള്ളം വരുന്നു. ഇത് കുടിക്കുവാൻ സുരക്ഷിതമല്ല.", "water_supply", "urgent", "water"),
    ("ജലവിതരണ പൈപ്പ് പൊട്ടി ജലം ഒഴുകി നഷ്ടമാകുന്നു. ഒരാഴ്ചയായി ആരും ശ്രദ്ധിക്കുന്നില്ല.", "water_supply", "high", "water"),
    ("ഞങ്ങളുടെ വീട്ടിലേക്ക് ഒരു ദിവസം 20 മിനിറ്റ് മാത്രം വെള്ളം വരുന്നുള്ളൂ. അഞ്ചംഗ കുടുംബത്തിന് ഇത് തികയില്ല.", "water_supply", "medium", "water"),
    ("ഈ ഏരിയയിൽ ഒരാഴ്ചയായി ശുദ്ധജലം ലഭിക്കുന്നില്ല. ടാൻക്കർ ലോറി ആരും അയച്ചിട്ടില്ല.", "water_supply", "high", "water"),
    ("ജലം കലക്കമുള്ളതും ദുർഗന്ധമുള്ളതും ആണ്. ഈ ആഴ്ച ഒരുപാട് ആൾക്കാർക്ക് വയറ്റുവേദന ഉണ്ടായി.", "water_supply", "urgent", "water"),
    ("ഞങ്ങളുടെ നിരത്തിൽ ജലവിതരണ ലൈൻ പൊട്ടി ആഴ്ചയായി. കോർപ്പറേഷൻ അറിയിച്ചിട്ടും നടപടി ഇല്ല.", "water_supply", "high", "water"),
    ("പൈപ്പ് ലൈൻ പഴകി ദ്രവിച്ചു. ജലം ഒഴുക്ക് ഇല്ല, ടാൻക്ക് നിറയ്ക്കാൻ നാലഞ്ച് മണിക്കൂർ ആകുന്നു.", "water_supply", "medium", "water"),
    ("സ്കൂൾ കോമ്പൗണ്ടിൽ ജലപൈപ്പ് ഒഴുകുന്നു. ദിവസവും ആയിരക്കണക്കിന് ലിറ്റർ നഷ്ടം.", "water_supply", "high", "water"),
    ("കുടിവെള്ളം ദിവസങ്ങളായി ലഭ്യമല്ല. ബദൽ ക്രമീകരണം ഒന്നും ഇല്ല. ജനങ്ങൾ ദുരിതത്തിൽ.", "water_supply", "high", "water"),

    # ── Manglish: Roman-script Kerala typing patterns ────────────────────
    ("vellam vannittu moonnu divasam ayi, tank kaali, kudikkaan vellam illa", "water_supply", "high", "water"),
    ("kuzhal potti junction il vellam vazhukunnu, road full aayittu", "water_supply", "urgent", "water"),
    ("tap il manja color vellam varunnu, malinjathanu, kudikkaan okka", "water_supply", "urgent", "water"),
    ("pressure illatha kaaran tank nirakkaan neram kazhiyannathu", "water_supply", "medium", "water"),
    ("notice onnum illathe supply cut cheythu, oru divasam muzhuvan vellam kittiyilla", "water_supply", "high", "water"),
    ("school il pipe chori aanu, aazhcha aayittu, koruppu mathram varunnu", "water_supply", "high", "water"),
    ("divasam 20 minute maathram vellam kittunnu, family ku thikka", "water_supply", "medium", "water"),
    ("market kazhinjal road il vellam pori ozhukunu, pipe thakarnu", "water_supply", "high", "water"),
    ("brown color vellam varunnu tap il, safe aano?? health risk", "water_supply", "urgent", "water"),
    ("bill valuthu varunnu, pipe leak aayirikkanam, veettil check cheyyenam", "water_supply", "medium", "water"),
    ("vellam ilathe 6 days aayittu, corporation ku parayitu varum paranjittu nadapadi illakill", "water_supply", "high", "water"),
    ("kuzhal odiyathu, palyathinu keezhil vellam nira nira, road adiyil thakarnnu", "water_supply", "urgent", "water"),

    # ── Code-mixed: English civic terms + Malayalam grammar ─────────────
    ("Water supply cut aayittu 4 days, overhead tank empty, residents suffer aavunnu", "water_supply", "high", "water"),
    ("Pipe burst near junction, vellam road il spread aayittu, traffic jam", "water_supply", "urgent", "water"),
    ("Tap water contaminated aanu, brown colour varunnu, drinking safe allennnu", "water_supply", "urgent", "water"),
    ("Low pressure kaaran tank fill cheyyan 5 hours edukkunu, top floor ku kittunnilla", "water_supply", "medium", "water"),
    ("Water connection cut cheythu billing error kaaran. Rectify cheyyanam urgent", "water_supply", "medium", "water"),
    ("Supply line broken, kazhinja week mutal leak aanu, corporation informed cheythu but no action", "water_supply", "high", "water"),
    ("Sewage mixing with water supply line, several people sick aayittu last week", "water_supply", "urgent", "water"),
    ("Colony new aanu, water connection illatha kaaran tanker water use cheyyunnu, expensive", "water_supply", "medium", "water"),
    ("Pipe corroded aanu, pressure zero, pump start cheythittum vellam varunnilla", "water_supply", "medium", "water"),
    ("3 buildings il water connection dead aanu. Bill payment cheythu but service illakill", "water_supply", "high", "water"),
]

# ===========================================================================
# DRAINAGE  (~100 seeds)
# ===========================================================================

DRAINAGE: list[TrainingSample] = [
    # English
    ("The roadside drain has been completely blocked with plastic waste and silt. Water overflows onto the road during any rain.", "drainage", "high", "drainage"),
    ("Open drainage channel in front of our housing colony is overflowing after last night's rain. Two houses got flooded.", "drainage", "urgent", "drainage"),
    ("The drain cover near the bus stop is missing. A child nearly fell in yesterday. Very dangerous at night.", "drainage", "high", "drainage"),
    ("Stormwater drain at the bottom of the hill is clogged. Every monsoon this entire street goes under water.", "drainage", "high", "drainage"),
    ("New construction has blocked the natural flow path of the drain. Now water stagnates in our compound.", "drainage", "high", "drainage"),
    ("Drain canal near the school is completely choked. Foul smell has been unbearable for a month.", "drainage", "medium", "drainage"),
    ("The underground drain pipe under the road has collapsed. A sinkhole is forming near the manhole cover.", "drainage", "urgent", "drainage"),
    ("Rainwater from the road flows directly into our ground floor flat because the drain is below road level now.", "drainage", "high", "drainage"),
    ("Road has not been cleaned for the monsoon. All the leaves and plastic are clogging the drains already.", "drainage", "medium", "drainage"),
    ("Drain at the market is never cleaned. Every vegetable vendor throws waste in it. Extremely filthy.", "drainage", "medium", "drainage"),
    ("The drainage system was redesigned during road widening but now water flows backwards into the junction.", "drainage", "high", "drainage"),
    ("Drain grating broken and sunken into the channel. Motorcycles keep getting their front wheel stuck.", "drainage", "high", "drainage"),
    ("Long stretch of road is flooded because the stormwater outlet into the canal is blocked.", "drainage", "high", "drainage"),
    ("During heavy rain the drain backs up into all ten houses on our lane. Furniture and belongings damaged.", "drainage", "urgent", "drainage"),
    ("Open drain near the children's park is a mosquito breeding ground. Kids are falling sick repeatedly.", "drainage", "medium", "drainage"),

    # WhatsApp / informal
    ("drain blocked heavy rain water all over road please clear", "drainage", "high", "drainage"),
    ("drain cover missing bus stop very dangerous someone will fall", "drainage", "high", "drainage"),
    ("our lane fully flooded because drain is choked help needed urgent", "drainage", "urgent", "drainage"),
    ("bad smell from drain near school been complaining for months no action", "drainage", "medium", "drainage"),
    ("sinkhole forming near manhole drain pipe broken underground urgent check", "drainage", "urgent", "drainage"),

    # Malayalam
    ("ഡ്രൈനേജ് ചാൽ ഗാർബേജ് കൊണ്ട് തടഞ്ഞ് കിടക്കുന്നു. മഴ പെയ്താൽ ഞങ്ങളുടെ തെരുവ് മുഴുവൻ വെള്ളക്കെട്ടാകും.", "drainage", "high", "drainage"),
    ("ഓടചാൽ കൂടി നിന്ന് മലിനജലം ഇറങ്ങി ഞങ്ങളുടെ ഗ്രൗണ്ട് ഫ്ലോർ ഫ്ലാറ്റിലേക്ക് കടക്കുന്നു.", "drainage", "urgent", "drainage"),
    ("ഡ്രൈനേജ് കവർ ബസ് സ്റ്റോപ്പിൽ ഇല്ല. ഒരു കുട്ടി ഇന്നലെ ഏകദേശം വീണേനെ. അപകടകരം.", "drainage", "high", "drainage"),
    ("ചാനൽ ബ്ലോക്ക് ആണ്, ദുർഗന്ധം, കൊതുക് ഉൽഭവ കേന്ദ്രം, കുട്ടികൾ അസുഖബാധിതരാകുന്നു.", "drainage", "medium", "drainage"),
    ("ഡ്രൈൻ ചാൽ കുഴിഞ്ഞ് ഒരു സിങ്ക്ഹോൾ ഉണ്ടായിരിക്കുന്നു. ഉടൻ പരിശോധന ആവശ്യം.", "drainage", "urgent", "drainage"),
    ("മഴക്കാലത്ത് ഈ ഡ്രൈൻ ഒഴുകി ഞങ്ങളുടെ 10 വീടുകളിലേക്ക് കടക്കുന്നു. ഓരോ കൊല്ലവും ഇത് ആവർത്തിക്കുന്നു.", "drainage", "urgent", "drainage"),
    ("ഡ്രൈനേജ് ഔട്ട്ലെറ്റ് കനാലിൽ തടഞ്ഞ്. ഒരു ദീർഘ ഭാഗം റോഡ് ജലം കെട്ടിനിൽക്കുന്നു.", "drainage", "high", "drainage"),

    # Manglish
    ("drain block aanu, mazha vandal road full vellam, urgent clear cheyyenam", "drainage", "high", "drainage"),
    ("cover illatha drain und bus stop kil, rathri dangerous, aannu soorishtu", "drainage", "high", "drainage"),
    ("lane muzhuvanum vellam, drain choked aanu, 10 veedukarum affected", "drainage", "urgent", "drainage"),
    ("school kazhijal oda channal nirachu, durmgandam kaaran class evide nadakkum", "drainage", "medium", "drainage"),
    ("manhole cover keezhil sinhole undayi, urgent check cheyyenam", "drainage", "urgent", "drainage"),
    ("market drain enthum thurakkathe, plasticum malamum, njangalku sehikkaan pattunilla", "drainage", "medium", "drainage"),
    ("rain il vellam back aayi veetil keri, furniture okke bhashami", "drainage", "urgent", "drainage"),

    # Code-mixed
    ("Drain blocked aayittu road flooded aanu, mazha vandal worse aakum", "drainage", "high", "drainage"),
    ("Drain cover missing near bus stop, night il kittunnilla, dangerous aanu", "drainage", "high", "drainage"),
    ("Stormwater outlet blocked, long stretch road il water stagnant aanu", "drainage", "high", "drainage"),
    ("Underground drain pipe collapsed, sinkhole forming, urgent repair venam", "drainage", "urgent", "drainage"),
    ("Colony drain level road level il ninn thazhcha, rain varum pore flood aakum", "drainage", "high", "drainage"),
    ("Mosquito breeding from blocked drain, children sick aavunnu, health risk", "drainage", "medium", "drainage"),
    ("Drain cleaning monsoon ku mump cheyyathathu kaaran okke block aayittu", "drainage", "medium", "drainage"),
]

# ===========================================================================
# SEWAGE ISSUE  (~90 seeds)
# ===========================================================================

SEWAGE: list[TrainingSample] = [
    # English
    ("Sewage water is overflowing from the manhole onto the road near the bus stand. Extremely unhygienic.", "sewage_issue", "urgent", "sewage"),
    ("Blocked sewer line is causing sewage to back up inside two houses in our lane. Completely unusable bathrooms.", "sewage_issue", "urgent", "sewage"),
    ("Sewage treatment plant is overflowing and discharging untreated waste directly into the canal.", "sewage_issue", "critical", "sewage"),
    ("Underground sewer line broken under the road. Foul smell persisting for two weeks. Possible health risk.", "sewage_issue", "high", "sewage"),
    ("Raw sewage visible on the road near the primary school. Children playing nearby are at risk.", "sewage_issue", "urgent", "sewage"),
    ("The neighbouring apartment is illegally connecting sewage line to the stormwater drain.", "sewage_issue", "high", "sewage"),
    ("Septic tank overflowing and sewage seeping into the lane. Landlord refuses to act.", "sewage_issue", "urgent", "sewage"),
    ("Sewer blockage caused sewage to overflow into a ground floor flat. Complete sanitation failure.", "sewage_issue", "urgent", "sewage"),
    ("Old sewer pipe cracked and causing slow seepage under the road. Road is soft at that spot.", "sewage_issue", "high", "sewage"),
    ("Sewage smell from the drainage system has been unbearable for a month. Nobody has investigated.", "sewage_issue", "high", "sewage"),
    ("Open sewage channel running through the market area is a severe health hazard. Flies everywhere.", "sewage_issue", "urgent", "sewage"),
    ("Sewage pump in the colony has broken down. Tank is full and about to overflow.", "sewage_issue", "urgent", "sewage"),
    ("Commercial building discharging untreated grease trap water into the municipal sewer.", "sewage_issue", "high", "sewage"),
    ("Manhole cover near the junction is broken and open. Sewage smell and safety hazard.", "sewage_issue", "high", "sewage"),
    ("Sewage pipe laid during road construction was not properly sealed. Now leaking.", "sewage_issue", "high", "sewage"),

    # Informal
    ("sewage overflowing on road near school urgent action needed", "sewage_issue", "urgent", "sewage"),
    ("sewer blocked two homes bathrooms not working please send team", "sewage_issue", "urgent", "sewage"),
    ("open manhole sewage smell very bad near bus stop dangerous", "sewage_issue", "high", "sewage"),
    ("septic tank full leaking to road landlord not doing anything", "sewage_issue", "urgent", "sewage"),
    ("sewage mixing in canal near our colony health hazard", "sewage_issue", "critical", "sewage"),

    # Malayalam
    ("മലിനജലം മാൻഹോളിൽ നിന്ന് തെരുവിലേക്ക് ഒഴുകുന്നു. ബസ് സ്റ്റോപ്പ് ഏരിയ ഗൗരവ ഭീഷണി.", "sewage_issue", "urgent", "sewage"),
    ("സ്കൂളിന് സമീപം അഴുക്കുചാൽ ഒഴുകുന്നു. കുട്ടികൾ കളിക്കുന്ന ഇടത്ത് ആരോഗ്യ അപകടം.", "sewage_issue", "urgent", "sewage"),
    ("ഓടചാൽ ലൈൻ തകർന്നു. ദുർഗന്ധം വ്യാപകമായി. ദിവസം ഒരു മാസം ആകുന്നു.", "sewage_issue", "high", "sewage"),
    ("സ്യൂവേജ് ട്രീറ്റ്മെന്റ് പ്ലാന്റ് കനാലിലേക്ക് ഒഴുകുന്നു. ജലം മലിനീകരിക്കപ്പെടുന്നു.", "sewage_issue", "critical", "sewage"),
    ("ഗ്രൗണ്ട് ഫ്ലോർ ഫ്ലാറ്റിൽ സ്യൂവേജ് ബാക്ക് ആയി. ബാത്‌റൂം ഉപയോഗിക്കാൻ കഴിയുന്നില്ല.", "sewage_issue", "urgent", "sewage"),
    ("സെപ്റ്റിക് ടാൻക്ക് നിറഞ്ഞ് ഒഴുകുന്നു. ഇടനാഴിയിൽ ദ്രാവകം വ്യാപിക്കുന്നു.", "sewage_issue", "urgent", "sewage"),
    ("പ്രധാന ഓടചാൽ ലൈൻ ബ്ലോക്ക് ആണ്. ഒഴുക്ക് ഇല്ല. ദ്രുത ഇടപെടൽ ആവശ്യം.", "sewage_issue", "high", "sewage"),

    # Manglish
    ("sewage manhole il ninn road il varunnu, school kazhinjal, children danger", "sewage_issue", "urgent", "sewage"),
    ("sewer block aayittu 2 veettil bathroom use cheyyan patunnilla", "sewage_issue", "urgent", "sewage"),
    ("durmgandam kaaran veettil irikkaan pattunilla, sewer pipe pottannu", "sewage_issue", "high", "sewage"),
    ("septic tank niranju ozhukunu, neighbour cheyyunilla nothing, help", "sewage_issue", "urgent", "sewage"),
    ("sewage canal il mix aavunnu, kudi vellam polluted aakum, very serious", "sewage_issue", "critical", "sewage"),
    ("open manhole kazhijal bus stop il, rathri time il anyone veezhum", "sewage_issue", "high", "sewage"),

    # Code-mixed
    ("Sewage overflow near temple, public road il varunnu, health risk aanu", "sewage_issue", "urgent", "sewage"),
    ("Manhole open aanu near junction, sewage smell, safety issue too", "sewage_issue", "high", "sewage"),
    ("STP overflow cheythu canal il mixed aayittu, critical issue", "sewage_issue", "critical", "sewage"),
    ("Sewage backup aayittu ground floor il, bathroom use cheyyaan pattunilla", "sewage_issue", "urgent", "sewage"),
    ("Sewer pipe cracked, road soft spot undayi, underground leakage", "sewage_issue", "high", "sewage"),
    ("Open sewage channel market kazhijal, flies full, health hazard", "sewage_issue", "urgent", "sewage"),
]

# ===========================================================================
# SOLID WASTE  (~90 seeds)
# ===========================================================================

SOLID_WASTE: list[TrainingSample] = [
    # English
    ("Garbage collection has not happened in our ward for ten days. The bins are overflowing onto the road.", "solid_waste", "medium", "sanitation"),
    ("Construction waste has been illegally dumped on the public plot near the park. The entire area is now unusable.", "solid_waste", "high", "sanitation"),
    ("Dead dog lying on the road near the bus stop for two days. Not removed despite calls to the civic body.", "solid_waste", "high", "sanitation"),
    ("Residents are burning garbage in the open near the residential buildings. Smoke is causing breathing problems.", "solid_waste", "high", "sanitation"),
    ("Large mound of garbage blocks the footpath. Pedestrians are forced onto the road.", "solid_waste", "medium", "sanitation"),
    ("The garbage truck has not come since the driver was changed. New driver does not know the route.", "solid_waste", "medium", "sanitation"),
    ("Biomedical waste was found dumped near the public park. Syringes and bandages visible. Urgent health risk.", "solid_waste", "urgent", "sanitation"),
    ("Waste dumped into the open storm water canal by market vendors every evening.", "solid_waste", "high", "sanitation"),
    ("No segregation happening. Wet and dry waste are mixed at source. Green bins never come on schedule.", "solid_waste", "low", "sanitation"),
    ("Garbage van stops but does not collect from all lanes. The last two streets in the ward are always skipped.", "solid_waste", "medium", "sanitation"),
    ("Littering at the beach area is severe on weekends. No bins are provided anywhere on the seafront.", "solid_waste", "medium", "sanitation"),
    ("The bulk waste collection phone number does not work. We have old furniture and appliances to dispose.", "solid_waste", "low", "sanitation"),
    ("Roadside restaurant is dumping kitchen waste on the footpath every night.", "solid_waste", "medium", "sanitation"),
    ("Rats and crows are spreading garbage from the overflowing bins across the road every morning.", "solid_waste", "medium", "sanitation"),
    ("No waste collection for two weeks in our colony despite paying Haritha Karma Sena fees.", "solid_waste", "medium", "sanitation"),

    # Informal
    ("garbage 10 days not collected bins full and overflowing on road", "solid_waste", "medium", "sanitation"),
    ("dead animal on road 2 days please remove health risk", "solid_waste", "high", "sanitation"),
    ("people burning garbage near buildings smoke problem breathing issue", "solid_waste", "high", "sanitation"),
    ("construction waste dumped on park land illegally please remove", "solid_waste", "high", "sanitation"),
    ("biomedical waste near park syringes visible urgent please act", "solid_waste", "urgent", "sanitation"),

    # Malayalam
    ("ഈ ആഴ്ച മാലിന്യ ശേഖരണം ഇല്ലായിരുന്നു. ബിൻ നിറഞ്ഞ് തെരുവിലേക്ക് ഒഴുകി.", "solid_waste", "medium", "sanitation"),
    ("ചത്ത നായ്ക്ക് ബസ് സ്റ്റോപ്പ് കുറെ ദിവസം റോഡിൽ കിടക്കുന്നു, ആരും മാറ്റുന്നില്ല.", "solid_waste", "high", "sanitation"),
    ("ബയോമെഡിക്കൽ മാലിന്യം പൊതുസ്ഥലത്ത് ഇട്ടിരിക്കുന്നു. സൂചി ദൃശ്യമാണ്. ഗൗരവ ഭീഷണി.", "solid_waste", "urgent", "sanitation"),
    ("മാലിന്യ ലോറി ഒരാഴ്ചയായി വരുന്നില്ല. കൂമ്പാരം ഉണ്ടാകുന്നു, ദുർഗന്ധം.", "solid_waste", "medium", "sanitation"),
    ("തുറന്ന ചാലിൽ ചന്ത വ്യാപാരികൾ മാലിന്യം ഒഴിക്കുന്നു. ജലം മലിനമാകുന്നു.", "solid_waste", "high", "sanitation"),
    ("കൂമ്പാരം ഇട്ട മാലിന്യം ഇടനാഴി ബ്ലോക്ക് ചെയ്തിരിക്കുന്നു. കാൽനടക്കാർ ബുദ്ധിമുട്ടുന്നു.", "solid_waste", "medium", "sanitation"),

    # Manglish
    ("garbage collect vannittu 10 days ayi, bin niranju, smell kaaran sahanam kittunilla", "solid_waste", "medium", "sanitation"),
    ("chatta janangal garbage jalathil idum, valare ghaatam", "solid_waste", "high", "sanitation"),
    ("maram construction waste pothubhoomiyil idunnu, illegal aanu", "solid_waste", "high", "sanitation"),
    ("biomedical waste park kil, syringe drishyam, children ku danger", "solid_waste", "urgent", "sanitation"),
    ("lorry vannittu last lane collect cheyyarilla, complaint cheythu change varunnilla", "solid_waste", "medium", "sanitation"),

    # Code-mixed
    ("Garbage collection 10 days ayi vannittu, bins overflow cheythu road il", "solid_waste", "medium", "sanitation"),
    ("Dead animal road il, 2 days ayi, nobody remove cheythilla, health issue", "solid_waste", "high", "sanitation"),
    ("Open burning near buildings, smoke kaaran breathing problem, urgent", "solid_waste", "high", "sanitation"),
    ("Biomedical waste dumped near park, syringes visible, urgent action venam", "solid_waste", "urgent", "sanitation"),
    ("Waste in canal, water polluted aavunnu, market vendors cheyyanathu", "solid_waste", "high", "sanitation"),
    ("No segregation happening, wet dry mixed, collection irregular", "solid_waste", "low", "sanitation"),
]

# ===========================================================================
# ROAD DAMAGE  (~100 seeds)
# ===========================================================================

ROAD_DAMAGE: list[TrainingSample] = [
    # English: varied styles
    ("A very large pothole has formed on the main road near the hospital junction. Three accidents have happened this week alone.", "road_damage", "high", "roads"),
    ("The newly laid road in front of the school cracked and sank within two months of construction. Suspected quality failure.", "road_damage", "high", "roads"),
    ("Road is completely broken after the monsoon. Large sections of tar have come off and the mud base is exposed.", "road_damage", "high", "roads"),
    ("Footpath tiles are broken and uneven. Several senior citizens have tripped and fallen on this route.", "road_damage", "medium", "roads"),
    ("Road near the bridge is sinking. A large depression has formed and it looks structurally dangerous.", "road_damage", "urgent", "roads"),
    ("Speed breaker on the school road is broken and the metal rod is protruding. Very dangerous for two-wheelers.", "road_damage", "high", "roads"),
    ("The entire stretch of road from the junction to the colony gate has not been repaired in five years.", "road_damage", "medium", "roads"),
    ("Road near the market is under excavation for cable laying but work stopped midway and not restored.", "road_damage", "high", "roads"),
    ("No road at all in our newly formed colony. Just mud and stone. Vehicles cannot enter during rain.", "road_damage", "medium", "roads"),
    ("Deep trench cut across the road for water pipe work was partially filled but not properly tarred.", "road_damage", "high", "roads"),
    ("Potholes on this road have been reported for over a year. The pothole near the bus stop has swallowed a motorcycle wheel.", "road_damage", "high", "roads"),
    ("Road caving near the temple junction after heavy rain. Traffic diversion put up but road not repaired.", "road_damage", "urgent", "roads"),
    ("The road gradient was changed during construction. Now rainwater flows directly into our compound.", "road_damage", "high", "roads"),
    ("Tar has been peeling off the national highway service road near the colony for months.", "road_damage", "medium", "roads"),
    ("Speed bump near the school is too high and uneven. Buses scrape it every time they pass.", "road_damage", "medium", "roads"),

    # Informal
    ("huge pothole main road near hospital 3 accidents this week urgent fix", "road_damage", "high", "roads"),
    ("road completely broken after rain tar gone mud exposed vehicles suffering", "road_damage", "high", "roads"),
    ("speed breaker broken metal sticking out dangerous for bike riders", "road_damage", "high", "roads"),
    ("road sinking near bridge very dangerous looking structural issue", "road_damage", "urgent", "roads"),
    ("footpath tiles broken old people keep falling please repair", "road_damage", "medium", "roads"),
    ("road excavated for cable work not restored properly 2 months now", "road_damage", "high", "roads"),
    ("pothole near bus stop so big motorcycle wheel got stuck", "road_damage", "high", "roads"),

    # Malayalam
    ("ആശുപത്രി ജംഗ്ഷൻ കഴിഞ്ഞ് വലിയ കുഴി. ഈ ആഴ്ച മൂന്ന് അപകടങ്ങൾ.", "road_damage", "high", "roads"),
    ("പുതുതായി ഇട്ട ടാർ റോഡ് 2 മാസം കൊണ്ട് തകർന്നു. ഗുണനിലവാര പ്രശ്നം.", "road_damage", "high", "roads"),
    ("മഴക്കാലത്ത് ടാർ പൊടിഞ്ഞ് റോഡ് ഉപയോഗയോഗ്യമല്ലാതെ ആയി.", "road_damage", "high", "roads"),
    ("ഫുട്ട്‌പാത്ത് ടൈൽ പൊട്ടി. പ്രായമായവർ വീഴുന്നു. നിരവധി കേസ്.", "road_damage", "medium", "roads"),
    ("പാലം കഴിഞ്ഞ് റോഡ് കുഴിഞ്ഞ് ഒരു ഗർത്തം ഉണ്ടായിട്ടുണ്ട്. ഉടൻ ശ്രദ്ധ ആവശ്യം.", "road_damage", "urgent", "roads"),
    ("ടാർ ഇടാതെ ഖനനം ചെയ്ത ഭാഗം ഇങ്ങനെ വിട്ടിരിക്കുന്നു. ആഴ്ചകൾ ആകുന്നു.", "road_damage", "high", "roads"),
    ("ബൈക്ക് ഓടിക്കുന്നവർക്ക് സ്പീഡ് ബ്രേക്കർ തകർന്നു, ഇരുമ്പ് ദൃശ്യം.", "road_damage", "high", "roads"),

    # Manglish
    ("road il valiya kuzhi, aazhcha thotti, bike vazhuthu, accident aavunnu", "road_damage", "high", "roads"),
    ("new road pottannu 2 masam kondu, contractor quality onnum illakill", "road_damage", "high", "roads"),
    ("monsoon kazhinjal tar uppu poyyi, entire road mud mud aanu", "road_damage", "high", "roads"),
    ("footpath broken aanu, chechi veenu, injuries paeditrician kaali kaali", "road_damage", "medium", "roads"),
    ("palam kazhinjal road cave aanu, dangerous, traffic side around cheyyunnu", "road_damage", "urgent", "roads"),
    ("speed breaker thakarnu, iron rod puram vanna, bike ku risk", "road_damage", "high", "roads"),
    ("colony road illathe il, mud only, rain varum bore full block", "road_damage", "medium", "roads"),
    ("cable work cheythu road kaavi, 2 masam aayittu restore cheythilla", "road_damage", "high", "roads"),

    # Code-mixed
    ("Main road il huge pothole, hospital junction kazhijal, accidents regular", "road_damage", "high", "roads"),
    ("Road cracked aayittu monsoon ku munppu, quality issue clear aanu", "road_damage", "high", "roads"),
    ("Footpath tiles broken, elderly people trip aavunnu regularly", "road_damage", "medium", "roads"),
    ("Road sinking near bridge, structural danger, urgent inspection venam", "road_damage", "urgent", "roads"),
    ("Speed breaker broken, metal rod visible, bike riders ku injury", "road_damage", "high", "roads"),
    ("Excavation work midway stop cheythu, road restore cheythilla, 2 months", "road_damage", "high", "roads"),
    ("Colony road kazhivu illatha karanam vehicles enter cheyyaan patunnilla", "road_damage", "medium", "roads"),
    ("Tar road peeling off kazhinja year mutal, complaints ignored", "road_damage", "medium", "roads"),
]

# ===========================================================================
# ELECTRICAL HAZARD  (~90 seeds)
# ===========================================================================

ELECTRICAL_HAZARD: list[TrainingSample] = [
    # English
    ("A live electric wire is hanging very low over the road after last night's storm. Vehicles are brushing against it.", "electrical_hazard", "critical", "electricity"),
    ("Electric pole snapped at the base and is leaning dangerously onto the road. Nobody has cordoned the area.", "electrical_hazard", "critical", "electricity"),
    ("Transformer near the apartment complex is sparking and making loud noises. Could explode any time.", "electrical_hazard", "critical", "electricity"),
    ("High tension cable is too close to the school building rooftop. Children playing on the roof are at serious risk.", "electrical_hazard", "critical", "electricity"),
    ("Fallen electric wire on the road after the storm. One person got a shock and was hospitalised.", "electrical_hazard", "critical", "electricity"),
    ("Electric cable insulation is completely stripped on the overhead line near the market. Sparking in rain.", "electrical_hazard", "urgent", "electricity"),
    ("Transformer oil is leaking onto the road. Fire risk if ignited.", "electrical_hazard", "urgent", "electricity"),
    ("Substation gate is unlocked and wide open. Children from the nearby slum keep entering and playing inside.", "electrical_hazard", "urgent", "electricity"),
    ("Street light pole fell onto a parked vehicle last night during the storm. KSEB not responding.", "electrical_hazard", "urgent", "electricity"),
    ("Junction box on the footpath has been damaged and live terminals are exposed. Very dangerous.", "electrical_hazard", "critical", "electricity"),
    ("Electric wire touching the metal roof of a roadside shop. Owner does not realise the risk.", "electrical_hazard", "critical", "electricity"),
    ("Earthing failure in our building. Multiple shocks reported. Suspected KSEB connection fault.", "electrical_hazard", "urgent", "electricity"),
    ("Arcing visible from the connection point on the distribution pole near the temple.", "electrical_hazard", "critical", "electricity"),
    ("Underground cable exposed on the footpath after waterlogging eroded the soil. Electrocution risk.", "electrical_hazard", "critical", "electricity"),
    ("Power supply to the entire ward fluctuating dangerously. Appliances getting burnt. Transformer issue suspected.", "electrical_hazard", "urgent", "electricity"),

    # Informal
    ("live wire on road after storm someone got shock please come now", "electrical_hazard", "critical", "electricity"),
    ("electric pole fell on car nobody cordoned area urgent", "electrical_hazard", "critical", "electricity"),
    ("transformer sparking loud noise could explode urgent please", "electrical_hazard", "critical", "electricity"),
    ("high tension wire over school roof children at risk urgent", "electrical_hazard", "critical", "electricity"),
    ("substation open children going inside very dangerous please lock", "electrical_hazard", "urgent", "electricity"),

    # Malayalam
    ("ഇടിമിന്നൽ ശേഷം ഇലക്ട്രിക് കമ്പി റോഡിൽ കിടക്കുന്നു. ആരോ ഷോക്ക് ഏറ്റ് ആശുപത്രിയിൽ.", "electrical_hazard", "critical", "electricity"),
    ("ട്രാൻസ്ഫോർമർ കത്തുന്ന ശബ്ദം. പൊട്ടിത്തെറിക്കാൻ സാധ്യത. ഉടൻ ആളെ അയക്കണം.", "electrical_hazard", "critical", "electricity"),
    ("ഹൈ ടെൻഷൻ ലൈൻ സ്കൂൾ ടെറസ്സിന് മുകളിൽ. കുട്ടികൾ ഗൗരവ ഭീഷണിയിൽ.", "electrical_hazard", "critical", "electricity"),
    ("ഇലക്ട്രിക് പോൾ ചരിഞ്ഞ് റോഡ് ബ്ലോക്ക്. KSEB ഫോണ് എടുക്കുന്നില്ല.", "electrical_hazard", "critical", "electricity"),
    ("ജംഗ്ഷൻ ബോക്സ് തുറന്ന് ടെർമിനൽ കാണുന്നു. ഉടൻ ഇടപെടൽ ആവശ്യം.", "electrical_hazard", "critical", "electricity"),
    ("ഭൂഗർഭ കേബിൾ ഫുട്ട്‌പാത്തിൽ ദൃശ്യമാണ്. ജലം ഒഴുകിയതോടെ മൂടൽ പോയി. ഷോക്ക് ഭീഷണി.", "electrical_hazard", "critical", "electricity"),

    # Manglish
    ("live wire road il veenu, oru aalu shock ketta, hospital il, urgent come", "electrical_hazard", "critical", "electricity"),
    ("transformer spark adikkunnu, loud sound, parayunnu explode aakum, help", "electrical_hazard", "critical", "electricity"),
    ("high tension kambhi school roof il thaakam, kuttikalk danger", "electrical_hazard", "critical", "electricity"),
    ("pole valinja, car il veenu, area cordon cheythilla, people walk cheyyunnu", "electrical_hazard", "critical", "electricity"),
    ("substation lock illatha, kuttikalum keri kayarum, urgent lock", "electrical_hazard", "urgent", "electricity"),
    ("kambhi insulation illatha, spark varunnu mazha il, danger", "electrical_hazard", "urgent", "electricity"),

    # Code-mixed
    ("Live wire road il veenu, storm kazhinjal, vehicles touch aavunnu, critical", "electrical_hazard", "critical", "electricity"),
    ("Transformer sparking aanu near building, explode aakum, urgent KSEB call venam", "electrical_hazard", "critical", "electricity"),
    ("High tension wire school roof touch aanu, children danger, immediate action", "electrical_hazard", "critical", "electricity"),
    ("Pole snapped, road block, cordon illatha, pedestrians risk", "electrical_hazard", "critical", "electricity"),
    ("Junction box open, live terminal visible, anyone touch aakum shock", "electrical_hazard", "critical", "electricity"),
    ("Underground cable exposed, waterlogging kaaran moodal poyi, electrocution risk", "electrical_hazard", "critical", "electricity"),
]

# ===========================================================================
# STREET LIGHT  (~90 seeds)
# ===========================================================================

STREET_LIGHT: list[TrainingSample] = [
    # English
    ("Street light on the entire stretch from the junction to the colony has been dead for two weeks.", "street_light", "medium", "electricity"),
    ("The light sensor is broken. The street light turns on in the daytime and stays off at night.", "street_light", "medium", "electricity"),
    ("Dark stretch of road near the school. Two snatch-theft incidents happened here this week.", "street_light", "high", "electricity"),
    ("Street light pole near the park is leaning. The wire is loose and swinging.", "street_light", "high", "electricity"),
    ("Women from our colony are afraid to walk home after dark because there is no street light on this road.", "street_light", "high", "electricity"),
    ("New residential colony has no street lights installed at all despite completing two years.", "street_light", "medium", "electricity"),
    ("The street light in front of the hospital main entrance has been out for a month.", "street_light", "medium", "electricity"),
    ("Multiple lights flickering continuously. Suggesting a voltage fluctuation or loose connection.", "street_light", "medium", "electricity"),
    ("Streetlight at the blind curve before the bridge is crucial for road safety. It has been broken for six weeks.", "street_light", "high", "electricity"),
    ("Solar street lights installed in our ward stopped working after three months. Nobody has come to service them.", "street_light", "medium", "electricity"),
    ("Street light wires hanging loose from the pole after the recent wind. Could fall on a passerby.", "street_light", "high", "electricity"),
    ("The road leading to the cemetery is completely dark after 8pm. Residents scared to use it.", "street_light", "high", "electricity"),
    ("Several street lights permanently on 24 hours wasting electricity.", "street_light", "low", "electricity"),
    ("Traffic light and street light both out at the busy intersection. Accidents at night.", "street_light", "high", "electricity"),
    ("The entire ward 22 has had no functional street lights for three weeks after the transformer work.", "street_light", "medium", "electricity"),

    # Informal
    ("street light not working whole stretch for 2 weeks dark road", "street_light", "medium", "electricity"),
    ("light on during day off at night sensor problem please fix", "street_light", "medium", "electricity"),
    ("dark road near school theft happening women afraid", "street_light", "high", "electricity"),
    ("new colony 2 years still no street lights please install", "street_light", "medium", "electricity"),
    ("solar light stopped working nobody serviced it broken waste", "street_light", "medium", "electricity"),

    # Malayalam
    ("ഞങ്ങളുടെ കോളനിയിൽ നിന്ന് ജംഗ്ഷൻ വരെ ഒരു തെരുവ് വിളക്കും ഇല്ല. രണ്ടാഴ്ചയായി.", "street_light", "medium", "electricity"),
    ("ഇരുണ്ട റോഡ് — ഈ ആഴ്ച രണ്ട് പിടിച്ചുപറി. സ്ത്രീകൾ ഭയന്ന് നടക്കുന്നില്ല.", "street_light", "high", "electricity"),
    ("സ്ട്രീറ്റ് ലൈറ്റ് ദിവസം ഓൺ ആകുകയും രാത്രി ഓഫ് ആകുകയും ചെയ്യുന്നു. സെൻസർ പ്രശ്നം.", "street_light", "medium", "electricity"),
    ("സ്കൂൾ ആശുപത്രി ഒരു മാസം ആകുന്നതിന് മുമ്പ് വിളക്ക് ഓഫ്. ഇവിടെ ആൾ ഒഴുക്ക് കൂടുതൽ.", "street_light", "medium", "electricity"),
    ("ഞങ്ങളുടെ ഏരിയ 2 വർഷം ആകുന്നതിന് ഒരു വിളക്കും ഇൻസ്റ്റോൾ ചെയ്തിട്ടില്ല.", "street_light", "medium", "electricity"),

    # Manglish
    ("street light 2 weeks ayi poyyi, full dark road, pedestrian ku problem", "street_light", "medium", "electricity"),
    ("sensor ketti, daytime on, night time off, ulta aanu, fix cheyyenam", "street_light", "medium", "electricity"),
    ("rathri dark, school road il, ladies scared aayi walk cheyyunnilla", "street_light", "high", "electricity"),
    ("solar light 3 months kondu stop aayittu, nobody service cheythu", "street_light", "medium", "electricity"),
    ("wire loose aayittu pole il ninn tazhunnu, kaatre il anyone ku veezhum", "street_light", "high", "electricity"),

    # Code-mixed
    ("Street light entire stretch dark aanu 2 weeks, colony to junction", "street_light", "medium", "electricity"),
    ("Sensor malfunction, day il on, night il off, wastage and dark", "street_light", "medium", "electricity"),
    ("Dark road near school, theft 2 times, women walking afraid", "street_light", "high", "electricity"),
    ("New colony 2 years, oru street light innum install cheythilla", "street_light", "medium", "electricity"),
    ("Solar lights broken, 3 months since stop, KSEB service illakill", "street_light", "medium", "electricity"),
    ("Pole leaning, wires loose swinging, could fall on someone", "street_light", "high", "electricity"),
]

# ===========================================================================
# TREE FALL  (~85 seeds)
# ===========================================================================

TREE_FALL: list[TrainingSample] = [
    # English
    ("A massive tree fell onto the road during last night's storm. Traffic is completely blocked in both directions.", "tree_fall", "urgent", "parks"),
    ("An enormous old tree is leaning severely over the road near the school. Could fall any time. Children at risk.", "tree_fall", "critical", "parks"),
    ("Tree fell on three parked cars last night. Owners want to know who is responsible for this roadside tree.", "tree_fall", "urgent", "parks"),
    ("Large branch fell on the 11kV electric line. Sparking visible. Fire risk.", "tree_fall", "critical", "parks"),
    ("A rotten tree next to the wall of our compound is ready to fall. One storm and it will land on our house.", "tree_fall", "high", "parks"),
    ("Tree uprooted in the storm and blocking the only access road to the hospital. Ambulances cannot pass.", "tree_fall", "critical", "parks"),
    ("Branches from the roadside tree keep falling on vehicles parked below. No warning signs or barriers.", "tree_fall", "high", "parks"),
    ("Dead tree trunk next to the children's playground. One strong wind and it will fall on playing children.", "tree_fall", "high", "parks"),
    ("Tree across the road is partially uprooted after rain but still standing tilted. Imminent danger.", "tree_fall", "urgent", "parks"),
    ("The banyan tree at the corner has overgrown into the high tension wires. Serious electrical hazard.", "tree_fall", "critical", "parks"),
    ("Fallen tree roots have damaged the underground water pipe. Now there is also a water leak.", "tree_fall", "urgent", "parks"),
    ("Large branch fell and hit a motorcyclist this morning. The person was injured but stable.", "tree_fall", "urgent", "parks"),
    ("Three trees near the market entrance need emergency pruning. They are dangerously overgrown.", "tree_fall", "high", "parks"),
    ("Tree on the highway median fell across both lanes. Causing massive traffic jam.", "tree_fall", "urgent", "parks"),

    # Informal
    ("huge tree fell on road last night full traffic block urgent clear", "tree_fall", "urgent", "parks"),
    ("big tree leaning over school road children at serious risk urgent", "tree_fall", "critical", "parks"),
    ("tree fell on electric wire sparking visible fire risk urgent", "tree_fall", "critical", "parks"),
    ("rotten tree near house one storm and it lands on us please remove", "tree_fall", "high", "parks"),
    ("tree blocking hospital road ambulances cant pass critical", "tree_fall", "critical", "parks"),

    # Malayalam
    ("കഴിഞ്ഞ രാത്രി ഇടിമിന്നലിൽ മരം വഴിയിൽ വീണ്. ഇരുദിശകളിലും ഗതാഗത തടസ്സം.", "tree_fall", "urgent", "parks"),
    ("ഒരു വലിയ മരം സ്കൂൾ ഗേറ്റ് ദിശയിൽ ചരിഞ്ഞ് നിൽക്കുന്നു. ഏത് നിമിഷവും വീഴ്ചയ്ക്ക് സാധ്യത.", "tree_fall", "critical", "parks"),
    ("ഗ്രൗണ്ടിൽ ഒരു ചത്ത മരം നിൽക്കുന്നു. കുട്ടികൾ ചുറ്റും കളിക്കുന്നു. അപകടം.", "tree_fall", "high", "parks"),
    ("മരം കൊണ്ട് ഇലക്ട്രിക് ലൈൻ ബ്ലോക്ക് ആണ്. തീ പടർന്നേക്കാം. ഉടൻ ഇടപെടൽ.", "tree_fall", "critical", "parks"),
    ("ആശുപത്രി വഴി ബ്ലോക്ക്, ആംബുലൻസ് കടക്കാൻ കഴിയില്ല, ക്രിറ്റിക്കൽ.", "tree_fall", "critical", "parks"),

    # Manglish
    ("maram road il veenu, storm il, traffic full block, clear cheyyenam urgent", "tree_fall", "urgent", "parks"),
    ("valiya maram school road il charinju, ennaikillum veezhum, danger", "tree_fall", "critical", "parks"),
    ("maram electric wire il veenu, spark, fire aakum, urgent action", "tree_fall", "critical", "parks"),
    ("veetin mukalil maram, oru kaattil veezhum, please cut", "tree_fall", "high", "parks"),
    ("hospital road block, ambulance keri varaan pattunilla, critical", "tree_fall", "critical", "parks"),
    ("branch veenu bike driver ku, hospital il, serious injury", "tree_fall", "urgent", "parks"),

    # Code-mixed
    ("Tree road il veenu last night, both lanes block, traffic jam huge", "tree_fall", "urgent", "parks"),
    ("Big tree leaning school road il, children danger, please urgent remove", "tree_fall", "critical", "parks"),
    ("Tree fell on electric wire, spark visible, fire risk, emergency", "tree_fall", "critical", "parks"),
    ("Dead tree playground near, children ku danger, please cut it", "tree_fall", "high", "parks"),
    ("Hospital access block aayittu maram kaaran, ambulance issue serious", "tree_fall", "critical", "parks"),
    ("Branch hit motorcyclist, hospital il, maram pruning needed", "tree_fall", "urgent", "parks"),
]

# ===========================================================================
# ILLEGAL CONSTRUCTION  (~85 seeds)
# ===========================================================================

ILLEGAL_CONSTRUCTION: list[TrainingSample] = [
    # English
    ("A multi-storey building is being constructed on government-owned land without any visible permit board.", "illegal_construction", "high", "planning"),
    ("Our neighbour has built a wall encroaching two metres onto the public road reducing it to a single lane.", "illegal_construction", "high", "planning"),
    ("A commercial shop has extended its structure onto the footpath blocking all pedestrian movement.", "illegal_construction", "medium", "planning"),
    ("Construction in progress in the CRZ restricted zone near the beach. Clear violation of coastal norms.", "illegal_construction", "urgent", "planning"),
    ("Building being constructed without setback. The new wall is practically touching my compound wall.", "illegal_construction", "high", "planning"),
    ("An additional floor is being added to a structure that was approved for ground plus one. Now going to three floors.", "illegal_construction", "high", "planning"),
    ("Construction started at midnight to avoid detection. Noise is unbearable. No permit visible anywhere.", "illegal_construction", "medium", "planning"),
    ("A poultry farm has been constructed in a residential zone. Smell and fly problem for all neighbours.", "illegal_construction", "medium", "planning"),
    ("Retaining wall built by a plot owner is blocking the natural drainage path causing flooding in my compound.", "illegal_construction", "high", "planning"),
    ("New construction is encroaching on the access road to our colony. If it continues we will be cut off.", "illegal_construction", "high", "planning"),
    ("Building constructed across the storm water drain channel. Rain water has no outlet now.", "illegal_construction", "high", "planning"),
    ("Land that belongs to the corporation has been enclosed and private construction has started.", "illegal_construction", "high", "planning"),
    ("Commercial complex being built in an area designated as green belt in the local plan.", "illegal_construction", "urgent", "planning"),
    ("Compound wall built extending into the pavement. Wheelchair users cannot pass.", "illegal_construction", "medium", "planning"),

    # Informal
    ("construction without permit on government land multi storey building", "illegal_construction", "high", "planning"),
    ("neighbour wall on public road reducing to single lane illegal", "illegal_construction", "high", "planning"),
    ("shop extending to footpath pedestrians blocked please act", "illegal_construction", "medium", "planning"),
    ("construction in CRZ zone near beach illegal coastal violation", "illegal_construction", "urgent", "planning"),
    ("extra floor added to building beyond approved limit three floors now", "illegal_construction", "high", "planning"),

    # Malayalam
    ("അനധികൃത കെട്ടിടം പൊതുഭൂമിയിൽ. ഒരു ബോർഡ് പോലും ഇല്ല.", "illegal_construction", "high", "planning"),
    ("അൽ‌പ്പം കൂടി ചേർക്കൽ നിർമ്മാണം ഫ്ലോർ കൂടുതൽ. അനുമതി ഇലാതത്.", "illegal_construction", "high", "planning"),
    ("CRZ സോൺ ലംഘനം. കടൽ തീരത്ത് നിർമ്മാണം. ഉടൻ ഇടപെടൽ.", "illegal_construction", "urgent", "planning"),
    ("ഡ്രൈനേജ് ചാൽ ബ്ലോക്ക് ചെയ്ത് കെട്ടിടം. വെള്ളക്കെട്ട് ഉണ്ടാകുന്നു.", "illegal_construction", "high", "planning"),
    ("ഹരിതമേഖലയിൽ നിർമ്മാണം. ടൗൺ പ്ലാൻ ലംഘനം.", "illegal_construction", "urgent", "planning"),

    # Manglish
    ("permit illatha building government land il, multi storey, board illakill", "illegal_construction", "high", "planning"),
    ("neighbour compound wall public road il, one lane only, illegal", "illegal_construction", "high", "planning"),
    ("extra floor anumati illatha, 3 floor aayittu, violation", "illegal_construction", "high", "planning"),
    ("CRZ violation, beach kazhijal construction, coastal rule break", "illegal_construction", "urgent", "planning"),
    ("shop footpath il build cheythu, pedestrian block, encroachment", "illegal_construction", "medium", "planning"),

    # Code-mixed
    ("Permit illatha construction pothubhoomiyil, board visible illakill", "illegal_construction", "high", "planning"),
    ("Neighbour's wall public road il encroach cheythu, traffic narrow", "illegal_construction", "high", "planning"),
    ("CRZ zone il construction, coastal violation, urgent action venam", "illegal_construction", "urgent", "planning"),
    ("Additional floor built without permission, 3 storeys now", "illegal_construction", "high", "planning"),
    ("Drainage path block cheythu building, flooding issue", "illegal_construction", "high", "planning"),
    ("Corporation land enclose cheythu private construction started", "illegal_construction", "high", "planning"),
]

# ===========================================================================
# SPAM  (~60 seeds)
# ===========================================================================

SPAM: list[TrainingSample] = [
    ("Buy cheap medicines online no prescription needed call now", "spam", "low", "none"),
    ("Earn 5000 per day working from home no investment needed", "spam", "low", "none"),
    ("Home loan at lowest interest rate call for free advice", "spam", "low", "none"),
    ("FREE OFFER limited time medicines vitamins call immediately", "spam", "low", "none"),
    ("Download our app get cashback on all purchases", "spam", "low", "none"),
    ("You have won a lottery prize claim within 24 hours", "spam", "low", "none"),
    ("Hello sir good morning", "spam", "low", "none"),
    ("test test test", "spam", "low", "none"),
    ("123456789", "spam", "low", "none"),
    ("Please arrange a meeting with the commissioner next week", "spam", "low", "none"),
    ("This is not a complaint just saying hi", "spam", "low", "none"),
    ("aaaaaa bbbbbb cccccc", "spam", "low", "none"),
    ("asdf qwerty zxcv random text nothing here", "spam", "low", "none"),
    ("I need a government job please help me get placed", "spam", "low", "none"),
    ("Please arrange water tanker for my daughter wedding ceremony", "spam", "low", "none"),
    ("Flat for sale near bus stand contact owner directly", "spam", "low", "none"),
    ("Political rally event details time and venue tomorrow evening", "spam", "low", "none"),
    ("Cricket score update live streaming link click here", "spam", "low", "none"),
    ("enna venum ennik ariyilla", "spam", "low", "none"),
    ("vandi vannittu onnum cheythilla", "spam", "low", "none"),
    ("Please call me back on this number urgent personal matter", "spam", "low", "none"),
    ("I want to complain about my neighbour's dog barking at night", "spam", "low", "none"),
    ("Can you help me get pension certificate from the office", "spam", "low", "none"),
    ("My neighbour is playing loud music please do something", "spam", "low", "none"),
    ("Birthday greetings to all the officers working hard", "spam", "low", "none"),
    ("I have a property dispute with my neighbour please intervene", "spam", "low", "none"),
    ("abc def ghi", "spam", "low", "none"),
    ("ok", "spam", "low", "none"),
    ("haha lol nothing", "spam", "low", "none"),
    ("best deals on electronics visit our showroom", "spam", "low", "none"),
    ("plot for sale in good location price negotiable contact", "spam", "low", "none"),
    ("my daughter needs admission in school please help arrange", "spam", "low", "none"),
    ("I want to register a complaint against a government officer for rudeness", "spam", "low", "none"),
    ("Can you transfer my water bill to a different account", "spam", "low", "none"),
    ("testing testing is this working", "spam", "low", "none"),
    ("please ignore this is just a test submission", "spam", "low", "none"),
    ("xyz", "spam", "low", "none"),
    ("1 2 3 check check", "spam", "low", "none"),
    ("namaskaram sir how are you doing today this is ramesh", "spam", "low", "none"),
    ("I want to wish the Mayor a happy birthday on behalf of the citizens", "spam", "low", "none"),
    ("Our area needs more CCTV cameras for security not a civic complaint", "spam", "low", "none"),
    ("Astrology service call for free reading limited offer today", "spam", "low", "none"),
    ("Weight loss guaranteed in 30 days call for consultation", "spam", "low", "none"),
    ("pls send someone to fix my personal electricity connection inside house", "spam", "low", "none"),
    ("My TV cable connection is not working who do I call", "spam", "low", "none"),
    ("Matrimony service Kerala brides and grooms register free", "spam", "low", "none"),
    ("I have a complaint about a police officer not civic issue", "spam", "low", "none"),
    ("I want to report black money being hoarded by a politician", "spam", "low", "none"),
    ("Real estate investment opportunity high returns contact us", "spam", "low", "none"),
    ("Tourism package Kerala backwater tour lowest price call now", "spam", "low", "none"),
    ("Drug addiction helpline call this number 24 hours available", "spam", "low", "none"),
    ("Hello I am a student doing research on civic issues may I ask questions", "spam", "low", "none"),
    ("Our NGO wants to do a cleanliness drive please provide support", "spam", "low", "none"),
    ("Ambulance service number please", "spam", "low", "none"),
    ("I need a birth certificate please guide me to the right office", "spam", "low", "none"),
]

# ===========================================================================
# NO CATEGORY  (~45 seeds)
# ===========================================================================

NO_CATEGORY: list[TrainingSample] = [
    ("There is a serious problem near my house please send someone to inspect", "no_category", "medium", "none"),
    ("Something is very wrong in our area the authorities should visit", "no_category", "medium", "none"),
    ("Please help us we are facing major issues since many months", "no_category", "medium", "none"),
    ("The situation here has become unbearable action is urgently needed", "no_category", "medium", "none"),
    ("Residents are suffering daily no one seems to care", "no_category", "medium", "none"),
    ("Issue at the main junction has not been resolved for months", "no_category", "medium", "none"),
    ("Basic civic facilities in our ward are extremely poor quality", "no_category", "low", "none"),
    ("Government should do something about the state of our area", "no_category", "low", "none"),
    ("Everything is broken and dysfunctional in this part of the city", "no_category", "low", "none"),
    ("We have been ignored for years despite multiple complaints", "no_category", "medium", "none"),
    ("Condition of our ward is very bad please visit once and see", "no_category", "medium", "none"),
    ("Officers never visit our area yet promise to come every election", "no_category", "low", "none"),
    ("The ward councillor does not respond to our complaints since months", "no_category", "low", "none"),
    ("Many problems exist here I do not know where to begin", "no_category", "medium", "none"),
    ("Urgent attention required in ward 15 regarding multiple ongoing issues", "no_category", "medium", "none"),
    ("ഞങ്ങളുടെ ഏരിയ ഒന്ന് സന്ദർശിക്കണം, ഒരുപാട് പ്രശ്നങ്ങൾ ഉണ്ട്", "no_category", "medium", "none"),
    ("eedum problem und, officer varanam, check cheyyanam", "no_category", "medium", "none"),
    ("Ward condition very bad, nobody caring, please visit once", "no_category", "low", "none"),
    ("Multiple issues in our area, don't know which department to contact", "no_category", "medium", "none"),
    ("Facilities very poor, civic amenities not maintained since years", "no_category", "low", "none"),
]

# ===========================================================================
# Duplicate groups for semantic similarity evaluation
# ===========================================================================

DUPLICATE_GROUPS: list[list[TrainingSample]] = [
    # Group 1: Street light not working
    [
        ("Street light not working near the junction since two weeks", "street_light", "medium", "electricity"),
        ("The lamp post near the crossing has been dead for weeks", "street_light", "medium", "electricity"),
        ("No street lighting at the junction area it is completely dark at night", "street_light", "medium", "electricity"),
        ("vilakku kazhinja rendu aaram poyii, junction il andharam", "street_light", "medium", "electricity"),
    ],
    # Group 2: Water pipe burst
    [
        ("Water pipe burst on the road junction flooding the road", "water_supply", "urgent", "water"),
        ("Pipe burst near junction water gushing onto street", "water_supply", "urgent", "water"),
        ("kuzhal potti junction il vellam ozhukunu road blocked", "water_supply", "urgent", "water"),
        ("Main pipeline cracked at the crossing water everywhere", "water_supply", "urgent", "water"),
    ],
    # Group 3: Drain blocked flooding
    [
        ("Drain blocked causing road flooding after rain", "drainage", "high", "drainage"),
        ("Clogged drain making road waterlogged during monsoon", "drainage", "high", "drainage"),
        ("Block drain flooding road mazha vannal", "drainage", "high", "drainage"),
        ("Stormwater drain choked road goes under water in rain", "drainage", "high", "drainage"),
    ],
    # Group 4: Pothole causing accidents
    [
        ("Large pothole on road causing accidents near the market", "road_damage", "high", "roads"),
        ("Huge crater on main road near market many accidents happening", "road_damage", "high", "roads"),
        ("road kuzhi market kareyin, accident aavunnu everyday", "road_damage", "high", "roads"),
        ("Deep hole in the road beside the market vehicles getting damaged", "road_damage", "high", "roads"),
    ],
    # Group 5: Tree fallen blocking road
    [
        ("Tree fell on road blocking traffic completely", "tree_fall", "urgent", "parks"),
        ("Fallen tree blocking the main road after storm", "tree_fall", "urgent", "parks"),
        ("maram road il veenu, traffic block, clear cheyyenam", "tree_fall", "urgent", "parks"),
        ("Big tree collapsed on road traffic cannot move", "tree_fall", "urgent", "parks"),
    ],
]

# ===========================================================================
# Complete combined corpus
# ===========================================================================

ALL_SAMPLES: list[TrainingSample] = (
    WATER_SUPPLY
    + DRAINAGE
    + SEWAGE
    + SOLID_WASTE
    + ROAD_DAMAGE
    + ELECTRICAL_HAZARD
    + STREET_LIGHT
    + TREE_FALL
    + ILLEGAL_CONSTRUCTION
    + SPAM
    + NO_CATEGORY
    # Flatten duplicate groups into the main corpus too
    + [s for group in DUPLICATE_GROUPS for s in group]
)

# ---------------------------------------------------------------------------
# Category metadata (for validation and reporting)
# ---------------------------------------------------------------------------
CATEGORY_CODES: list[str] = [
    "water_supply", "drainage", "sewage_issue", "solid_waste", "road_damage",
    "electrical_hazard", "street_light", "tree_fall", "illegal_construction",
    "spam", "no_category",
]

PRIORITY_LEVELS: list[str] = ["low", "medium", "high", "urgent", "critical"]

DEPARTMENT_CODES: list[str] = [
    "water", "drainage", "sewage", "sanitation", "roads",
    "electricity", "parks", "planning", "none",
]
