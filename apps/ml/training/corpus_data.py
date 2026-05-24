"""apps/ml/training/corpus_data.py

Multilingual civic complaint training corpus for the TVMC grievance platform.

Languages represented
---------------------
* English           — standard civic complaint register
* Malayalam         — native script, formal and colloquial
* Manglish          — transliterated Malayalam in Roman script
* Mixed             — code-switched English + Malayalam/Manglish (most realistic)

Category codes align with the production rule engine in analyzer.py:
    water_supply, drainage, sewage_issue, solid_waste, road_damage,
    electrical_hazard, street_light, tree_fall, illegal_construction

Additional labels:
    spam              — irrelevant, promotional, abusive, or nonsensical text
    no_category       — genuine complaints that lack enough context to classify

Priority levels: low, medium, high, urgent, critical

Each template is a raw text string. generate_corpus.py expands these into
a labeled CSV by filling placeholder slots and applying random variation.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
# Each entry: (text, category_code, priority, department_code)
# department_code is the canonical routing target used as a label for the
# department prediction model.
# ---------------------------------------------------------------------------
TrainingSample = tuple[str, str, str, str]

# ---------------------------------------------------------------------------
# WATER SUPPLY
# ---------------------------------------------------------------------------
WATER_SUPPLY: list[TrainingSample] = [
    # English
    ("No water supply in our area for the past three days", "water_supply", "high", "water"),
    ("Water pipe burst near the market road causing flooding", "water_supply", "urgent", "water"),
    ("Low water pressure in taps since last week", "water_supply", "medium", "water"),
    ("Broken water pipe on the main road leaking continuously", "water_supply", "high", "water"),
    ("Water supply has been cut off without any prior notice", "water_supply", "high", "water"),
    ("Contaminated water coming from the tap, brownish colour", "water_supply", "urgent", "water"),
    ("Water meter is not working properly for months", "water_supply", "medium", "water"),
    ("The overhead water tank is overflowing and wasting water daily", "water_supply", "medium", "water"),
    ("Water pipe leaking near the school compound wall", "water_supply", "high", "water"),
    ("No drinking water supply to our ward for two days", "water_supply", "high", "water"),
    ("Rust coloured water coming from tap, not drinkable", "water_supply", "urgent", "water"),
    ("Water supply pipe broken under the road near the temple junction", "water_supply", "urgent", "water"),
    ("Pipe burst at main junction water is flowing onto the road", "water_supply", "urgent", "water"),
    ("Water pipe near my house is cracked and leaking non-stop", "water_supply", "high", "water"),
    ("Our street gets water only for 30 minutes each day", "water_supply", "medium", "water"),
    # Malayalam (native script)
    ("ഞങ്ങളുടെ പ്രദേശത്ത് മൂന്ന് ദിവസമായി വെള്ളം ഇല്ല", "water_supply", "high", "water"),
    ("ജലവിതരണ പൈപ്പ് പൊട്ടി വഴിയിൽ വെള്ളം ഒഴുകുന്നു", "water_supply", "urgent", "water"),
    ("ടാപ്പിൽ നിന്ന് മലിനജലം വരുന്നു, കുടിക്കാൻ കഴിയില്ല", "water_supply", "urgent", "water"),
    ("ഈ ഏരിയയിൽ വെള്ളം വളരെ കുറഞ്ഞ പ്രഷർ ആണ്", "water_supply", "medium", "water"),
    ("ജലപൈപ്പ് ലൈൻ തകർന്നു, അടിയന്തര നടപടി ആവശ്യം", "water_supply", "high", "water"),
    # Manglish
    ("vellam vannittu moonnu divasam ayi, oru thulli vellam illa", "water_supply", "high", "water"),
    ("kuzhal potti vellam road il ozhukunu, urgent aanu", "water_supply", "urgent", "water"),
    ("tap il vellam varunnilla, pressure illa", "water_supply", "medium", "water"),
    ("malinam vellam vannu, kudikkan pattilla", "water_supply", "urgent", "water"),
    ("water pipe leak aanu school compound il", "water_supply", "high", "water"),
    ("vellam supply cut cheythu, notice innum kittiyilla", "water_supply", "high", "water"),
    ("kuzhal adangi water flow blocked aanu", "water_supply", "medium", "water"),
    # Mixed
    ("Water supply illatha kaaran 3 days ayi, please help", "water_supply", "high", "water"),
    ("Pipe burst near junction, vellam road il ozhukunu", "water_supply", "urgent", "water"),
    ("Tap il vannu vellam brown colour, contaminated aanennu thonaanu", "water_supply", "urgent", "water"),
    ("No water for 2 days in ward 12, pipe leak undo?", "water_supply", "high", "water"),
    ("Water line cracked near school, kids having problem", "water_supply", "high", "water"),
    ("Supply pipe broken, vellam waste aavunnu, eppo fix cheyum?", "water_supply", "high", "water"),
]

# ---------------------------------------------------------------------------
# DRAINAGE
# ---------------------------------------------------------------------------
DRAINAGE: list[TrainingSample] = [
    # English
    ("Blocked drain causing flooding on the road after rain", "drainage", "high", "drainage"),
    ("Open drainage channel in front of my house is overflowing", "drainage", "medium", "drainage"),
    ("Stormwater drain blocked with garbage, water not flowing", "drainage", "high", "drainage"),
    ("Drain cover missing near the bus stop, dangerous for pedestrians", "drainage", "high", "drainage"),
    ("The roadside drain has not been cleaned for months", "drainage", "medium", "drainage"),
    ("Drainage overflow flooding into residential area during rain", "drainage", "urgent", "drainage"),
    ("New construction has blocked the natural drain path", "drainage", "high", "drainage"),
    ("Clogged drain near the market is causing foul smell", "drainage", "medium", "drainage"),
    ("Open manhole in drain, very dangerous at night", "drainage", "urgent", "drainage"),
    ("Drain canal near school flooded up to the entrance", "drainage", "high", "drainage"),
    ("Rainwater not draining from road, flooding the entire street", "drainage", "high", "drainage"),
    ("The drain pipe under the road is broken, causing sinkhole", "drainage", "urgent", "drainage"),
    # Malayalam
    ("ഡ്രൈനേജ് ലൈൻ ബ്ലോക്ക് ആണ്, മഴ വന്നാൽ വഴിയൊക്കെ വെള്ളം", "drainage", "high", "drainage"),
    ("ഓടച്ചനൽ നിറഞ്ഞ് ഒഴുകുന്നു, ദുർഗന്ധം വ്യാപകമാണ്", "drainage", "medium", "drainage"),
    ("ഡ്രൈനേജ് കവർ ഇല്ല, ആളുകൾക്ക് അപകടകരം", "drainage", "high", "drainage"),
    # Manglish
    ("drain block aanu, mazha vandal road il vellam nilkkunnu", "drainage", "high", "drainage"),
    ("drainage overflow aavunnu, veetil vellam keri", "drainage", "urgent", "drainage"),
    ("oda channal full aanu, durmgandam vannu", "drainage", "medium", "drainage"),
    ("drain cover illatha kuzhi und, night il dangerous", "drainage", "high", "drainage"),
    # Mixed
    ("Drain blocked aayi flooding, ward 7 il problem serious aanu", "drainage", "high", "drainage"),
    ("Open drain near school, kids ku dangerous, drain cover venam", "drainage", "high", "drainage"),
    ("Rain kazhinjal road il vellam nilkkunnu, drain block", "drainage", "high", "drainage"),
    ("Drainage system overflow, residential area flooded aayittu", "drainage", "urgent", "drainage"),
]

# ---------------------------------------------------------------------------
# SEWAGE
# ---------------------------------------------------------------------------
SEWAGE: list[TrainingSample] = [
    # English
    ("Sewage water overflowing onto the road near the temple", "sewage_issue", "urgent", "sewage"),
    ("Blocked sewer line causing sewage backup in our homes", "sewage_issue", "urgent", "sewage"),
    ("Manhole overflowing with sewage near the bus stand", "sewage_issue", "urgent", "sewage"),
    ("Sewage treatment plant is leaking waste into the river", "sewage_issue", "critical", "sewage"),
    ("Strong sewage smell from the underground drain near park", "sewage_issue", "high", "sewage"),
    ("The underground sewer line is damaged causing foul odour", "sewage_issue", "high", "sewage"),
    ("Sewage pipe burst under the road near school", "sewage_issue", "urgent", "sewage"),
    ("Raw sewage visible on the street, health hazard", "sewage_issue", "urgent", "sewage"),
    ("Sewage from next building being discharged onto public road", "sewage_issue", "high", "sewage"),
    ("Septic tank overflowing, sewage coming out near gate", "sewage_issue", "urgent", "sewage"),
    # Malayalam
    ("മലിനജലം വഴിയിൽ ഒഴുകുന്നു, ആരോഗ്യ പ്രശ്നം ഉണ്ടാകും", "sewage_issue", "urgent", "sewage"),
    ("സ്യൂവേജ് പൈപ്പ് പൊട്ടി ദുർഗന്ധം പരക്കുന്നു", "sewage_issue", "urgent", "sewage"),
    ("മാൻഹോൾ നിറഞ്ഞ് മലിനജലം തെരുവിൽ", "sewage_issue", "urgent", "sewage"),
    # Manglish
    ("sewage water road il ozhukunu, urgent action venam", "sewage_issue", "urgent", "sewage"),
    ("manhole overflow aanu, malinjalam nira nira", "sewage_issue", "urgent", "sewage"),
    ("sewer line potti, veetil durmgandam", "sewage_issue", "high", "sewage"),
    ("septic tank niranju, sewage puram varunnu", "sewage_issue", "urgent", "sewage"),
    # Mixed
    ("Sewage overflow near market, public health risk aanu", "sewage_issue", "urgent", "sewage"),
    ("Manhole blocked, sewage road il varunnu, please send team", "sewage_issue", "urgent", "sewage"),
    ("Sewer pipe broken, durmgandam kaaran aayi, health hazard", "sewage_issue", "high", "sewage"),
    ("Septic tank overflow aavunnu, street il vannu, emergency", "sewage_issue", "urgent", "sewage"),
]

# ---------------------------------------------------------------------------
# SOLID WASTE / GARBAGE
# ---------------------------------------------------------------------------
SOLID_WASTE: list[TrainingSample] = [
    # English
    ("Garbage not collected for the past week in our area", "solid_waste", "medium", "sanitation"),
    ("Overflowing garbage bin near the vegetable market", "solid_waste", "medium", "sanitation"),
    ("Illegal dumping of construction waste on public land", "solid_waste", "high", "sanitation"),
    ("Dead animal on the road not removed for two days", "solid_waste", "high", "sanitation"),
    ("Waste collection vehicle has not come for 10 days", "solid_waste", "medium", "sanitation"),
    ("Burning garbage near residential area causing smoke hazard", "solid_waste", "high", "sanitation"),
    ("Garbage dump in open ground attracts mosquitoes and rodents", "solid_waste", "high", "sanitation"),
    ("Public bin is full and overflowing near the bus stop", "solid_waste", "medium", "sanitation"),
    ("Plastic and organic waste mixed, not segregated at source", "solid_waste", "low", "sanitation"),
    ("Waste dumped into the canal near the bridge", "solid_waste", "high", "sanitation"),
    ("Biomedical waste dumped near residential area, dangerous", "solid_waste", "urgent", "sanitation"),
    ("Large pile of garbage blocking the pedestrian footpath", "solid_waste", "medium", "sanitation"),
    # Malayalam
    ("ഈ ആഴ്ച മാലിന്യം ശേഖരിക്കാൻ ആളുകൾ വന്നില്ല", "solid_waste", "medium", "sanitation"),
    ("കൂമ്പാരം ഇട്ട മാലിന്യം കത്തിക്കുന്നു, ദോഷകരം", "solid_waste", "high", "sanitation"),
    ("ചത്ത മൃഗം വഴിയിൽ കിടക്കുന്നു, നീക്കം ചെയ്തിട്ടില്ല", "solid_waste", "high", "sanitation"),
    # Manglish
    ("garbage collection vannittu onnum illa, bin niranju", "solid_waste", "medium", "sanitation"),
    ("malinya kuppayam road block cheyyunnu", "solid_waste", "medium", "sanitation"),
    ("chatta janangal kanal il idum, water polute aavunnu", "solid_waste", "high", "sanitation"),
    ("garbage ithinte smell, mosquito full aanu", "solid_waste", "medium", "sanitation"),
    # Mixed
    ("Garbage collection illa last week mutal, bin overflow aayittu", "solid_waste", "medium", "sanitation"),
    ("Waste dump near school, children ku health risk aanu", "solid_waste", "high", "sanitation"),
    ("Dead dog road il kida, two days ayi, neekkam venam", "solid_waste", "high", "sanitation"),
    ("Open burning of garbage, smoke varunnu, problem aanu", "solid_waste", "high", "sanitation"),
]

# ---------------------------------------------------------------------------
# ROAD DAMAGE
# ---------------------------------------------------------------------------
ROAD_DAMAGE: list[TrainingSample] = [
    # English
    ("Large pothole on the main road causing accidents", "road_damage", "high", "roads"),
    ("Road full of potholes, very dangerous for vehicles", "road_damage", "high", "roads"),
    ("Newly laid road has already developed cracks", "road_damage", "medium", "roads"),
    ("Footpath broken and uneven, pedestrians falling down", "road_damage", "medium", "roads"),
    ("Road sinking near the bridge, structural problem", "road_damage", "urgent", "roads"),
    ("Severe road damage after monsoon, multiple potholes", "road_damage", "high", "roads"),
    ("No road markings or speed bumps on the school zone", "road_damage", "medium", "roads"),
    ("Broken speed breaker causing accidents near the junction", "road_damage", "high", "roads"),
    ("Road maintenance not done for years, completely damaged", "road_damage", "high", "roads"),
    ("Metal sheet sticking out of the road, tyre puncture risk", "road_damage", "high", "roads"),
    ("Road to our residential colony not tarred yet", "road_damage", "medium", "roads"),
    ("Deep crater on the road, two-wheelers are at risk", "road_damage", "high", "roads"),
    ("Footpath encroached and damaged near the shopping complex", "road_damage", "medium", "roads"),
    # Malayalam
    ("റോഡിൽ വലിയ കുഴിയുണ്ട്, അപകടം ഉണ്ടാകുന്നു", "road_damage", "high", "roads"),
    ("മഴക്കാലം കഴിഞ്ഞ് റോഡ് പൂർണ്ണമായും നശിച്ചു", "road_damage", "high", "roads"),
    ("ഫുട്ട്‌പാത്ത് പൊട്ടി, ആൾക്കാർ വീഴുന്നു", "road_damage", "medium", "roads"),
    # Manglish
    ("road il valiya kuzhi und, accident aavunnu", "road_damage", "high", "roads"),
    ("pothole kaaran bike valiyathu aayittu, urgent fix venam", "road_damage", "high", "roads"),
    ("road crack aayittu, mazha il vellam niranju", "road_damage", "medium", "roads"),
    ("speed breaker broken aanu, vehicle control pattunilla", "road_damage", "high", "roads"),
    # Mixed
    ("Road il pothole, 2 accidents ithu kaaran last month", "road_damage", "high", "roads"),
    ("Monsoon kazhinjal road fully damaged, urgent repair venam", "road_damage", "high", "roads"),
    ("Junction il road sinking, engineers varanam immediately", "road_damage", "urgent", "roads"),
    ("School front il speed breaker illathathu kaaran accident", "road_damage", "high", "roads"),
    ("Tar road poyi, kuzhi niranja road, vehicle ku problem", "road_damage", "medium", "roads"),
]

# ---------------------------------------------------------------------------
# ELECTRICAL HAZARD
# ---------------------------------------------------------------------------
ELECTRICAL_HAZARD: list[TrainingSample] = [
    # English
    ("Live electric wire hanging low over the road, dangerous", "electrical_hazard", "critical", "electricity"),
    ("Broken electric pole leaning on the road", "electrical_hazard", "critical", "electricity"),
    ("Transformer sparking near the residential building", "electrical_hazard", "critical", "electricity"),
    ("Electric wire touching water pipe, shock risk", "electrical_hazard", "critical", "electricity"),
    ("Loose live wire on the footpath after the storm", "electrical_hazard", "critical", "electricity"),
    ("Electric pole damaged by lorry, wires hanging dangerously", "electrical_hazard", "urgent", "electricity"),
    ("Transformer making loud humming noise and leaking oil", "electrical_hazard", "urgent", "electricity"),
    ("High tension wire very close to the school building roof", "electrical_hazard", "critical", "electricity"),
    ("Electric cable on the road after the storm last night", "electrical_hazard", "critical", "electricity"),
    ("Arcing from an electrical junction box near the market", "electrical_hazard", "critical", "electricity"),
    ("Broken insulation on overhead cables, shock hazard", "electrical_hazard", "urgent", "electricity"),
    ("Substation gate open, children might enter, very dangerous", "electrical_hazard", "urgent", "electricity"),
    # Malayalam
    ("ഇലക്ട്രിക് കമ്പി റോഡിൽ വീണു കിടക്കുന്നു, അതി അപകടകരം", "electrical_hazard", "critical", "electricity"),
    ("ട്രാൻസ്ഫോർമർ കത്തുന്നു, ഉടൻ നടപടി വേണം", "electrical_hazard", "critical", "electricity"),
    ("ഇലക്ട്രിക് പോൾ ചരിഞ്ഞ് റോഡ് ബ്ലോക്ക് ആണ്", "electrical_hazard", "urgent", "electricity"),
    # Manglish
    ("electric wire road il veenu, shock aavum, urgent!", "electrical_hazard", "critical", "electricity"),
    ("kambhi thadangi road il, very dangerous", "electrical_hazard", "critical", "electricity"),
    ("transformer spark adikkunnu, parayathiri", "electrical_hazard", "critical", "electricity"),
    ("electric pole valinja, road block aanu", "electrical_hazard", "urgent", "electricity"),
    # Mixed
    ("Live wire road il veenu, accident aakum, emergency team venam", "electrical_hazard", "critical", "electricity"),
    ("Electric pole broken, wires hanging low, please send KSEB", "electrical_hazard", "urgent", "electricity"),
    ("Transformer sparking aanu near school, children ku danger", "electrical_hazard", "critical", "electricity"),
    ("High tension wire school roof ku near aanu, risk aanu", "electrical_hazard", "critical", "electricity"),
]

# ---------------------------------------------------------------------------
# STREET LIGHT
# ---------------------------------------------------------------------------
STREET_LIGHT: list[TrainingSample] = [
    # English
    ("Street light not working on our road for two weeks", "street_light", "medium", "electricity"),
    ("Several street lights are broken near the market area", "street_light", "medium", "electricity"),
    ("Street lights turn on during the day and stay off at night", "street_light", "medium", "electricity"),
    ("Dark stretch of road without any street lights, accidents happen", "street_light", "high", "electricity"),
    ("The street light pole is broken, wire exposed", "street_light", "high", "electricity"),
    ("Women feel unsafe walking after dark due to no street lights", "street_light", "high", "electricity"),
    ("New colony area has no street lights installed yet", "street_light", "medium", "electricity"),
    ("Street light flickering continuously for a month", "street_light", "low", "electricity"),
    ("Street light in front of the school not working", "street_light", "medium", "electricity"),
    # Malayalam
    ("തെരുവ് വിളക്ക് ഒരു മാസമായി കത്തുന്നില്ല", "street_light", "medium", "electricity"),
    ("ഇരുട്ടായ റോഡ്, രാത്രി സ്ത്രീകൾക്ക് സുരക്ഷിതമല്ല", "street_light", "high", "electricity"),
    # Manglish
    ("street light poyi, road andhakaram aanu", "street_light", "medium", "electricity"),
    ("vilakku maasam aay therinju, pinne kattunilla", "street_light", "medium", "electricity"),
    ("dark road, night il nadakkan bhayam", "street_light", "high", "electricity"),
    # Mixed
    ("Street light one month aay poyittu, road dark aanu", "street_light", "medium", "electricity"),
    ("Vilakku illatha kaaran rathri road il accident aayittu", "street_light", "high", "electricity"),
    ("Street light during day on aanu, night off aanu, ulta aanu", "street_light", "medium", "electricity"),
]

# ---------------------------------------------------------------------------
# TREE FALL / TREE HAZARD
# ---------------------------------------------------------------------------
TREE_FALL: list[TrainingSample] = [
    # English
    ("Large tree fell on the road blocking traffic completely", "tree_fall", "urgent", "parks"),
    ("Huge tree about to fall, leaning dangerously over the road", "tree_fall", "critical", "parks"),
    ("Tree fell on parked vehicles causing damage", "tree_fall", "urgent", "parks"),
    ("Rotten tree branch hanging over the road, could fall any time", "tree_fall", "high", "parks"),
    ("Tree uprooted in the storm last night, blocking main road", "tree_fall", "urgent", "parks"),
    ("Old tree leaning towards the school building, very dangerous", "tree_fall", "critical", "parks"),
    ("Branch fell on electric wire, now sparking", "tree_fall", "critical", "parks"),
    ("Fallen tree blocking the ambulance route to the hospital", "tree_fall", "critical", "parks"),
    ("Dead tree likely to collapse on houses below", "tree_fall", "high", "parks"),
    ("Large branch blocking the footpath near the park", "tree_fall", "medium", "parks"),
    # Malayalam
    ("മരം വഴിയിൽ വീണ് ട്രാഫിക് ബ്ലോക്ക്", "tree_fall", "urgent", "parks"),
    ("ചരിഞ്ഞ് നിൽക്കുന്ന വലിയ മരം, ഏത് നിമിഷവും വീഴും", "tree_fall", "critical", "parks"),
    # Manglish
    ("maram road il veenu, traffic block aanu", "tree_fall", "urgent", "parks"),
    ("valiya maram charinnu, veenekil venam, urgent", "tree_fall", "critical", "parks"),
    ("storm il maram valinja, road close aanu", "tree_fall", "urgent", "parks"),
    # Mixed
    ("Tree road il veenu last night, traffic jam aanu, clear cheyyunam", "tree_fall", "urgent", "parks"),
    ("Big maram leaning towards house, danger, clear cheyyunam", "tree_fall", "critical", "parks"),
    ("Branch fell on electric wire, spark vannu, emergency", "tree_fall", "critical", "parks"),
]

# ---------------------------------------------------------------------------
# ILLEGAL CONSTRUCTION
# ---------------------------------------------------------------------------
ILLEGAL_CONSTRUCTION: list[TrainingSample] = [
    # English
    ("Illegal construction on public land near the park", "illegal_construction", "high", "planning"),
    ("Building construction without permit violating setback rules", "illegal_construction", "high", "planning"),
    ("Encroachment on footpath by the shop owner next door", "illegal_construction", "medium", "planning"),
    ("Unauthorized floor added to the existing building", "illegal_construction", "high", "planning"),
    ("Construction work going on in the prohibited zone near the lake", "illegal_construction", "urgent", "planning"),
    ("Building constructed blocking the natural waterway", "illegal_construction", "high", "planning"),
    ("Shop extension encroaching onto the public road", "illegal_construction", "medium", "planning"),
    ("Construction of compound wall on government land", "illegal_construction", "high", "planning"),
    ("Construction waste dumped on public land", "illegal_construction", "medium", "planning"),
    ("Multi-storey building going up without any visible permit board", "illegal_construction", "high", "planning"),
    ("Poultry farm constructed in residential zone", "illegal_construction", "medium", "planning"),
    ("New construction blocking the drainage path near my house", "illegal_construction", "high", "planning"),
    # Malayalam
    ("അനധികൃത നിർമ്മാണം പൊതുഭൂമിയിൽ നടക്കുന്നു", "illegal_construction", "high", "planning"),
    ("പെർമിറ്റ് ഇല്ലാതെ കെട്ടിടം നിർമ്മിക്കുന്നു", "illegal_construction", "high", "planning"),
    # Manglish
    ("illegal building permit illatha, report cheyyunam", "illegal_construction", "high", "planning"),
    ("encroachment footpath il, kadakaran build cheythu", "illegal_construction", "medium", "planning"),
    ("construction govement land il, unauthorized", "illegal_construction", "high", "planning"),
    # Mixed
    ("Illegal construction pothubhoomiyil, permit illatha building", "illegal_construction", "high", "planning"),
    ("Footpath encroach cheythu shop build cheythu, block aanu", "illegal_construction", "medium", "planning"),
    ("Building without permit, violations report cheyythu", "illegal_construction", "high", "planning"),
]

# ---------------------------------------------------------------------------
# SPAM / IRRELEVANT
# ---------------------------------------------------------------------------
SPAM: list[TrainingSample] = [
    ("Buy cheap medicines online, call now for discount", "spam", "low", "none"),
    ("Earn money from home, work part time, Rs 5000 per day guaranteed", "spam", "low", "none"),
    ("Best home loan lowest interest rate call us immediately", "spam", "low", "none"),
    ("FREE OFFER call now limited time discount medicines", "spam", "low", "none"),
    ("Download our app for best deals on electronics", "spam", "low", "none"),
    ("You have won a prize, claim now by calling this number", "spam", "low", "none"),
    ("hello sir good morning", "spam", "low", "none"),
    ("test test test test", "spam", "low", "none"),
    ("123456 abc", "spam", "low", "none"),
    ("Please arrange a meeting with the commissioner tomorrow", "spam", "low", "none"),
    ("This is not a complaint just saying hi to the team", "spam", "low", "none"),
    ("aaaaaaaaaa bbbbbbb ccccc", "spam", "low", "none"),
    ("asdf jkl qwerty zxcv random text", "spam", "low", "none"),
    ("I need a government job, please help me get employment", "spam", "low", "none"),
    ("Sir please help us arrange a water tanker for wedding", "spam", "low", "none"),
    ("cheap flat for sale near bus stand contact owner directly", "spam", "low", "none"),
    ("political party event rally details time and place", "spam", "low", "none"),
    ("cricket match live score update latest news", "spam", "low", "none"),
    ("Enna venum ennik ariyilla", "spam", "low", "none"),
    ("vandi vannilla, mathi ittu", "spam", "low", "none"),
]

# ---------------------------------------------------------------------------
# NO CATEGORY (genuine but unclassifiable)
# ---------------------------------------------------------------------------
NO_CATEGORY: list[TrainingSample] = [
    ("There is a problem near my house please come and see", "no_category", "medium", "none"),
    ("Something is wrong in our area, officials should visit", "no_category", "medium", "none"),
    ("Please help us, we are facing issues here", "no_category", "medium", "none"),
    ("The situation here is very bad, action needed", "no_category", "medium", "none"),
    ("Residents are suffering, authority should take action", "no_category", "medium", "none"),
    ("Issue at junction, please check", "no_category", "medium", "none"),
    ("Problem here unresolved for months", "no_category", "medium", "none"),
    ("civic amenities poor quality in our ward", "no_category", "low", "none"),
    ("Facilities in our area are inadequate", "no_category", "low", "none"),
    ("Government must do something about our area", "no_category", "low", "none"),
]

# ---------------------------------------------------------------------------
# Complete combined corpus
# ---------------------------------------------------------------------------
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
)

# ---------------------------------------------------------------------------
# Category metadata (for validation and reporting)
# ---------------------------------------------------------------------------
CATEGORY_CODES: list[str] = [
    "water_supply",
    "drainage",
    "sewage_issue",
    "solid_waste",
    "road_damage",
    "electrical_hazard",
    "street_light",
    "tree_fall",
    "illegal_construction",
    "spam",
    "no_category",
]

PRIORITY_LEVELS: list[str] = ["low", "medium", "high", "urgent", "critical"]

DEPARTMENT_CODES: list[str] = [
    "water",
    "drainage",
    "sewage",
    "sanitation",
    "roads",
    "electricity",
    "parks",
    "planning",
    "none",
]
