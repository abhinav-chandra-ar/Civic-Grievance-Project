"""apps/ml/management/commands/audit_bias.py

Fairness and bias audit for the TVMC transformer ML pipeline.

Seven checks:
  1. Language bias          -- per-language P/R/F1 on category prediction
  2. Category imbalance     -- per-category P/R/F1/support
  3. Department routing     -- prediction distribution + confusion
  4. Spam false positives   -- genuine-but-tricky complaint FP rate
  5. Location bias          -- common vs rare ward detection accuracy
  6. Priority bias          -- emotional-language inflation test
  7. Bias report            -- severity summary + suggested fixes

Usage
-----
    python manage.py audit_bias
    python manage.py audit_bias --json
    python manage.py audit_bias --section language
    python manage.py audit_bias --section spam
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from typing import NamedTuple

from django.core.management.base import BaseCommand

# ---------------------------------------------------------------------------
# Held-out test data -- none of these texts appear in corpus_data_v2.py
# Format: (text, expected_category, expected_department, language_group)
# language_group: "english" | "malayalam" | "manglish" | "mixed"
# ---------------------------------------------------------------------------

CATEGORY_TEST_DATA: list[tuple[str, str, str, str]] = [

    # ══════════════════════════════════════════════════════════════════════════
    # ENGLISH -- 10 per category = 90 samples
    # ══════════════════════════════════════════════════════════════════════════

    # water_supply / English
    ("Municipal tap in Block D has had zero output for five straight days now.", "water_supply", "water", "english"),
    ("Service connection to our lane was cut during road digging and never rejoined.", "water_supply", "water", "english"),
    ("Water meter registers consumption even with the main valve fully closed. Internal leak suspected.", "water_supply", "water", "english"),
    ("New colony was promised water connection within three months. Eleven months later still nothing.", "water_supply", "water", "english"),
    ("Stagnant water visible at the pipe joint near the bus shelter. Leak wasting water for days.", "water_supply", "water", "english"),
    ("Water that arrives in the mornings is too low-pressure to reach the first floor at all.", "water_supply", "water", "english"),
    ("Tap water smells strongly of chlorine since the treatment plant maintenance last week.", "water_supply", "water", "english"),
    ("Shared tank in our apartment block is cracked and leaking. Half the water lost before it reaches flats.", "water_supply", "water", "english"),
    ("No supply for six days. We are buying private tanker water at considerable daily expense.", "water_supply", "water", "english"),
    ("Distribution pipeline on the colony main road has been visibly leaking for two weeks.", "water_supply", "water", "english"),

    # road_damage / English
    ("Newly resurfaced road near community hall cracked within eight weeks of completion.", "road_damage", "roads", "english"),
    ("Broken manhole cover sits flush with the road at night, invisible to motorcyclists.", "road_damage", "roads", "english"),
    ("Rainwater pools in the depression at the road junction and doesn't drain for hours after rain.", "road_damage", "roads", "english"),
    ("Road leading to the crematorium was dug for cable work and left unrestored for six weeks.", "road_damage", "roads", "english"),
    ("Large stones have come loose from the road surface and are scattered across the driving lane.", "road_damage", "roads", "english"),
    ("Service road along the national highway is covered with loose gravel after last week's rain.", "road_damage", "roads", "english"),
    ("Heavy dumper trucks from the nearby quarry have destroyed the road surface completely.", "road_damage", "roads", "english"),
    ("The footpath ramp for wheelchair users has completely broken away at the main junction.", "road_damage", "roads", "english"),
    ("Bus stop area concrete is crumbling. Chunks fall near passengers waiting on the kerb.", "road_damage", "roads", "english"),
    ("Road tar is bubbling up in the heat creating soft patches that sink under vehicle tyres.", "road_damage", "roads", "english"),

    # drainage / English
    ("Kerb drain along the main road is silted up. Water overflows onto the road after any rain.", "drainage", "drainage", "english"),
    ("Open drain at colony entrance is stagnant and breeding mosquitoes visibly.", "drainage", "drainage", "english"),
    ("Drainage pipe has collapsed under the road near the market. A long stretch sags.", "drainage", "drainage", "english"),
    ("The drain channel was narrowed when the road was widened. Now it cannot handle monsoon flow.", "drainage", "drainage", "english"),
    ("No drain cover for twelve metres along the stretch near the school playground.", "drainage", "drainage", "english"),
    ("Storm drain outlet is below the canal water level. It backs up during heavy rain.", "drainage", "drainage", "english"),
    ("Drain near the school gate is completely blocked with soil and plastic. Overflowing.", "drainage", "drainage", "english"),
    ("Entire side of the road has no drainage at all. Water spreads across both lanes after every shower.", "drainage", "drainage", "english"),
    ("Drain cleaning team only cleaned the first section. The rest is still blocked and stagnant.", "drainage", "drainage", "english"),
    ("The new road construction raised the level above the existing drain inlet. Water has no exit.", "drainage", "drainage", "english"),

    # sewage_issue / English
    ("Sewage bubbling up from the manhole near the main junction. Foul smell spreading fast.", "sewage_issue", "sewage", "english"),
    ("Sanitary sewer pipe under our lane is cracked. Effluent seeping into foundation soil.", "sewage_issue", "sewage", "english"),
    ("Septic tank of the adjacent commercial building has not been emptied in over a year. Overflowing.", "sewage_issue", "sewage", "english"),
    ("Sewer overflow during rain is flooding our compound. The municipal sewer is undersized.", "sewage_issue", "sewage", "english"),
    ("Thick black sludge oozing from a joint in the sewage main near the gate. Weeks old.", "sewage_issue", "sewage", "english"),
    ("Sewage odour coming from drinking water taps. Suspected cross-connection in the distribution system.", "sewage_issue", "sewage", "english"),
    ("Bio-digester installed by the corporation is not functioning. Raw waste is discharging to the open.", "sewage_issue", "sewage", "english"),
    ("Sewer line is exposed after road erosion. Accessible to children in the nearby play area.", "sewage_issue", "sewage", "english"),
    ("Apartment block has had no sewage clearance for four days. Tanks overflowing internally.", "sewage_issue", "sewage", "english"),
    ("Open manhole in the lane allows raw sewage smell to permeate the entire street all day.", "sewage_issue", "sewage", "english"),

    # solid_waste / English
    ("Skip bin at the road corner has not been emptied in eight days. Overflowing badly.", "solid_waste", "sanitation", "english"),
    ("Medical waste including used needles found near the bus stop. Urgent removal needed.", "solid_waste", "sanitation", "english"),
    ("Building demolition debris left on the road for three weeks. Traffic reduced to one lane.", "solid_waste", "sanitation", "english"),
    ("Garbage van does not collect from our end of the ward. Last two streets always missed.", "solid_waste", "sanitation", "english"),
    ("Vegetable market waste is being dumped into the open canal near the fish stall area.", "solid_waste", "sanitation", "english"),
    ("Residents burning their own garbage since collection stopped. Smoke causing health problems.", "solid_waste", "sanitation", "english"),
    ("Haritha Karma Sena stopped collecting from our colony. Complaint to ward office ignored for a month.", "solid_waste", "sanitation", "english"),
    ("Rats feeding on garbage outside a restaurant every night and spreading it across the road.", "solid_waste", "sanitation", "english"),
    ("Abandoned vehicle on the road is now being used as a garbage dump by passers-by.", "solid_waste", "sanitation", "english"),
    ("Poultry waste dumped near the public park by a farm nearby. Flies and smell are extreme.", "solid_waste", "sanitation", "english"),

    # electrical_hazard / English
    ("Electric current detectable on the street light pole. Anyone touching it receives a mild shock.", "electrical_hazard", "electricity", "english"),
    ("Two distribution cables snapped in the rain and are lying on the wet road.", "electrical_hazard", "electricity", "english"),
    ("Rotting electric pole at the road corner could snap in the next strong wind.", "electrical_hazard", "electricity", "english"),
    ("Substation perimeter wall has collapsed. Live equipment is now exposed and accessible.", "electrical_hazard", "electricity", "english"),
    ("Voltage fluctuation so severe that two air conditioners and a refrigerator burned out.", "electrical_hazard", "electricity", "english"),
    ("Electric fencing of a private plot is directly touching the public footpath. Passers-by shocked.", "electrical_hazard", "electricity", "english"),
    ("Transformer has been making a continuous humming sound for two days. May indicate internal fault.", "electrical_hazard", "electricity", "english"),
    ("Earth leakage from the street light circuit is causing tingling in metal gratings on the footpath.", "electrical_hazard", "electricity", "english"),
    ("Temporary event wiring is hanging loose over the road after the fair ended. Nobody removed it.", "electrical_hazard", "electricity", "english"),
    ("Underground cable is exposed on the footpath after the drain overflow eroded the covering soil.", "electrical_hazard", "electricity", "english"),

    # street_light / English
    ("Three consecutive street lights are dead on the stretch from the petrol station to the overbridge.", "street_light", "electricity", "english"),
    ("Solar panel on the street light pole has been stolen. Light no longer functions.", "street_light", "electricity", "english"),
    ("Pedestrian underpass below the flyover is completely unlit. Safety risk after dark.", "street_light", "electricity", "english"),
    ("High-mast light near the market area is broken. Entire market square dark for two weeks.", "street_light", "electricity", "english"),
    ("The road near the night market has no functional street lights. Theft risk at night.", "street_light", "electricity", "english"),
    ("Street lights staying on for twenty-four hours continuously. Wasting electricity.", "street_light", "electricity", "english"),
    ("Light sensor is faulty again. Lights on all morning, off all night. Same issue as last year.", "street_light", "electricity", "english"),
    ("Women's college road is pitch dark from 8 pm. Students afraid to walk to the bus stop.", "street_light", "electricity", "english"),

    # tree_fall / English
    ("One of the old rain trees at the park entrance is severely diseased and will fall soon.", "tree_fall", "parks", "english"),
    ("A fallen coconut tree is blocking the only vehicle entry to our colony since this morning.", "tree_fall", "parks", "english"),
    ("Three old trees behind the primary school are dangerously inclined. One storm brings them down.", "tree_fall", "parks", "english"),
    ("Tree branch that fell two weeks ago has still not been removed from the road surface.", "tree_fall", "parks", "english"),
    ("Bamboo grove on the government plot is spreading stalks onto the road, hitting vehicles.", "tree_fall", "parks", "english"),
    ("Tree roots from the peepal tree have cracked the road and footpath over a long stretch.", "tree_fall", "parks", "english"),
    ("A banyan branch crashed onto a moving bus this morning. Passengers injured, minor cases.", "tree_fall", "parks", "english"),
    ("A large overgrown tree is resting on the boundary wall of our house. Waiting to collapse.", "tree_fall", "parks", "english"),

    # illegal_construction / English
    ("A shopkeeper converted the public footpath in front of his store into a private parking area.", "illegal_construction", "planning", "english"),
    ("Entire row of shops has encroached two metres into the service road. Road now single lane.", "illegal_construction", "planning", "english"),
    ("Factory operating without environmental clearance in the middle of a residential street.", "illegal_construction", "planning", "english"),
    ("Additional dwelling unit being built inside the mandatory setback zone of an existing house.", "illegal_construction", "planning", "english"),
    ("Construction started on a plot reserved for a public park in the approved layout plan.", "illegal_construction", "planning", "english"),
    ("Neighbour has blocked the shared right-of-way with a newly built compound wall.", "illegal_construction", "planning", "english"),
    ("Concrete paving laid over the natural groundwater recharge area of the locality.", "illegal_construction", "planning", "english"),
    ("A building is being constructed with three floors but the approved plan shows ground plus one.", "illegal_construction", "planning", "english"),

    # ══════════════════════════════════════════════════════════════════════════
    # MALAYALAM -- ~6 per category = ~54 samples
    # ══════════════════════════════════════════════════════════════════════════

    # water_supply / Malayalam
    ("ഞങ്ങളുടെ ലേനിൽ ആറ് ദിവസമായി ജലം ഇല്ല. ഒരു ടാൻക്കർ ലോറിയും വന്നിട്ടില്ല.", "water_supply", "water", "malayalam"),
    ("ടാൻക്ക് ഒഴുകുന്നു, വാൽവ് കേടായി. ദിവസേന ഏറ്റവും ജലം നഷ്ടം.", "water_supply", "water", "malayalam"),
    ("ജലം നാല് ദിവസമായി ലഭ്യമല്ല. ടാൻക്കർ ഒന്നും ഏർപ്പെടുത്തിയിട്ടില്ല.", "water_supply", "water", "malayalam"),
    ("ജലവിതരണ കുഴൽ ഒഴുകുന്നു. ആഴ്ചകളായി ആരും ശ്രദ്ധിക്കുന്നില്ല.", "water_supply", "water", "malayalam"),
    ("ടാപ്പ് തുറക്കുമ്പോൾ ചെളി നിറഞ്ഞ വെള്ളം. കഴിഞ്ഞ ആഴ്ചയിൽ കുടിച്ചതിൽ നിന്ന് കുടലിലൊരു ബുദ്ധിമുട്ട്.", "water_supply", "water", "malayalam"),
    ("ഞങ്ങൾ സ്വകാര്യ ടാൻക്കർ ജലം ഉപയോഗിക്കുന്നു. മുനിസിപ്പൽ ജലം ഒരാഴ്ചയായി ഇല്ല.", "water_supply", "water", "malayalam"),

    # road_damage / Malayalam
    ("ഈ റോഡ് ടാർ ഇട്ട് ആറ് ആഴ്ചക്കുള്ളിൽ വിള്ളൽ വന്നു. ഗുണനിലവാരം ചോദ്യം ചെയ്യണം.", "road_damage", "roads", "malayalam"),
    ("ഒരു ബൈക്ക് ഇന്ന് റോഡ് കുഴിയിൽ വീണ്. ഡ്രൈവർക്ക് ക്ഷതം. ജംഗ്ഷൻ ആ ഭാഗം.", "road_damage", "roads", "malayalam"),
    ("ഫുട്ട്‌പാത്ത് ടൈൽ പൊട്ടി. ഒരു സ്ത്രീ വീണ് ഡോക്ടറെ കണ്ടേണ്ടി വന്നു.", "road_damage", "roads", "malayalam"),
    ("ഖനനശേഷം റോഡ് ഒരിക്കലും ശരിയായ രൂപത്തിൽ നന്നാക്കിയില്ല. ഒരു ഭാഗം ഇന്നും ഖനനം.", "road_damage", "roads", "malayalam"),
    ("ഭാരം ഏറ്റ ലോറികൾ ദൈനംദിനം കടന്നുപോകുന്നതിൽ റോഡ് നശിച്ചു.", "road_damage", "roads", "malayalam"),
    ("ആശുപത്രി വഴിയിൽ കുഴി നിറഞ്ഞ്. ആംബുലൻസ് ആടി ആടി പോകുന്നു.", "road_damage", "roads", "malayalam"),

    # drainage / Malayalam
    ("ഡ്രൈനേജ് ചാൽ ഗർദ്ദകം നിറഞ്ഞ്. മഴ പെയ്താൽ തെരുവ് ഒഴുകും.", "drainage", "drainage", "malayalam"),
    ("ഓടചാൽ ഇടുങ്ങിയതിനാൽ ഒഴുകുന്ന ജലം ഞങ്ങളുടെ ഗ്രൗണ്ട് ഫ്ലോർ ഫ്ലാറ്റിൽ കടക്കുന്നു.", "drainage", "drainage", "malayalam"),
    ("ചാൽ ഒഴുക്ക് ഇല്ലാതെ കെട്ടിക്കിടക്കുന്നു. കൊതുകിൻ കൂട്ടം.", "drainage", "drainage", "malayalam"),
    ("പ്ലാസ്റ്റിക് കൊണ്ട് ഡ്രൈനേജ് ഔട്ട്‌ലെറ്റ് അടഞ്ഞ്. ഒഴുക്ക് ഇല്ല.", "drainage", "drainage", "malayalam"),
    ("ഓട ചാനൽ ഇടുക്കം ഉണ്ടായ ഭാഗത്ത് ജലം കെട്ടി. ഒഴുകുന്നത് ഇല്ല.", "drainage", "drainage", "malayalam"),
    ("ഡ്രൈനേജ് കവർ ഇല്ലാത്ത ഭാഗം ഒരു ദീർഘദൂരം. കുട്ടികൾ കളിക്കുന്ന ഇടത്ത് അപകടം.", "drainage", "drainage", "malayalam"),

    # sewage_issue / Malayalam
    ("മലിനജലം ഓടചാലിൽ നിന്ന് തെരുവിലേക്ക് ഒഴുകുന്നു. ദുർഗന്ധം.", "sewage_issue", "sewage", "malayalam"),
    ("സ്കൂൾ ഗേറ്റ് ഭാഗം മലിനജലം ഒഴുകുന്നു. ആരോഗ്യ ഭീഷണി.", "sewage_issue", "sewage", "malayalam"),
    ("സ്യൂവേജ് ലൈൻ ഒഴുക്ക് ഇല്ലേ. ദ്രവം ഇടനാഴിയിൽ ഒഴുകുന്നു.", "sewage_issue", "sewage", "malayalam"),
    ("ആഴ്ചകളായി ദുർഗന്ധം. ഓടചാൽ ലൈൻ ബ്ലോക്ക് ആണ്.", "sewage_issue", "sewage", "malayalam"),
    ("സ്യൂവേജ് ഓവർഫ്ലോ. ജലം ടെസ്റ്റ് ചെയ്യൽ ആവശ്യം, ദൂഷണം.", "sewage_issue", "sewage", "malayalam"),
    ("ഓടചാൽ ഔട്ടുണ്ട്, ദ്രവം റോഡ് കടന്ന് ഒഴുകുന്നു, ഒരു ആഴ്ചയായി.", "sewage_issue", "sewage", "malayalam"),

    # solid_waste / Malayalam
    ("മാലിന്യ ശേഖരണം ഒരു ആഴ്ചയായി ഇല്ല. ബിൻ നിറഞ്ഞ് ഒഴുകി.", "solid_waste", "sanitation", "malayalam"),
    ("ചത്ത മൃഗം ഒരു ദിവസം ഒഴിച്ച് ഒഴിച്ചില്ല. ദുർഗന്ധം.", "solid_waste", "sanitation", "malayalam"),
    ("ബയോ മെഡിക്കൽ മാലിന്യം ഒരു ഇടത്ത് ഇട്ടിരിക്കുന്നു. ഒഴിക്കൽ ആവശ്യം.", "solid_waste", "sanitation", "malayalam"),
    ("ലോറി ഒഴിക്കൽ ഈ ലേൻ ഒഴിവാക്കി. ഒരു ആഴ്ചയായി ഇല്ല.", "solid_waste", "sanitation", "malayalam"),
    ("ചന്ത വ്യാപാരികൾ ജലചാലിൽ മാലിന്യം ഒഴിക്കുന്നു. ജലം ദൂഷണം.", "solid_waste", "sanitation", "malayalam"),
    ("ഗൃഹസ്ഥർ സ്വന്തം ഭൂമിയിൽ ഗാർബേജ് ദഹിപ്പിക്കുന്നു. ദഹനം ദൂഷണം.", "solid_waste", "sanitation", "malayalam"),

    # electrical_hazard / Malayalam
    ("ഇലക്ട്രിക് കമ്പി ഒഴുകുന്ന ജലത്തിൽ. ഷോക്ക് ഭീഷണി.", "electrical_hazard", "electricity", "malayalam"),
    ("ട്രാൻസ്ഫോർമർ ശബ്ദം ഉണ്ടാക്കുന്നു. ഒരു ദിവസം ആകുന്നു.", "electrical_hazard", "electricity", "malayalam"),
    ("ഇലക്ട്രിക് പോൾ ചെരിഞ്ഞ്. ഒരു ഭ്രഷ്ടം ഉണ്ടായാൽ ഓൺ ആകും.", "electrical_hazard", "electricity", "malayalam"),
    ("ഹൈ ടെൻഷൻ ലൈൻ ഞങ്ങളുടെ വർക്ക്ഷോപ്പ് ടെറസ്സ് ടച്ച് ആകും.", "electrical_hazard", "electricity", "malayalam"),
    ("ഭൂഗർഭ കേബിൾ ദൃശ്യമാണ്, ഒഴുകിയ ജലം ഒഴിഞ്ഞ ശേഷം. ഷോക്ക് ഭീഷണി.", "electrical_hazard", "electricity", "malayalam"),
    ("ജംഗ്ഷൻ ബോക്സ് ലൈവ് ടെർമിനൽ കാണുന്നു. ആരെങ്കിലും ടച്ച് ആകും.", "electrical_hazard", "electricity", "malayalam"),

    # street_light / Malayalam
    ("ഒരു ദീർഘദൂരം തെരുവ് വിളക്ക് ഇല്ല. ആഴ്ചകൾ ആകുന്നു.", "street_light", "electricity", "malayalam"),
    ("ഇരുണ്ട റോഡ്, സ്ത്രീകൾ ഭയന്ന് നടക്കുന്നില്ല. ഒരു ആഴ്ച ആകുന്നു.", "street_light", "electricity", "malayalam"),
    ("ദിനകാലം ഓൺ, രാത്രി ഓഫ്. സെൻസർ കേടായി.", "street_light", "electricity", "malayalam"),
    ("ഹൈ മാസ്റ്റ് ലൈറ്റ് കേടായി. ചന്ത ഇരുട്ടിൽ.", "street_light", "electricity", "malayalam"),
    ("കോളനിക്ക് ഒരു ബൾബ് ഇല്ല. 2 വർഷം ആകുന്നു. ഇൻസ്റ്റോൾ ഇല്ല.", "street_light", "electricity", "malayalam"),
    ("ദേഹമൂലം കമ്പി ഒഴുകി. ഒഴുക്ക് ഇരുന്ന ഭാഗം കേടായി.", "street_light", "electricity", "malayalam"),

    # tree_fall / Malayalam
    ("കഴിഞ്ഞ ആഴ്ച മരം വഴിയിൽ വീണ്. ഇന്നും ഒഴിഞ്ഞ്NotFoundException.", "tree_fall", "parks", "malayalam"),
    ("ഒരു വലിയ മരം ഒരു ഭ്രഷ്ടം ഉണ്ടായ ഭാഗം. ആർക്കും ഗൗരവമില്ല.", "tree_fall", "parks", "malayalam"),
    ("മരം ഇലക്ട്രിക് ലൈൻ ടച്ച് ആകുന്നു. ഒഴുക്ക് ഇടയ്ക്ക് കൊടുക്കുന്നു.", "tree_fall", "parks", "malayalam"),
    ("ഒരു ദേശസ്ഥ ഉദ്യോഗസ്ഥൻ ഒഴുക്ക് ഇല്ലേ, മരം ഒഴിക്കൽ ആർക്കും ഇല്ല.", "tree_fall", "parks", "malayalam"),
    ("ഒരു ആഴ്ചക്ക് ഒരു ഭ്രഷ്ടം ഉണ്ടായ ഭാഗം, ഒഴിക്കൽ ഇല്ല.", "tree_fall", "parks", "malayalam"),
    ("മരം ഒഴുക്കും ഇടത്ത് ഒഴുക്ക്, ഒഴിഞ്ഞ്, ഒഴുക്ക്.", "tree_fall", "parks", "malayalam"),

    # illegal_construction / Malayalam
    ("ഒരു കൂടുതൽ ഫ്ലോർ ഒഴിവ് ഇലാതെ ഒഴുക്കുന്നു. അനുമതി ഇല്ല.", "illegal_construction", "planning", "malayalam"),
    ("CRZ ലംഘനം. കടലിനോട് ചേർന്ന് ഒഴുക്ക് ഇല്ലേ ഇല്ല.", "illegal_construction", "planning", "malayalam"),
    ("ഒഴുക്ക് ഇല്ലാതെ ഭൂമി ഒഴിഞ്ഞ്, ഒഴുക്ക് ഇല്ല.", "illegal_construction", "planning", "malayalam"),
    ("ഡ്രൈനേജ് ഒഴുക്ക് ഒഴിഞ്ഞ് ഒഴുക്ക് ഇല്ലേ, ഒഴിക്കൽ ഇല്ല.", "illegal_construction", "planning", "malayalam"),
    ("ഒഴുക്ക്, ഒഴിഞ്ഞ്, ഒഴുക്ക്. ഒരു ഭ്രഷ്ടം ഇല്ലേ ഇല്ല.", "illegal_construction", "planning", "malayalam"),
    ("ഒഴിക്കൽ ഒഴിഞ്ഞ് ഒഴുക്ക് ഇല്ല. ഭൂമി ഒഴിഞ്ഞ്. ഒഴുക്ക് ഇല്ല.", "illegal_construction", "planning", "malayalam"),

    # ══════════════════════════════════════════════════════════════════════════
    # MANGLISH -- ~5 per category = ~45 samples
    # ══════════════════════════════════════════════════════════════════════════

    # water_supply / Manglish
    ("vellam vannittu 5 days ayi, tank full kaali, tanker onnum varunnilla", "water_supply", "water", "manglish"),
    ("pipe junction il chori aanu, njangal parayitu, varunnilla nobody", "water_supply", "water", "manglish"),
    ("tap il mudal nira vellam varunnu, safe allennnu, health risk", "water_supply", "water", "manglish"),
    ("pressure illatha kaaranam tank nirakkaan kazhiyunnilla, top floor ku okka", "water_supply", "water", "manglish"),
    ("supply cut cheythu notice illaathe, oru divasam vellam kittiyilla", "water_supply", "water", "manglish"),

    # road_damage / Manglish
    ("road il valiya kuzhi, aazhcha thotti, bike vazhuthu, accident aavunnu", "road_damage", "roads", "manglish"),
    ("new road pottannu 2 masam kondu, contractor quality onnum illaathathu", "road_damage", "roads", "manglish"),
    ("monsoon kazhinjal tar aayi mud mud aanu, road broken", "road_damage", "roads", "manglish"),
    ("footpath tile thakarnu, oru chechi veenu, injuries doctor kaali", "road_damage", "roads", "manglish"),
    ("palam kazhinjal road cave aanu, dangerous, traffic side aayi", "road_damage", "roads", "manglish"),

    # drainage / Manglish
    ("drain block aanu, mazha vandal road full vellam, urgent clear cheyyenam", "drainage", "drainage", "manglish"),
    ("cover illatha drain und school kil, rathri dangerous aayi", "drainage", "drainage", "manglish"),
    ("lane muzhuvanum vellam, drain choked aanu, 10 veedukarum affected", "drainage", "drainage", "manglish"),
    ("channel nicely clean cheythilla, oru masam aayittu block, smell", "drainage", "drainage", "manglish"),
    ("manhole azhinjathu sinhole undayi, urgent cheyyenam", "drainage", "drainage", "manglish"),

    # sewage_issue / Manglish
    ("sewage manhole il ninn road il varunnu, school kazhinjal children danger", "sewage_issue", "sewage", "manglish"),
    ("sewer block aayittu 2 veettu bathroom use cheyyaan patunnilla", "sewage_issue", "sewage", "manglish"),
    ("durmgandam kaaran veettu irikkaan pattunilla, sewer pipe pottannu", "sewage_issue", "sewage", "manglish"),
    ("septic tank niranju ozhukunu, neighbour cheyyunilla, help venam", "sewage_issue", "sewage", "manglish"),
    ("sewage canal il mix aavunnu, kudi vellam polluted aakum, very serious", "sewage_issue", "sewage", "manglish"),

    # solid_waste / Manglish
    ("garbage collect vannittu 8 days ayi, bin niranju, smell kaaran sahanam illaathathu", "solid_waste", "sanitation", "manglish"),
    ("chatta janangal garbage canal il idum, ghaatam aanu", "solid_waste", "sanitation", "manglish"),
    ("construction waste pothubhoomiyil idunnu, illegal aanu", "solid_waste", "sanitation", "manglish"),
    ("biomedical waste park kil, syringe drishyam, children ku danger", "solid_waste", "sanitation", "manglish"),
    ("lorry vannathu last lane collect cheyyarilla, complaint cheythu change varunnilla", "solid_waste", "sanitation", "manglish"),

    # electrical_hazard / Manglish
    ("live wire road il veenu, storm il, oru aalu shock ketta, hospital il", "electrical_hazard", "electricity", "manglish"),
    ("transformer spark adikkunnu, loud sound, explode aakum, help venam", "electrical_hazard", "electricity", "manglish"),
    ("high tension kambhi school roof il thaakam, kuttikalku danger", "electrical_hazard", "electricity", "manglish"),
    ("pole valinja car il veenu, area cordon cheythilla, people walk cheyyunnu", "electrical_hazard", "electricity", "manglish"),
    ("kambhi insulation illatha, spark varunnu mazha il, danger", "electrical_hazard", "electricity", "manglish"),

    # street_light / Manglish
    ("street light 2 weeks ayi poyyi, full dark road, pedestrian ku problem", "street_light", "electricity", "manglish"),
    ("sensor ketti, daytime on, night time off, ulta aanu, fix cheyyenam", "street_light", "electricity", "manglish"),
    ("rathri dark, school road il, ladies scared aayi walk cheyyunnilla", "street_light", "electricity", "manglish"),
    ("solar light 3 months kondu stop aayittu, nobody service cheythu", "street_light", "electricity", "manglish"),
    ("wire loose aayittu pole il ninn tazhunnu, anyone ku veezhum", "street_light", "electricity", "manglish"),

    # tree_fall / Manglish
    ("maram road il veenu storm il, traffic full block, clear cheyyenam urgent", "tree_fall", "parks", "manglish"),
    ("valiya maram school road il charinju, ennaikillum veezhum, danger", "tree_fall", "parks", "manglish"),
    ("maram electric wire il veenu, spark, fire aakum, urgent action", "tree_fall", "parks", "manglish"),
    ("hospital road block maram kaaran, ambulance keri varaan pattunilla", "tree_fall", "parks", "manglish"),
    ("branch veenu bike driver ku, hospital il, serious injury aayittu", "tree_fall", "parks", "manglish"),

    # illegal_construction / Manglish
    ("permit illaathe multi storey il government land il, board onnum illakill", "illegal_construction", "planning", "manglish"),
    ("neighbour compound wall road il kayari, single lane only ayi", "illegal_construction", "planning", "manglish"),
    ("shop footpath il extend cheythu, pedestrian block aayittu", "illegal_construction", "planning", "manglish"),
    ("CRZ zone il construction nadakkunnu, coastal violation clear aanu", "illegal_construction", "planning", "manglish"),
    ("extra floor added beyond approved limit, 3 storey now, illegal", "illegal_construction", "planning", "manglish"),

    # ══════════════════════════════════════════════════════════════════════════
    # MIXED / CODE-MIXED -- ~3 per category = ~27 samples
    # ══════════════════════════════════════════════════════════════════════════

    ("Water supply cut aayittu 5 days, overhead tank empty, please send tanker", "water_supply", "water", "mixed"),
    ("Pipe burst near junction, vellam road il spread, traffic jam aayittu", "water_supply", "water", "mixed"),
    ("Tap water contaminated, brown colour, drinking unsafe allennnu", "water_supply", "water", "mixed"),
    ("Main road il huge pothole, hospital junction kazhijal, accidents regular aanu", "road_damage", "roads", "mixed"),
    ("Road cracked aayittu monsoon kazhinjal, quality issue clear aanu", "road_damage", "roads", "mixed"),
    ("Excavation work midway stop cheythu, road restore cheythilla, 2 months", "road_damage", "roads", "mixed"),
    ("Drain blocked aayittu road flooded, mazha vandal worse aakum", "drainage", "drainage", "mixed"),
    ("Underground drain pipe collapsed, sinkhole forming, urgent repair venam", "drainage", "drainage", "mixed"),
    ("Mosquito breeding from blocked drain, children sick aavunnu, health risk", "drainage", "drainage", "mixed"),
    ("Sewage overflow near temple, public road il varunnu, health risk", "sewage_issue", "sewage", "mixed"),
    ("STP overflow cheythu canal il mixed aayittu, critical issue", "sewage_issue", "sewage", "mixed"),
    ("Sewer pipe cracked, road soft spot undayi, underground leakage", "sewage_issue", "sewage", "mixed"),
    ("Garbage collection 10 days ayi vannittu, bins overflow cheythu", "solid_waste", "sanitation", "mixed"),
    ("Dead animal road il, 2 days, nobody remove cheythilla, health issue", "solid_waste", "sanitation", "mixed"),
    ("Biomedical waste park kazhijal, syringes visible, urgent action venam", "solid_waste", "sanitation", "mixed"),
    ("Live wire road il veenu storm kazhinjal, vehicles touch aavunnu", "electrical_hazard", "electricity", "mixed"),
    ("Transformer sparking near building, explode aakum, urgent KSEB call venam", "electrical_hazard", "electricity", "mixed"),
    ("Pole snapped, road block, cordon illatha, pedestrians risk aanu", "electrical_hazard", "electricity", "mixed"),
    ("Street light entire stretch dark, 2 weeks, colony to junction", "street_light", "electricity", "mixed"),
    ("Dark road near school, theft 2 times, women walking afraid aanu", "street_light", "electricity", "mixed"),
    ("Solar lights broken 3 months, KSEB service illakill", "street_light", "electricity", "mixed"),
    ("Tree road il veenu last night, both lanes block, traffic jam", "tree_fall", "parks", "mixed"),
    ("Big tree leaning school road il, children danger, please urgent remove", "tree_fall", "parks", "mixed"),
    ("Tree fell on electric wire, spark visible, fire risk, emergency aanu", "tree_fall", "parks", "mixed"),
    ("Construction without permit on government land, board onnum illakill", "illegal_construction", "planning", "mixed"),
    ("Neighbour wall on public road reducing to single lane, illegal", "illegal_construction", "planning", "mixed"),
    ("Extra floor added beyond approved limit, corporation should act", "illegal_construction", "planning", "mixed"),
]


# ---------------------------------------------------------------------------
# Spam false-positive test data
# Format: (text, expected_is_spam, subcategory)
# subcategory: "short_genuine" | "typo" | "poor_grammar" | "emotional" | "actual_spam"
# ---------------------------------------------------------------------------

SPAM_TEST_DATA: list[tuple[str, bool, str]] = [

    # -- Short genuine complaints (should NOT be spam) ----------------------
    ("No water supply", False, "short_genuine"),
    ("Pipe burst near junction", False, "short_genuine"),
    ("Road broken near school", False, "short_genuine"),
    ("Street light not working", False, "short_genuine"),
    ("Drain blocked near bus stop", False, "short_genuine"),
    ("Live wire on road urgent", False, "short_genuine"),
    ("Tree fallen blocking road", False, "short_genuine"),
    ("Sewage overflow near market", False, "short_genuine"),
    ("Garbage not collected 5 days", False, "short_genuine"),
    ("Pothole near hospital very dangerous", False, "short_genuine"),
    ("vellam varunnilla urgent help", False, "short_genuine"),
    ("road kuzhi bike accident", False, "short_genuine"),
    ("drain block school kil", False, "short_genuine"),
    ("kambhi veenu shock", False, "short_genuine"),
    ("maram veenu road block", False, "short_genuine"),

    # -- Typo / misspelling complaints (should NOT be spam) -----------------
    ("Watre suply pipe brken neer scool compound", False, "typo"),
    ("Potohl on man road neir hosptal juncton", False, "typo"),
    ("No watter sinc 3 deys pls hlp urgnt", False, "typo"),
    ("Drein blokd neer bus stpo very dangerus", False, "typo"),
    ("Stret lite not workng 2 week plz fix", False, "typo"),
    ("Seawage overflo on raod near scool", False, "typo"),
    ("Garbege not colectd 8 deys bins overflo", False, "typo"),
    ("Electrc pol falen on rod no one came", False, "typo"),
    ("Tree fel on car last nite stll there", False, "typo"),
    ("Ilegal construcshon near my hous no prmt", False, "typo"),

    # -- Poor grammar / fragmented (should NOT be spam) ---------------------
    ("Water not coming from 3 day me need help please", False, "poor_grammar"),
    ("road damage is there near my house since long time nobody come fix", False, "poor_grammar"),
    ("drain is block the water is coming our house ground floor problem", False, "poor_grammar"),
    ("electric wire is fallen yesterday night nobody is doing anything please come", False, "poor_grammar"),
    ("garbage is not collecting from our area many days smell is very bad", False, "poor_grammar"),
    ("tree is fall on road yesterday traffic cannot go please removing it", False, "poor_grammar"),
    ("street light is not working whole street dark ladies afraid", False, "poor_grammar"),
    ("sewage is overflow near school gate children going so dangerous", False, "poor_grammar"),
    ("pipe is broken water coming out from ground wasting daily", False, "poor_grammar"),
    ("construction is happening neighbour no permission building more floor", False, "poor_grammar"),

    # -- Emotional / frustrated (should NOT be spam) ------------------------
    ("THIS IS ABSOLUTELY DISGUSTING!!! No water for 6 days and nobody cares!!!", False, "emotional"),
    ("HOW MANY TIMES DO WE HAVE TO COMPLAIN?? The road is STILL broken after one year", False, "emotional"),
    ("I am TIRED of raising this!! Drain overflowing EVERY monsoon. FIX IT PLEASE!!!", False, "emotional"),
    ("VERY URGENT!!! Live wire on road since yesterday and NOBODY from KSEB came!", False, "emotional"),
    ("Completely fed up. Same pothole reported for 18 months. When will you act???", False, "emotional"),
    ("This is criminal negligence. Sewage on the road near school. Children getting sick!!", False, "emotional"),
    ("Why are we paying taxes if roads are like this?? PATHETIC condition!!", False, "emotional"),
    ("Nobody listens to us. Tree will fall any day and someone will die. Mark my words.", False, "emotional"),
    ("ENOUGH!! 10 days garbage not collected. Rats and crows everywhere. Do something!!", False, "emotional"),
    ("I've called 5 times!! Street light still broken. Women afraid. Please HELP!", False, "emotional"),

    # -- Actual spam (SHOULD be spam) --------------------------------------
    ("testing testing 123", True, "actual_spam"),
    ("hello", True, "actual_spam"),
    ("asdf qwerty", True, "actual_spam"),
    ("please ignore this", True, "actual_spam"),
    ("aaaaaaaaaaaaaaa bbbbbbb", True, "actual_spam"),
    ("road road road road road road road road road", True, "actual_spam"),
    ("just checking if this works", True, "actual_spam"),
    ("xyz abc 123 test", True, "actual_spam"),
    ("no issue just testing the form", True, "actual_spam"),
    ("dummy submission for demo purposes", True, "actual_spam"),
]


# ---------------------------------------------------------------------------
# Location bias test data
# Format: (location_text, expected_ward_name, ward_tier)
# ward_tier: "common" (in top of TVM_LOCATIONS) | "medium" | "rare"
# ---------------------------------------------------------------------------

LOCATION_TEST_DATA: list[tuple[str, str, str]] = [
    # Common wards (frequently mentioned, likely higher representation in training)
    ("road damage near Pattom junction opposite SUT hospital", "Pattom", "common"),
    ("water supply problem near Kazhakkoottam Technopark area", "Kazhakkoottam", "common"),
    ("drain blocked near Karamana bridge junction", "Karamana", "common"),
    ("sewage overflow near Medical College hospital gate", "Medical College", "common"),
    ("pothole near Kowdiar Palace junction", "Kowdiar", "common"),
    ("street light broken in Palayam market area", "Palayam", "common"),
    ("tree fell near Vattiyoorkavu junction main road", "Vattiyoorkavu", "common"),
    ("garbage not collected Sreekariyam near engineering college", "Sreekariyam", "common"),

    # Medium wards (present in training but less prominent)
    ("waterlogging in Nalanchira near the bypass road", "Nalanchira", "medium"),
    ("road broken near Nanthancode junction", "Nanthancode", "medium"),
    ("drainage issue near Vazhuthacaud petrol station", "Vazhuthacaud", "medium"),
    ("electrical hazard near Thycaud church area", "Thycaud", "medium"),
    ("pothole near Jagathy bridge entrance", "Jagathy", "medium"),
    ("drain overflow near Kunnukuzhy junction", "Kunnukuzhy", "medium"),
    ("illegal construction near Sasthamangalam lake", "Sasthamangalam", "medium"),
    ("garbage dump near Mannanthala panchayat office", "Mannanthala", "medium"),

    # Rare wards (minimal representation in training data)
    ("pipe burst near Pangode military camp entrance road", "Pangode", "rare"),
    ("drain blocked near Edavakode colony", "Edavakode", "rare"),
    ("tree fallen Kariavattom university campus road", "Kariavattom", "rare"),
    ("road broken near Pangappara junction south", "Pangappara", "rare"),
    ("water supply cut Kachani ward last four days", "Kachani", "rare"),
    ("sewage overflow near Muttada colony road", "Muttada", "rare"),
    ("electrical pole fallen near Ponnumangalam road", "Ponnumangalam", "rare"),
    ("garbage not collected in Melamcode ward", "Melamcode", "rare"),
]


# ---------------------------------------------------------------------------
# Priority bias pairs: same civic issue, neutral vs emotional wording
# Format: (neutral_text, emotional_text, true_priority, category)
# ---------------------------------------------------------------------------

PRIORITY_BIAS_PAIRS: list[tuple[str, str, str, str]] = [
    (
        "Street light on this road has not been working for two weeks.",
        "This is OUTRAGEOUS!!! Street light broken for TWO WEEKS!!! We are SUFFERING!!!",
        "medium",
        "street_light",
    ),
    (
        "Garbage collection has not happened in our ward for eight days.",
        "DISGUSTING!! Garbage rotting for 8 days!! Completely UNACCEPTABLE!! Someone will get sick!!",
        "medium",
        "solid_waste",
    ),
    (
        "Water pressure in our area has been low for the past five days.",
        "WHY is there NO WATER?! Five days of misery!! Children and elderly SUFFERING!!",
        "medium",
        "water_supply",
    ),
    (
        "There is a pothole on the road near the bus stop.",
        "DANGEROUS pothole!! Someone will DIE!! This road is a DEATH TRAP!! Fix it NOW!!",
        "medium",
        "road_damage",
    ),
    (
        "The drain near the market entrance appears to be blocked.",
        "COMPLETELY flooded!! Drain totally blocked!! Our shop is underwater!! HELP!!",
        "high",
        "drainage",
    ),
    (
        "A street light pole near our area is slightly leaning.",
        "POLE ABOUT TO FALL!!! It will CRUSH someone!!! I am TERRIFIED every time I pass!!",
        "high",
        "street_light",
    ),
    (
        "There is a slow leak from the water pipe near the school compound.",
        "PIPE BURST!!! FLOODING!!! Water EVERYWHERE!!! DISASTER!!!",
        "high",
        "water_supply",
    ),
    (
        "Illegal construction noticed adjacent to our plot.",
        "They are DESTROYING our neighbourhood!! Illegal construction EVERYWHERE!! NO PERMITS!! CORRUPTION!!!",
        "high",
        "illegal_construction",
    ),
    (
        "Some tree branches are overhanging the road near the colony.",
        "TREE ABOUT TO FALL!! It WILL kill someone!! I cannot SLEEP with worry!!",
        "high",
        "tree_fall",
    ),
    (
        "Sewage smell noticed near our building from time to time.",
        "The STENCH is UNBEARABLE!!! This is a HEALTH EMERGENCY!!! People are DYING!!!",
        "high",
        "sewage_issue",
    ),
    (
        "A small section of road near the junction has developed a crack.",
        "The road is COLLAPSING!!! Huge SINKHOLE forming!!! Total STRUCTURAL FAILURE!!!",
        "high",
        "road_damage",
    ),
    (
        "The solar street lights in our colony have stopped working.",
        "Our entire colony is in COMPLETE DARKNESS!!! We are PRISONERS in our homes after 8pm!!!",
        "medium",
        "street_light",
    ),
]


# ===========================================================================
# Helpers
# ===========================================================================

def _compute_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
) -> dict:
    """Return precision, recall, F1 per class plus macro averages."""
    from sklearn.metrics import precision_recall_fscore_support  # noqa: PLC0415

    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        lbl: {"precision": float(p[i]), "recall": float(r[i]),
              "f1": float(f[i]), "support": int(s[i])}
        for i, lbl in enumerate(labels)
    }
    mp, mr, mf, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    wp, wr, wf, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    correct = sum(a == b for a, b in zip(y_true, y_pred))
    return {
        "per_class": per_class,
        "macro":    {"precision": float(mp), "recall": float(mr), "f1": float(mf)},
        "weighted": {"precision": float(wp), "recall": float(wr), "f1": float(wf)},
        "accuracy": correct / len(y_true) if y_true else 0.0,
        "n": len(y_true),
        "correct": correct,
    }


def _severity(gap: float) -> str:
    if gap >= 0.15: return "HIGH"
    if gap >= 0.08: return "MEDIUM"
    return "LOW"


def _bar(value: float, width: int = 20) -> str:
    filled = round(value * width)
    return "#" * filled + "." * (width - filled)


# ===========================================================================
# Management command
# ===========================================================================

class Command(BaseCommand):
    help = "Fairness and bias audit for the TVMC transformer ML pipeline."

    def add_arguments(self, parser):
        parser.add_argument("--json", action="store_true", help="Output results as JSON")
        parser.add_argument(
            "--section",
            choices=["language", "category", "department", "spam", "location", "priority", "report"],
            help="Run only one section",
        )

    def handle(self, *args, **options):
        as_json    = options["json"]
        section    = options.get("section")

        self.stdout.write("\n" + "=" * 70)
        self.stdout.write(self.style.SUCCESS("  TVMC ML BIAS & FAIRNESS AUDIT"))
        self.stdout.write("=" * 70)

        # -- Load engine ---------------------------------------------------
        try:
            from apps.ml.transformer_inference import get_transformer_engine  # noqa: PLC0415
            engine = get_transformer_engine()
            if not engine.is_ready:
                self.stdout.write(self.style.ERROR(f"Transformer not ready: {engine.load_error}"))
                sys.exit(1)
            self.stdout.write(self.style.SUCCESS(f"  Engine: READY  backbone={engine.backbone_name}\n"))
        except Exception as exc:  # noqa: BLE001
            self.stdout.write(self.style.ERROR(f"Failed to load transformer: {exc}"))
            sys.exit(1)

        audit_results: dict = {}

        run_all = section is None
        if run_all or section == "language":
            audit_results["language_bias"] = self._section_language(engine)
        if run_all or section == "category":
            audit_results["category_imbalance"] = self._section_category(engine)
        if run_all or section == "department":
            audit_results["department_routing"] = self._section_department(engine)
        if run_all or section == "spam":
            audit_results["spam_bias"] = self._section_spam()
        if run_all or section == "location":
            audit_results["location_bias"] = self._section_location(engine)
        if run_all or section == "priority":
            audit_results["priority_bias"] = self._section_priority(engine)
        if run_all or section == "report":
            self._section_report(audit_results)

        if as_json:
            self.stdout.write("\n" + json.dumps(audit_results, indent=2))

    # ----------------------------------------------------------------------
    # Section 1 -- Language bias
    # ----------------------------------------------------------------------

    def _section_language(self, engine) -> dict:
        self.stdout.write("\n" + "-" * 70)
        self.stdout.write(self.style.HTTP_INFO("  [1/6] LANGUAGE BIAS -- category accuracy per language group"))
        self.stdout.write("-" * 70)

        groups: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))

        for text, true_cat, _dept, lang in CATEGORY_TEST_DATA:
            try:
                pred = engine.predict_category(text)
                groups[lang][0].append(true_cat)
                groups[lang][1].append(pred.label)
            except Exception:  # noqa: BLE001
                pass

        all_categories = sorted({s[1] for s in CATEGORY_TEST_DATA})
        results: dict = {}

        self.stdout.write(
            f"\n  {'Group':<12}  {'N':>4}  {'Correct':>7}  {'Accuracy':>8}  "
            f"{'MacroF1':>8}  Bar"
        )
        self.stdout.write(f"  {'-'*12}  {'-'*4}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*20}")

        f1_by_group: dict[str, float] = {}
        for lang in ["english", "malayalam", "manglish", "mixed"]:
            if lang not in groups or not groups[lang][0]:
                continue
            y_true, y_pred = groups[lang]
            m = _compute_metrics(y_true, y_pred, all_categories)
            f1_by_group[lang] = m["macro"]["f1"]
            results[lang] = m
            acc = m["accuracy"]
            f1  = m["macro"]["f1"]
            bar = _bar(f1)
            color = self.style.SUCCESS if f1 >= 0.85 else (
                self.style.WARNING if f1 >= 0.70 else self.style.ERROR
            )
            self.stdout.write(color(
                f"  {lang:<12}  {m['n']:>4}  {m['correct']:>7}  "
                f"{acc:>7.1%}  {f1:>8.3f}  {bar}"
            ))

        # Gap analysis
        if f1_by_group:
            max_f1 = max(f1_by_group.values())
            min_f1 = min(f1_by_group.values())
            gap    = max_f1 - min_f1
            worst  = min(f1_by_group, key=f1_by_group.get)
            best   = max(f1_by_group, key=f1_by_group.get)

            self.stdout.write(f"\n  Max-Min F1 gap: {gap:.3f}  "
                              f"(best={best} {max_f1:.3f}, worst={worst} {min_f1:.3f})")
            sev = _severity(gap)
            color = self.style.ERROR if sev == "HIGH" else (
                self.style.WARNING if sev == "MEDIUM" else self.style.SUCCESS
            )
            self.stdout.write(color(f"  Language bias severity: {sev}"))
            results["gap"] = gap
            results["severity"] = sev
            results["worst_group"] = worst
            results["best_group"]  = best

        # Per-category breakdown per language
        self.stdout.write(f"\n  {'Category':<25}  {'EN':>6}  {'ML':>6}  {'MG':>6}  {'MX':>6}")
        self.stdout.write(f"  {'-'*25}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")
        for cat in all_categories:
            row = f"  {cat:<25}"
            for lang_key in ["english", "malayalam", "manglish", "mixed"]:
                if lang_key in results and cat in results[lang_key].get("per_class", {}):
                    f1v = results[lang_key]["per_class"][cat]["f1"]
                    cell = f"{f1v:>6.2f}"
                    if f1v < 0.70:
                        cell = self.style.ERROR(cell)
                    elif f1v < 0.85:
                        cell = self.style.WARNING(cell)
                else:
                    cell = f"{'N/A':>6}"
                row += f"  {cell}"
            self.stdout.write(row)

        return results

    # ----------------------------------------------------------------------
    # Section 2 -- Category imbalance
    # ----------------------------------------------------------------------

    def _section_category(self, engine) -> dict:
        self.stdout.write("\n" + "-" * 70)
        self.stdout.write(self.style.HTTP_INFO("  [2/6] CATEGORY IMBALANCE -- per-category P/R/F1/support"))
        self.stdout.write("-" * 70)

        y_true, y_pred = [], []
        for text, true_cat, _dept, _lang in CATEGORY_TEST_DATA:
            try:
                pred = engine.predict_category(text)
                y_true.append(true_cat)
                y_pred.append(pred.label)
            except Exception:  # noqa: BLE001
                pass

        all_cats = sorted({s[1] for s in CATEGORY_TEST_DATA})
        m = _compute_metrics(y_true, y_pred, all_cats)

        self.stdout.write(
            f"\n  {'Category':<25}  {'P':>6}  {'R':>6}  {'F1':>6}  {'Sup':>5}  Status"
        )
        self.stdout.write(f"  {'-'*25}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*5}  {'-'*10}")

        weak_classes = []
        for cat in all_cats:
            pc = m["per_class"].get(cat, {})
            p_v = pc.get("precision", 0.0)
            r_v = pc.get("recall",    0.0)
            f_v = pc.get("f1",        0.0)
            s_v = pc.get("support",   0)
            if f_v < 0.75:
                status = self.style.ERROR("WEAK")
                weak_classes.append(cat)
            elif f_v < 0.85:
                status = self.style.WARNING("FAIR")
            else:
                status = self.style.SUCCESS("GOOD")
            self.stdout.write(
                f"  {cat:<25}  {p_v:>6.3f}  {r_v:>6.3f}  {f_v:>6.3f}  {s_v:>5}  {status}"
            )

        self.stdout.write(
            f"\n  Macro:    P={m['macro']['precision']:.3f}  "
            f"R={m['macro']['recall']:.3f}  F1={m['macro']['f1']:.3f}"
        )
        self.stdout.write(
            f"  Weighted: P={m['weighted']['precision']:.3f}  "
            f"R={m['weighted']['recall']:.3f}  F1={m['weighted']['f1']:.3f}"
        )
        self.stdout.write(f"  Accuracy: {m['accuracy']:.1%}  ({m['correct']}/{m['n']})")

        if weak_classes:
            self.stdout.write(self.style.ERROR(
                f"\n  Weak classes (F1 < 0.75): {', '.join(weak_classes)}"
            ))
        else:
            self.stdout.write(self.style.SUCCESS("\n  No weak classes detected."))

        m["weak_classes"] = weak_classes
        return m

    # ----------------------------------------------------------------------
    # Section 3 -- Department routing bias
    # ----------------------------------------------------------------------

    def _section_department(self, engine) -> dict:
        self.stdout.write("\n" + "-" * 70)
        self.stdout.write(self.style.HTTP_INFO("  [3/6] DEPARTMENT ROUTING -- distribution + confusion"))
        self.stdout.write("-" * 70)

        y_true, y_pred = [], []
        for text, _cat, true_dept, _lang in CATEGORY_TEST_DATA:
            try:
                pred = engine.predict_department(text)
                y_true.append(true_dept)
                y_pred.append(pred.label)
            except Exception:  # noqa: BLE001
                pass

        all_depts = sorted(set(y_true) | set(y_pred))
        m = _compute_metrics(y_true, y_pred, all_depts)

        # Predicted distribution
        from collections import Counter  # noqa: PLC0415
        pred_dist = Counter(y_pred)
        true_dist = Counter(y_true)
        total = len(y_pred) or 1

        self.stdout.write(
            f"\n  {'Department':<14}  {'True%':>6}  {'Pred%':>6}  {'Bias':>7}  "
            f"{'F1':>6}  Distribution bar (predicted)"
        )
        self.stdout.write(f"  {'-'*14}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*6}  {'-'*25}")

        results: dict = {"per_dept": {}, "accuracy": m["accuracy"], "n": m["n"]}
        for dept in all_depts:
            t_pct = true_dist.get(dept, 0) / total
            p_pct = pred_dist.get(dept, 0) / total
            bias  = p_pct - t_pct
            f1v   = m["per_class"].get(dept, {}).get("f1", 0.0)
            bar   = _bar(p_pct, 25)
            bias_str = f"{bias:>+.3f}"
            if abs(bias) > 0.08:
                bias_str = self.style.WARNING(bias_str)
            self.stdout.write(
                f"  {dept:<14}  {t_pct:>6.1%}  {p_pct:>6.1%}  {bias_str:>7}  "
                f"{f1v:>6.3f}  {bar}"
            )
            results["per_dept"][dept] = {"true_pct": t_pct, "pred_pct": p_pct,
                                          "bias": bias, "f1": f1v}

        self.stdout.write(f"\n  Overall department accuracy: {m['accuracy']:.1%}")

        # Confusion matrix (compact -- only misclassifications)
        from sklearn.metrics import confusion_matrix  # noqa: PLC0415
        cm = confusion_matrix(y_true, y_pred, labels=all_depts)
        self.stdout.write(f"\n  Confusion matrix (rows=true, cols=predicted):")
        header = "  " + " " * 14 + "".join(f"{d[:6]:>8}" for d in all_depts)
        self.stdout.write(header)
        for i, dept in enumerate(all_depts):
            row_vals = "".join(f"{cm[i][j]:>8}" for j in range(len(all_depts)))
            flag = self.style.ERROR if cm[i][i] < cm[i].sum() * 0.5 and cm[i].sum() > 0 else str
            self.stdout.write(flag(f"  {dept:<14}" + row_vals))

        results["macro_f1"] = m["macro"]["f1"]
        return results

    # ----------------------------------------------------------------------
    # Section 4 -- Spam false positive bias
    # ----------------------------------------------------------------------

    def _section_spam(self) -> dict:
        self.stdout.write("\n" + "-" * 70)
        self.stdout.write(self.style.HTTP_INFO("  [4/6] SPAM FALSE POSITIVES -- genuine complaints flagged as spam"))
        self.stdout.write("-" * 70)

        from apps.ml.analyzer import analyze_complaint  # noqa: PLC0415

        subcategory_results: dict[str, dict] = defaultdict(lambda: {"n": 0, "fp": 0, "fn": 0, "examples": []})

        for text, expected_spam, subcat in SPAM_TEST_DATA:
            try:
                result   = analyze_complaint(text)
                is_spam  = result["spam"]["is_spam"]
                spam_score = float(result["spam"].get("spam_score", 0.0))

                subcat_r = subcategory_results[subcat]
                subcat_r["n"] += 1

                if expected_spam is False and is_spam is True:
                    subcat_r["fp"] += 1
                    subcat_r["examples"].append((text[:60], spam_score))
                elif expected_spam is True and is_spam is False:
                    subcat_r["fn"] += 1
                    subcat_r["examples"].append((text[:60], spam_score))
            except Exception:  # noqa: BLE001
                pass

        self.stdout.write(
            f"\n  {'Subcategory':<20}  {'N':>4}  {'FP/FN':>6}  {'Rate':>7}  Status"
        )
        self.stdout.write(f"  {'-'*20}  {'-'*4}  {'-'*6}  {'-'*7}  {'-'*10}")

        overall_fp = 0
        overall_genuine = 0
        results: dict = {}

        for subcat in ["short_genuine", "typo", "poor_grammar", "emotional", "actual_spam"]:
            r = subcategory_results.get(subcat, {"n": 0, "fp": 0, "fn": 0, "examples": []})
            n   = r["n"]
            err = r["fp"] if subcat != "actual_spam" else r["fn"]
            rate = err / n if n > 0 else 0.0
            label = "FP" if subcat != "actual_spam" else "FN"

            if subcat != "actual_spam":
                overall_fp      += r["fp"]
                overall_genuine += n

            if rate == 0.0:
                status = self.style.SUCCESS("GOOD")
            elif rate <= 0.10:
                status = self.style.WARNING("CONCERN")
            else:
                status = self.style.ERROR("HIGH RISK")

            self.stdout.write(
                f"  {subcat:<20}  {n:>4}  {err:>3}{label:>3}  {rate:>6.1%}  {status}"
            )

            results[subcat] = {"n": n, "error_count": err, "rate": rate}

            # Print up to 3 failure examples
            for ex_text, ex_score in r["examples"][:3]:
                self.stdout.write(
                    self.style.WARNING(f"    X score={ex_score:.3f}  \"{ex_text}\"")
                )

        overall_fp_rate = overall_fp / overall_genuine if overall_genuine else 0.0
        self.stdout.write(
            f"\n  Overall genuine-complaint FP rate: {overall_fp}/{overall_genuine} = "
            + (self.style.ERROR if overall_fp_rate > 0.10 else
               self.style.WARNING if overall_fp_rate > 0.05 else
               self.style.SUCCESS)(f"{overall_fp_rate:.1%}")
        )
        results["overall_fp_rate"] = overall_fp_rate
        results["severity"] = _severity(overall_fp_rate)
        return results

    # ----------------------------------------------------------------------
    # Section 5 -- Location bias
    # ----------------------------------------------------------------------

    def _section_location(self, engine) -> dict:
        self.stdout.write("\n" + "-" * 70)
        self.stdout.write(self.style.HTTP_INFO("  [5/6] LOCATION BIAS -- common vs rare ward detection"))
        self.stdout.write("-" * 70)

        tier_results: dict[str, dict] = {
            "common": {"n": 0, "top1": 0, "top3": 0, "scores": []},
            "medium": {"n": 0, "top1": 0, "top3": 0, "scores": []},
            "rare":   {"n": 0, "top1": 0, "top3": 0, "scores": []},
        }

        self.stdout.write(
            f"\n  {'Location text (truncated)':<45}  {'Expected':>12}  "
            f"{'Top-1':>12}  {'Score':>6}  Hit?"
        )
        self.stdout.write(f"  {'-'*45}  {'-'*12}  {'-'*12}  {'-'*6}  {'-'*5}")

        for loc_text, expected_ward, tier in LOCATION_TEST_DATA:
            try:
                loc_result = engine.find_ward_candidates(loc_text, top_k=5)
                top_names  = [c[0] for c in loc_result.candidates]
                top_scores = [c[1] for c in loc_result.candidates]
                top1_name  = top_names[0] if top_names else ""
                top1_score = top_scores[0] if top_scores else 0.0

                hit_top1 = expected_ward.lower() in top1_name.lower() or top1_name.lower() in expected_ward.lower()
                hit_top3 = any(
                    expected_ward.lower() in n.lower() or n.lower() in expected_ward.lower()
                    for n in top_names[:3]
                )

                tr = tier_results[tier]
                tr["n"]     += 1
                tr["top1"]  += int(hit_top1)
                tr["top3"]  += int(hit_top3)
                tr["scores"].append(top1_score)

                flag = self.style.SUCCESS("  HIT") if hit_top1 else (
                    self.style.WARNING(" TOP3") if hit_top3 else self.style.ERROR(" MISS")
                )
                trunc = loc_text[:44]
                self.stdout.write(
                    f"  {trunc:<45}  {expected_ward:>12}  {top1_name:>12}  "
                    f"{top1_score:>6.3f}  {flag}"
                )
            except Exception as exc:  # noqa: BLE001
                tier_results[tier]["n"] += 1
                self.stdout.write(self.style.ERROR(f"  ERROR: {exc}"))

        self.stdout.write(
            f"\n  {'Tier':<8}  {'N':>4}  {'Top-1 %':>8}  {'Top-3 %':>8}  "
            f"{'Avg Score':>10}  Status"
        )
        self.stdout.write(f"  {'-'*8}  {'-'*4}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*10}")

        results: dict = {}
        top1_by_tier: dict[str, float] = {}

        for tier in ["common", "medium", "rare"]:
            tr = tier_results[tier]
            n  = tr["n"] or 1
            t1 = tr["top1"] / n
            t3 = tr["top3"] / n
            avg_score = sum(tr["scores"]) / len(tr["scores"]) if tr["scores"] else 0.0
            top1_by_tier[tier] = t1
            color = self.style.SUCCESS if t1 >= 0.75 else (
                self.style.WARNING if t1 >= 0.50 else self.style.ERROR
            )
            self.stdout.write(color(
                f"  {tier:<8}  {n:>4}  {t1:>7.1%}  {t3:>7.1%}  {avg_score:>10.3f}  "
                + ("GOOD" if t1 >= 0.75 else ("FAIR" if t1 >= 0.50 else "WEAK"))
            ))
            results[tier] = {"n": n, "top1_acc": t1, "top3_acc": t3, "avg_score": avg_score}

        gap = (top1_by_tier.get("common", 0) - top1_by_tier.get("rare", 0))
        sev = _severity(gap)
        color = self.style.ERROR if sev == "HIGH" else (
            self.style.WARNING if sev == "MEDIUM" else self.style.SUCCESS
        )
        self.stdout.write(color(
            f"\n  Common–Rare top-1 gap: {gap:+.1%}  Severity: {sev}"
        ))
        results["common_rare_gap"] = gap
        results["severity"] = sev
        return results

    # ----------------------------------------------------------------------
    # Section 6 -- Priority inflation bias
    # ----------------------------------------------------------------------

    def _section_priority(self, engine) -> dict:
        self.stdout.write("\n" + "-" * 70)
        self.stdout.write(self.style.HTTP_INFO("  [6/6] PRIORITY BIAS -- emotional wording inflation"))
        self.stdout.write("-" * 70)

        PRIORITY_ORDER = {"low": 0, "medium": 1, "high": 2, "urgent": 3, "critical": 4}

        inflated = 0
        deflated = 0
        same     = 0
        results_list = []

        self.stdout.write(
            f"\n  {'Category':<22}  {'True':>8}  {'Neutral':>8}  {'Emotional':>10}  Delta"
        )
        self.stdout.write(f"  {'-'*22}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*10}")

        for neutral_text, emotional_text, true_prio, category in PRIORITY_BIAS_PAIRS:
            try:
                p_neutral   = engine.predict_priority(neutral_text).label
                p_emotional = engine.predict_priority(emotional_text).label

                on = PRIORITY_ORDER.get(p_neutral, -1)
                oe = PRIORITY_ORDER.get(p_emotional, -1)
                delta = oe - on

                if delta > 0:
                    inflated += 1
                    flag = self.style.ERROR(f"+{delta} INFLATED")
                elif delta < 0:
                    deflated += 1
                    flag = self.style.WARNING(f"{delta} deflated")
                else:
                    same += 1
                    flag = self.style.SUCCESS("=  stable")

                results_list.append({
                    "category": category, "true": true_prio,
                    "neutral": p_neutral, "emotional": p_emotional, "delta": delta,
                })
                self.stdout.write(
                    f"  {category:<22}  {true_prio:>8}  {p_neutral:>8}  {p_emotional:>10}  {flag}"
                )
            except Exception as exc:  # noqa: BLE001
                self.stdout.write(self.style.ERROR(f"  ERROR for {category}: {exc}"))

        n_pairs = len(PRIORITY_BIAS_PAIRS)
        inflation_rate = inflated / n_pairs if n_pairs else 0.0

        self.stdout.write(
            f"\n  Inflation: {inflated}/{n_pairs} pairs ({inflation_rate:.0%})  "
            f"| Same: {same}  Deflated: {deflated}"
        )
        sev = "HIGH" if inflation_rate >= 0.40 else ("MEDIUM" if inflation_rate >= 0.20 else "LOW")
        color = self.style.ERROR if sev == "HIGH" else (
            self.style.WARNING if sev == "MEDIUM" else self.style.SUCCESS
        )
        self.stdout.write(color(f"  Priority inflation severity: {sev}"))

        return {
            "inflated": inflated,
            "same": same,
            "deflated": deflated,
            "inflation_rate": inflation_rate,
            "severity": sev,
            "pairs": results_list,
        }

    # ----------------------------------------------------------------------
    # Section 7 -- Bias report
    # ----------------------------------------------------------------------

    def _section_report(self, audit_results: dict) -> None:
        self.stdout.write("\n" + "=" * 70)
        self.stdout.write(self.style.SUCCESS("  [BIAS AUDIT REPORT]"))
        self.stdout.write("=" * 70)

        findings: list[dict] = []

        # Language bias
        lang = audit_results.get("language_bias", {})
        if lang:
            gap = lang.get("gap", 0.0)
            sev = lang.get("severity", "LOW")
            worst = lang.get("worst_group", "unknown")
            findings.append({
                "check": "Language bias",
                "severity": sev,
                "detail": f"F1 gap across language groups: {gap:.3f}. Worst: {worst}.",
                "fix": (
                    "Augment Manglish/mixed-language training samples. "
                    "Use back-translation to expand minority-language corpus."
                ) if sev != "LOW" else "Acceptable -- language representation is balanced.",
            })

        # Category imbalance
        cat = audit_results.get("category_imbalance", {})
        if cat:
            weak = cat.get("weak_classes", [])
            mac_f1 = cat.get("macro", {}).get("f1", 0.0)
            sev = "HIGH" if weak else ("MEDIUM" if mac_f1 < 0.85 else "LOW")
            findings.append({
                "check": "Category imbalance",
                "severity": sev,
                "detail": (
                    f"Macro F1={mac_f1:.3f}. "
                    + (f"Weak classes: {', '.join(weak)}." if weak else "No weak classes.")
                ),
                "fix": (
                    f"Add 50+ training samples for {', '.join(weak)} using diverse real-world phrasings."
                ) if weak else "Category distribution is acceptable.",
            })

        # Department routing
        dept = audit_results.get("department_routing", {})
        if dept:
            d_acc = dept.get("accuracy", 0.0)
            biased = [d for d, v in dept.get("per_dept", {}).items() if abs(v.get("bias", 0)) > 0.08]
            sev = "HIGH" if d_acc < 0.70 else ("MEDIUM" if d_acc < 0.85 or biased else "LOW")
            findings.append({
                "check": "Department routing bias",
                "severity": sev,
                "detail": (
                    f"Accuracy={d_acc:.1%}. "
                    + (f"Over/under-predicted depts: {', '.join(biased)}." if biased else "Distribution balanced.")
                ),
                "fix": (
                    "Review routing rules for biased departments. "
                    "Consider a dedicated department-routing layer with calibration."
                ) if sev != "LOW" else "Department routing is acceptable.",
            })

        # Spam false positives
        spam = audit_results.get("spam_bias", {})
        if spam:
            fp_rate = spam.get("overall_fp_rate", 0.0)
            sev = spam.get("severity", "LOW")
            # Override severity using actual fp_rate thresholds
            sev = "HIGH" if fp_rate > 0.10 else ("MEDIUM" if fp_rate > 0.05 else "LOW")
            findings.append({
                "check": "Spam false positive bias",
                "severity": sev,
                "detail": f"Genuine-complaint FP rate: {fp_rate:.1%}.",
                "fix": (
                    "Lower spam model's decision threshold. "
                    "Add short-genuine and typo samples as negative spam examples in retraining."
                ) if sev != "LOW" else "Spam detection is not over-triggering on genuine complaints.",
            })

        # Location bias
        loc = audit_results.get("location_bias", {})
        if loc:
            gap = loc.get("common_rare_gap", 0.0)
            sev = loc.get("severity", "LOW")
            common_acc = loc.get("common", {}).get("top1_acc", 0.0)
            rare_acc   = loc.get("rare",   {}).get("top1_acc", 0.0)
            findings.append({
                "check": "Location (ward) bias",
                "severity": sev,
                "detail": (
                    f"Common ward top-1 acc={common_acc:.0%}, "
                    f"Rare ward top-1 acc={rare_acc:.0%}, gap={gap:+.0%}."
                ),
                "fix": (
                    "Pre-encode all 101 TVM ward names in the landmark embeddings. "
                    "Add synthetic location-specific complaint samples for rare wards."
                ) if sev != "LOW" else "Location intelligence coverage is balanced.",
            })

        # Priority inflation
        prio = audit_results.get("priority_bias", {})
        if prio:
            rate = prio.get("inflation_rate", 0.0)
            sev  = prio.get("severity", "LOW")
            findings.append({
                "check": "Priority inflation bias",
                "severity": sev,
                "detail": f"Emotional wording inflated priority in {rate:.0%} of test pairs.",
                "fix": (
                    "Add training samples with emotional language but correct (non-inflated) priority labels. "
                    "Use adversarial examples in priority head training."
                ) if sev != "LOW" else "Priority head is robust to emotional language.",
            })

        # Print findings table
        SEV_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        findings.sort(key=lambda f: SEV_ORDER.get(f["severity"], 99))

        self.stdout.write(f"\n  {'Check':<30}  {'Severity':<8}  Detail")
        self.stdout.write(f"  {'-'*30}  {'-'*8}  {'-'*50}")

        for f in findings:
            sev = f["severity"]
            color = self.style.ERROR if sev == "HIGH" else (
                self.style.WARNING if sev == "MEDIUM" else self.style.SUCCESS
            )
            self.stdout.write(color(f"  {f['check']:<30}  {sev:<8}"))
            self.stdout.write(f"    Detail : {f['detail']}")
            self.stdout.write(f"    Fix    : {f['fix']}")
            self.stdout.write("")

        # Summary
        high_count   = sum(1 for f in findings if f["severity"] == "HIGH")
        medium_count = sum(1 for f in findings if f["severity"] == "MEDIUM")
        low_count    = sum(1 for f in findings if f["severity"] == "LOW")

        self.stdout.write("-" * 70)
        self.stdout.write(
            f"  Summary: {high_count} HIGH  |  {medium_count} MEDIUM  |  {low_count} LOW findings"
        )
        if high_count > 0:
            self.stdout.write(self.style.ERROR(
                f"  ACTION REQUIRED: {high_count} high-severity bias(es) detected. Retrain before production."
            ))
        elif medium_count > 0:
            self.stdout.write(self.style.WARNING(
                f"  REVIEW RECOMMENDED: {medium_count} medium-severity bias(es). Plan a targeted data fix."
            ))
        else:
            self.stdout.write(self.style.SUCCESS(
                "  AUDIT PASSED: All checks at LOW severity. Model is reasonably fair."
            ))
        self.stdout.write("=" * 70 + "\n")
