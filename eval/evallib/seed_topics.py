"""Curated seed topics for the golden set — the human-authored starting point.

These are real, recognisable themes that ran through European parliaments between
1996 and 2024. Each becomes a golden item once `build_golden_set.py` has *verified*
it against the actual parsed corpus (match counts per country).

Cross-lingual judging without a translator:
    The corpus is in each parliament's native language, so plain English keywords
    would only match the English (GB) speeches. Instead the patterns lean on tokens
    that actually recur across languages:
      * international proper nouns / borrowings — Ukraine, COVID, NATO, Brexit, euro,
        Schengen, Putin — often with Cyrillic/Greek spellings added (UA/RS/BG/GR);
      * shared Latin/Greek-rooted stems — "migr", "energ", "infla", "pandemi",
        "demokrat", "terror", "reform", "klima"/"clim", "pensi", "vacc"/"vakc"/"vakz".
    The builder's per-country coverage report shows which patterns are genuinely
    multilingual (hits in many countries) vs. effectively English-only, and that
    evidence is what we curate against.

Schema per item (the builder fills in the `corpus_*` verification fields):
    id, question, topic_domain, patterns[] (ALL must match; each is an OR-regex),
    strong_patterns[] (optional -> grade 2), countries (optional whitelist or None),
    date_from/date_to (optional ISO window).
"""

SEED_TOPICS = [
    # --- Pan-European, anchored on international tokens (work cross-lingually) ---
    {
        "id": "ukraine_invasion_2022",
        "question": "How did members of parliament respond to Russia's full-scale invasion of Ukraine in 2022?",
        "topic_domain": "International Affairs",
        "patterns": [
            r"ukrain|ukrajin|ukrayin|ucrai|ucrania|ucrân|ουκραν|україн|украин|украйн",
            r"invasion|invad|invas|aggress|\bwar\b|guerr|krieg|війн|война|wojn|πόλεμ|troops|"
            r"putin|путін|путин|πούτιν|napad|invaze|invázia|inwazj",
        ],
        "strong_patterns": [r"invasion of ukraine|russian aggression|invaze na ukrajin|napad na ukrajin"],
        "countries": None,
        "date_from": "2022-02-01",
        "date_to": "2024-12-31",
    },
    {
        "id": "covid_pandemic_response",
        "question": "What measures did parliaments debate in response to the COVID-19 pandemic?",
        "topic_domain": "Health",
        "patterns": [r"covid|corona|sars-cov|pandemi|παν[δ]ημ|пандеми|pandémi"],
        "countries": None,
        "date_from": "2020-01-01",
        "date_to": "2022-12-31",
    },
    {
        "id": "covid_vaccination",
        "question": "What positions did MPs take on COVID-19 vaccination and vaccine mandates?",
        "topic_domain": "Health",
        "patterns": [
            r"vacc|vakc|vakz|vaksi|εμβολ|вакцин|aşı",
            r"covid|corona|pandemi|παν[δ]ημ|пандеми",
        ],
        "countries": None,
        "date_from": "2020-06-01",
        "date_to": "2022-12-31",
    },
    {
        "id": "brexit",
        "question": "How was the United Kingdom's withdrawal from the European Union discussed?",
        "topic_domain": "International Affairs",
        "patterns": [r"brexit|withdrawal agreement|leave the eu|leave the european union|odchod.{0,10}eu"],
        "countries": None,
        "date_from": "2016-01-01",
        "date_to": "2020-12-31",
    },
    {
        "id": "greek_debt_crisis",
        "question": "What was said about Greece's debt crisis and the eurozone bailouts?",
        "topic_domain": "Macroeconomics",
        "patterns": [
            r"greece|greek|hellenic|griechen|grèce|grecia|grčk|grčij|ελλάδ|ελλην|гръц|grčka",
            r"bailout|debt|\beuro|evro|rescue|austerity|memorandum|schuld|deficit|divid|deud|"
            r"debit|dluh|dług|dolg|долг|χρέος|adóss",
        ],
        "countries": None,
        "date_from": "2010-01-01",
        "date_to": "2015-12-31",
    },
    {
        "id": "nato_membership",
        "question": "How did parliaments discuss NATO membership, accession or the alliance?",
        "topic_domain": "Defense",
        "patterns": [r"\bnato\b|\botan\b|нато|severoatlant|atlanti.{0,4}alianc"],
        "countries": None,
    },
    {
        "id": "schengen_borders",
        "question": "What was debated about the Schengen area and the reintroduction of border controls?",
        "topic_domain": "International Affairs",
        "patterns": [r"schengen|шенген|σένγκεν"],
        "countries": None,
    },
    # --- Pan-European, shared Latin/Greek stems ---
    {
        "id": "migration_asylum",
        "question": "What positions were taken on migration, asylum seekers and refugees?",
        "topic_domain": "Immigration",
        "patterns": [r"migr|asyl|asil|refug|réfugi|flücht|flykt|flygt|invandr|innvandr|indvandr|"
                     r"migrant|μετανάστ|άσυλο|мигра|избегли|uchodźc|menekül|maahanmuut"],
        "countries": None,
    },
    {
        "id": "refugee_crisis_2015",
        "question": "How did parliaments respond to the 2015 surge in refugees and asylum seekers?",
        "topic_domain": "Immigration",
        "patterns": [r"migr|asyl|asil|refug|réfugi|flücht|μετανάστ|άσυλο|мигра|uchodźc|menekül"],
        "strong_patterns": [r"crisis|krise|kríz|crise|crisi|κρίση|криз|kryzys"],
        "countries": None,
        "date_from": "2015-01-01",
        "date_to": "2016-12-31",
    },
    {
        "id": "climate_change",
        "question": "What did MPs say about climate change and emission-reduction targets?",
        "topic_domain": "Environment",
        "patterns": [r"klima|clima|climat|κλίμα|клима|клімат|éghajlat|ilmasto|klíma"],
        "strong_patterns": [r"emission|emisi|emisj|kibocsát|παρίσι|paris agreement|neutral|carbon|co2|co₂"],
        "countries": None,
    },
    {
        "id": "energy_prices_crisis",
        "question": "What was debated about rising energy and gas prices?",
        "topic_domain": "Energy",
        "patterns": [
            r"energ|énerg|ενέργ|енерг",
            r"price|prix|preis|prezz|precio|cena|cijen|ár[a-z]|τιμ|gas|plyn|gáz",
        ],
        "date_from": "2021-06-01",
        "date_to": "2023-12-31",
        "countries": None,
    },
    {
        "id": "inflation_cost_of_living",
        "question": "How did parliaments address inflation and the cost of living?",
        "topic_domain": "Macroeconomics",
        "patterns": [r"infla|infláci|inflácia|inflazion|πληθωρισ|инфла"],
        "countries": None,
    },
    {
        "id": "nuclear_energy",
        "question": "What positions were taken on nuclear power — expansion or phase-out?",
        "topic_domain": "Energy",
        "patterns": [r"nuclear|nuklear|nukleár|nucléair|nucleare|atom|πυρηνικ|ядерн|jádern|jadrov"],
        "countries": None,
    },
    {
        "id": "renewable_energy",
        "question": "What was said about renewable energy such as wind and solar power?",
        "topic_domain": "Energy",
        "patterns": [
            r"renewable|erneuerbar|obnoviteln|obnovljiv|odnawialn|ανανεώσιμ|возобновля|megújul|uusiutuv",
            r"wind|solar|fotovolta|photovolta|солнечн|ηλιακ",
        ],
        "countries": None,
    },
    {
        "id": "terrorism_security",
        "question": "How was the threat of terrorism and counter-terrorism policy debated?",
        "topic_domain": "Defense",
        "patterns": [r"terror|τρομοκρατ|террор|terorist|terrorism"],
        "countries": None,
    },
    {
        "id": "corruption",
        "question": "What was said about corruption and anti-corruption measures?",
        "topic_domain": "Government Operations",
        "patterns": [r"corrup|korrup|korup|διαφθορ|коррупц|корупц|yolsuzluk"],
        "countries": None,
    },
    {
        "id": "pension_reform",
        "question": "What positions were taken on pension reform and the retirement age?",
        "topic_domain": "Social Welfare",
        "patterns": [
            r"pensi|retrait|rente|penzij|nyugdíj|σύνταξ|пенси|пенсі|eläke",
            r"reform|retirement age|retraite|rentenalter|odchod do důchod|věk|âge",
        ],
        "countries": None,
    },
    {
        "id": "minimum_wage",
        "question": "What did MPs say about introducing or raising a statutory minimum wage?",
        "topic_domain": "Labor",
        "patterns": [r"minimum wage|mindestlohn|salaire minimum|salario mínimo|minimáln.{0,6}mzd|minimaln.{0,6}plač|κατώτατ.{0,8}μισθ|минимальн.{0,6}зарплат|minimálbér"],
        "countries": None,
    },
    {
        "id": "gender_equality",
        "question": "How was gender equality and the gender pay gap discussed?",
        "topic_domain": "Civil Rights",
        "patterns": [
            r"gender|gleichstellung|rovnost|enakost|égalité.{0,12}femme|igualdad.{0,12}mujer|ισότητα.{0,12}φύλ|равноправ|nemek között",
            r"women|frauen|femme|mujer|γυναικ|жен|женск|nők|ženy",
        ],
        "countries": None,
    },
    {
        "id": "eu_enlargement",
        "question": "What was debated about European Union enlargement and accession of candidate countries?",
        "topic_domain": "International Affairs",
        "patterns": [
            # EU name/token — broad multilingual stems (europ/evrop covers European,
            # Europäische, Europese, Europejsk, Evropsk…) plus the EU abbreviation.
            r"europ|evrop|ευρωπ|європ|европ|\beu\b|\beú\b|\bue\b",
            # enlargement / accession of candidate countries, across languages.
            r"enlarge|erweiter|rozšíř|rozsir|razšir|razsir|prošir|accession|beitritt|"
            r"přistoup|pristop|pristup|προσχώρησ|adhési|adhesión|adesione|bővít|"
            r"laajentum|udvidels|utvidels|разшир|приєдн|candidate|kandidat",
        ],
        "countries": None,
    },
    # --- Country-anchored items (native-language terms, single parliament) ---
    {
        "id": "at_eurofighter",
        "question": "What criticism was raised in the Austrian parliament about the Eurofighter procurement?",
        "topic_domain": "Defense",
        "patterns": [r"eurofighter"],
        "countries": ["AT"],
    },
    {
        "id": "at_ibiza_affair",
        "question": "How was the Ibiza affair and the fall of the government debated in Austria?",
        "topic_domain": "Government Operations",
        "patterns": [r"\bibiza\b"],
        "countries": ["AT"],
        "date_from": "2019-05-01",
        "date_to": "2019-12-31",
    },
    {
        "id": "gb_brexit_referendum",
        "question": "How did the UK Parliament react to the EU referendum result?",
        "topic_domain": "International Affairs",
        "patterns": [r"brexit|referendum", r"european union|\beu\b|leave|remain"],
        "countries": ["GB"],
        "date_from": "2016-06-01",
        "date_to": "2017-12-31",
    },
    {
        "id": "gb_minimum_wage",
        "question": "What did members of the UK Parliament say about the national minimum wage?",
        "topic_domain": "Labor",
        "patterns": [r"minimum wage", r"national living wage|low pay|workers|employ"],
        "countries": ["GB"],
    },
    {
        "id": "fr_pension_reform",
        "question": "What was debated in the French parliament about pension reform and the retirement age?",
        "topic_domain": "Social Welfare",
        "patterns": [r"retraite", r"réforme|âge|cotisation|régime"],
        "countries": ["FR"],
    },
    {
        "id": "gr_austerity_memorandum",
        "question": "How did the Greek parliament debate the austerity memoranda during the debt crisis?",
        "topic_domain": "Macroeconomics",
        "patterns": [r"μνημόνιο|λιτότητ|τρόικα|μνημονί"],
        "countries": ["GR"],
    },
    # --- Additional pan-European topics (single broad multilingual stem each, which the
    #     eu_enlargement lesson showed is more robust than two co-occurring tokens) ---
    {
        "id": "healthcare_system",
        "question": "What did MPs say about funding the public health system and healthcare services?",
        "topic_domain": "Health",
        "patterns": [r"health|healthcare|hospital|\bnhs\b|gesundheit|gezondheid|zdravot|zdravst|sanit|"
                     r"santé|saúde|salud|salute|υγεί|здрав|terveyden|sjukvård|helse|sundhed|"
                     r"egészségügy|sağlık|sănăt|zdrowot"],
        "countries": None,
    },
    {
        "id": "education_policy",
        "question": "What positions were taken on education policy, schools and universities?",
        "topic_domain": "Education",
        "patterns": [r"education|\bschool|universit|college|onderwijs|bildung|vzdělá|vzděláv|"
                     r"izobraž|obrazov|éducation|educaci|educaç|istruzione|εκπαίδευ|освіт|"
                     r"образован|koulutus|utdann|utbildning|oktatás|eğitim|educați|edukacj|"
                     r"školst|školstv"],
        "countries": None,
    },
    {
        "id": "taxation",
        "question": "What was debated about taxes and tax policy?",
        "topic_domain": "Macroeconomics",
        "patterns": [r"steuer|taxation|\btaxes\b|income tax|belasting|imposto|\bdaň|\bdaní|davč|"
                     r"\bporez|impôt|impuesto|imposta|φορολογ|φόρο|налог|verot|skatte|adóz|"
                     r"impozit|podatk"],
        "countries": None,
    },
    {
        "id": "agriculture_farming",
        "question": "What did MPs say about agriculture, farming and support for farmers?",
        "topic_domain": "Agriculture",
        "patterns": [r"agricultur|landwirtschaft|zeměděl|kmetijstv|poljoprivred|αγροτ|γεωργ|"
                     r"сільськ|сельск.{0,4}хозяй|maatalous|jordbruk|landbruk|mezőgazda|"
                     r"tarım|agricol|rolnictw|\bfarm"],
        "countries": None,
    },
    {
        "id": "housing",
        "question": "What was debated about housing, rents and affordability?",
        "topic_domain": "Housing",
        "patterns": [r"housing|affordab|tenant|rental|\brents\b|wohnung|wohnbau|woning|woon|"
                     r"\bhuur|\bhuren|habitaç|arrendament|bydlen|stanovanj|stambeno|logement|"
                     r"vivienda|alloggi|στέγασ|κατοικ|житло|жиль|asunto|\bbolig|bostad|lakás|"
                     r"konut|locuinț|mieszkani"],
        "countries": None,
    },
    {
        "id": "unemployment",
        "question": "How was unemployment and joblessness addressed?",
        "topic_domain": "Labor",
        "patterns": [r"unemploy|jobless|werkloos|werkloze|arbeitslos|nezaměstnan|brezposel|"
                     r"nezaposlen|chômage|desemple|disoccupa|ανεργ|безробіт|безработ|työttöm|"
                     r"arbeidsled|arbetslös|munkanélkül|işsizlik|şomaj|bezroboc"],
        "countries": None,
    },

    # ===== Expansion batch (broad multilingual stems; English always included) =====
    # --- Fiscal / economy ---
    {"id": "public_debt", "question": "What was debated about the level of public and national debt?",
     "topic_domain": "Macroeconomics", "countries": None,
     "patterns": [r"public debt|national debt|sovereign debt|staatsschuld|öffentliche schuld|"
                  r"dette publique|deuda públic|debito pubblic|veřejný dluh|javni dolg|javni dug|"
                  r"δημόσιο χρέος|державний борг|государственн.{0,4}долг|statsgæld|statsskuld|"
                  r"államadósság|dług publiczn"]},
    {"id": "budget_deficit", "question": "How were budget deficits and balancing the budget discussed?",
     "topic_domain": "Macroeconomics", "countries": None,
     "patterns": [r"budget deficit|fiscal deficit|haushaltsdefizit|déficit budgétaire|"
                  r"déficit públic|disavanzo|rozpočtov.{0,6}schod|proračunsk.{0,8}primanjkljaj|"
                  r"proračunsk.{0,8}deficit|έλλειμμα|бюджетн.{0,6}дефіцит|дефицит бюджет|"
                  r"alijäämä|budgetunderskott|költségvetési hiány|deficyt budżet"]},
    {"id": "vat_taxation", "question": "What positions were taken on value-added tax (VAT) rates?",
     "topic_domain": "Macroeconomics", "countries": None,
     "patterns": [r"value added tax|\bvat\b|mehrwertsteuer|\btva\b|\biva\b|\bdph\b|\bddv\b|\bpdv\b|"
                  r"φπα|\bпдв\b|\bндс\b|\balv\b|\bmoms\b|áfa"]},
    {"id": "privatization", "question": "What was debated about privatising state-owned enterprises?",
     "topic_domain": "Macroeconomics", "countries": None,
     "patterns": [r"privatis|privatiz|privatizac|приватизац|özelleştir|magánosít"]},
    {"id": "state_aid", "question": "How were state aid and government subsidies to companies discussed?",
     "topic_domain": "Macroeconomics", "countries": None,
     "patterns": [r"state aid|state subsid|staatliche beihilf|subvention|aide d'état|ayuda estatal|"
                  r"državna pomoč|državn.{0,4}potpor|κρατικ.{0,6}ενίσχυσ|державн.{0,4}допомог|"
                  r"állami támogatás|statsstöd|valtiontuki"]},
    {"id": "trade_agreement", "question": "What was said about free-trade agreements?",
     "topic_domain": "Foreign Trade", "countries": None,
     "patterns": [r"free trade|trade agreement|freihandel|libre-échange|libre comercio|"
                  r"wolny handel|prosti trgovin|slobodn.{0,4}trgovin|ελεύθερο εμπόριο|"
                  r"вільна торгівл|свободн.{0,4}торговл|frihandel|szabadkereskede|vapaakauppa"]},
    {"id": "child_benefit", "question": "What was debated about child benefit and family allowances?",
     "topic_domain": "Social Welfare", "countries": None,
     "patterns": [r"child benefit|child allowance|family allowance|kindergeld|allocation.{0,12}famil|"
                  r"prestación.{0,12}hij|assegno.{0,10}figli|otroški dodatek|dječji doplatak|"
                  r"допомог.{0,8}дітей|családi pótlék|barnbidrag|lapsilisä"]},
    {"id": "banking_union", "question": "How was European banking regulation and the banking union discussed?",
     "topic_domain": "Macroeconomics", "countries": None,
     "patterns": [r"banking union|bankenunion|union bancaire|unión bancaria|banková únia|"
                  r"bančna unija|банківськ.{0,4}союз|банковск.{0,4}союз|bank supervis|"
                  r"bankenaufsicht|supervisión bancaria|bankunion"]},
    # --- Social / rights ---
    {"id": "abortion", "question": "What positions were taken on abortion law?",
     "topic_domain": "Civil Rights", "countries": None,
     "patterns": [r"abortion|abtreibung|\bavort|aborto|interrupcij|\bpotrat|\bsplav|έκτρωση|"
                  r"аборт|\babort|terhességmegszakít|kürtaj"]},
    {"id": "lgbt_rights", "question": "How were same-sex partnership and LGBT rights debated?",
     "topic_domain": "Civil Rights", "countries": None,
     "patterns": [r"same-sex|same sex|gay marriage|marriage equality|gleichgeschlecht|"
                  r"homosexuell|mariage.{0,12}homosexuel|matrimonio.{0,12}homosexual|istospoln|"
                  r"однополы|одностатев|azonos nemű|\blgbt|homoseksual|eşcinsel"]},
    {"id": "domestic_violence", "question": "What was debated about domestic violence and violence against women?",
     "topic_domain": "Civil Rights", "countries": None,
     "patterns": [r"domestic violence|violence against women|häusliche gewalt|gewalt gegen frauen|"
                  r"violence.{0,12}femme|violencia.{0,12}género|violencia.{0,12}mujer|"
                  r"nasilje v družini|nasilje nad ženama|насильство.{0,10}жінок|"
                  r"насилие в семь|perheväkivalta|kvinnovåld|nők elleni erőszak"]},
    {"id": "disability_rights", "question": "How were the rights and support of people with disabilities discussed?",
     "topic_domain": "Civil Rights", "countries": None,
     "patterns": [r"disabilit|disabled people|behinder|handicap|discapacid|invalidnost|"
                  r"osoba.{0,6}invalid|інвалідн|инвалидн|fogyaték|vammais|"
                  r"funktionsnedsätt|engelli birey"]},
    {"id": "child_poverty", "question": "What was said about child poverty?",
     "topic_domain": "Social Welfare", "countries": None,
     "patterns": [r"child poverty|children.{0,6}poverty|kinderarmut|pauvreté.{0,10}enfant|"
                  r"pobreza infantil|otroška revščina|dječj.{0,8}siromaš|дитяч.{0,6}бідн|"
                  r"gyermekszegény|lapsiköyhyys|barnfattigdom"]},
    {"id": "drug_policy", "question": "What positions were taken on drug policy and narcotics?",
     "topic_domain": "Law and Crime", "countries": None,
     "patterns": [r"\bdrug|narcotic|\bdroge|stupéfiant|narkotik|\bdroga|наркотик|наркоман|"
                  r"kábítószer|uyuşturucu|narcótic|cannabis|marihuana|huumaus"]},
    {"id": "gambling", "question": "How was gambling and its regulation debated?",
     "topic_domain": "Law and Crime", "countries": None,
     "patterns": [r"gambling|glücksspiel|jeux de hasard|juego.{0,8}azar|igre na srečo|kockanj|"
                  r"азартн|szerencsejáték|kumar|rahapeli|spelmissbruk"]},
    {"id": "tobacco_smoking", "question": "What was debated about tobacco and smoking restrictions?",
     "topic_domain": "Health", "countries": None,
     "patterns": [r"tobacco|smoking|\btabak|\btabac|tabaco|kajenj|pušenj|курінн|курени|"
                  r"dohányz|tütün|tupakka|rökning"]},
    {"id": "homelessness", "question": "How was homelessness addressed?",
     "topic_domain": "Social Welfare", "countries": None,
     "patterns": [r"homeless|rough sleep|obdachlos|sans-abri|sin hogar|sin techo|brezdomstvo|"
                  r"beskućni|бездомн|hajléktalan|evsiz|asunnotto|hemlös"]},
    {"id": "mental_health", "question": "What was said about mental health and psychological care?",
     "topic_domain": "Health", "countries": None,
     "patterns": [r"mental health|psychische gesundheit|santé mentale|salud mental|"
                  r"duševno zdravje|mentalno zdravlje|психічн.{0,6}здоров|психическ.{0,6}здоров|"
                  r"mielenterveys|mentális egészség|psykisk hälsa"]},
    # --- Environment / energy ---
    {"id": "air_pollution", "question": "What was debated about air pollution?",
     "topic_domain": "Environment", "countries": None,
     "patterns": [r"air pollution|air quality|luftverschmutzung|luftqualität|pollution.{0,10}air|"
                  r"contaminación.{0,10}aire|onesnaž.{0,6}zrak|zagađenj.{0,6}zrak|"
                  r"забруднення повітря|загрязнение воздуха|ilmansaaste|levegőszennyez|luftföroren"]},
    {"id": "biodiversity", "question": "How was biodiversity and nature protection discussed?",
     "topic_domain": "Environment", "countries": None,
     "patterns": [r"biodiversit|biodiverzit|biotska raznovrstnost|bioraznolikost|"
                  r"біорізноманіт|биоразнообраз|luonnon monimuotois|biológiai sokféleség"]},
    {"id": "deforestation_forests", "question": "What positions were taken on forests and deforestation?",
     "topic_domain": "Environment", "countries": None,
     "patterns": [r"deforest|forest protection|\bwald|\bforêt|\bbosque|\bgozd|\bšuma|\bліс|"
                  r"\bлес[аоу]|\bmetsä|\berdő|\borman|\bskog"]},
    {"id": "waste_recycling", "question": "What was debated about waste management and recycling?",
     "topic_domain": "Environment", "countries": None,
     "patterns": [r"recycl|recikl|waste management|\babfall|\bdéchet|residuo|\bodpad|\botpad|"
                  r"відход|отход|\bjäte|hulladék|\batık|återvinning"]},
    {"id": "electric_vehicles", "question": "How were electric vehicles and the transition away from combustion engines debated?",
     "topic_domain": "Energy", "countries": None,
     "patterns": [r"electric vehicle|electric car|elektrofahrzeug|elektroauto|"
                  r"véhicule électrique|vehículo eléctrico|električn.{0,8}vozil|"
                  r"електромобіл|электромобил|sähköauto|elbil|elektromos.{0,8}jármű"]},
    {"id": "water_supply", "question": "What was debated about drinking-water supply and quality?",
     "topic_domain": "Environment", "countries": None,
     "patterns": [r"drinking water|trinkwasser|eau potable|agua potable|pitna voda|питна вод|"
                  r"питьев.{0,6}вод|ivóvíz|juomavesi|dricksvatten|water supply|wasserversorgung"]},
    # --- Justice / governance ---
    {"id": "rule_of_law", "question": "How was the rule of law and judicial independence discussed?",
     "topic_domain": "Law and Crime", "countries": None,
     "patterns": [r"rule of law|judicial independence|rechtsstaat|unabhängigkeit der justiz|"
                  r"état de droit|estado de derecho|pravna država|neodvisnost sodstva|"
                  r"vladavina prava|верховенств.{0,6}прав|правова держав|jogállam|hukuk devleti|"
                  r"oikeusvaltio|rättsstat|"
                  r"independen.{0,18}(?:justice|magistrat|judiciaire|judicial|tribunal|rechter|"
                  r"rechtspraak|sodstv|sud)|onafhankelijkheid.{0,16}(?:rechter|rechtspraak|rechtsp)"]},
    {"id": "press_freedom", "question": "What was said about press and media freedom?",
     "topic_domain": "Civil Rights", "countries": None,
     "patterns": [r"press freedom|media freedom|freedom of the press|pressefreiheit|"
                  r"liberté.{0,10}presse|libertad.{0,10}prensa|svoboda tiska|sloboda medij|"
                  r"свобода прес|свобода сло|sajtószabadság|basın özgürlüğü|lehdistönvapaus"]},
    {"id": "death_penalty", "question": "What positions were taken on the death penalty?",
     "topic_domain": "Law and Crime", "countries": None,
     "patterns": [r"death penalty|capital punishment|todesstrafe|peine de mort|pena de muerte|"
                  r"pena capital|smrtna kazen|smrtna kazna|смертна кара|смертн.{0,4}казн|"
                  r"halálbüntetés|idam cezas|dödsstraff|kuolemanrangaistus"]},
    {"id": "data_protection_gdpr", "question": "How were data protection and privacy debated?",
     "topic_domain": "Civil Rights", "countries": None,
     "patterns": [r"data protection|\bgdpr\b|datenschutz|protection.{0,10}données|"
                  r"protección.{0,10}datos|varstvo podatkov|zaštita podataka|захист даних|"
                  r"защит.{0,6}данн|adatvédelem|tietosuoja|dataskydd|kişisel veri"]},
    {"id": "human_trafficking", "question": "What was debated about human trafficking?",
     "topic_domain": "Law and Crime", "countries": None,
     "patterns": [r"human trafficking|trafficking in human|menschenhandel|traite.{0,10}humain|"
                  r"traite.{0,10}être humain|trata.{0,10}personas|trgovina z ljudmi|"
                  r"trgovina ljudima|торгівля людьми|торговл.{0,6}людьм|emberkereskedelem|"
                  r"insan ticareti|ihmiskauppa"]},
    {"id": "prison_reform", "question": "How were prisons and the penal system discussed?",
     "topic_domain": "Law and Crime", "countries": None,
     "patterns": [r"\bprison|penitentiary|gefängnis|strafvollzug|détention|\bcárcel|\bprisión|"
                  r"\bzapor|\bzatvor|в'язниц|тюрьм|\bbörtön|hapishane|\bfängelse|\bvankila"]},
    {"id": "electoral_reform", "question": "What was debated about electoral law and the voting system?",
     "topic_domain": "Government Operations", "countries": None,
     "patterns": [r"electoral law|electoral reform|voting system|wahlrecht|wahlsystem|"
                  r"loi électorale|sistema electoral|ley electoral|volilni sistem|izborn.{0,6}zakon|"
                  r"виборч.{0,6}закон|избирательн.{0,6}закон|választási.{0,6}törvény|seçim yasas|"
                  r"vaalilaki|valsystem"]},
    {"id": "transparency", "question": "How were transparency and open government discussed?",
     "topic_domain": "Government Operations", "countries": None,
     "patterns": [r"transparency|transparenz|transparence|transparencia|transparentnost|"
                  r"прозорість|прозрачност|átláthatóság|şeffaflık|avoimuus|öppenhet"]},
    {"id": "decentralization", "question": "What was debated about decentralisation and regional self-government?",
     "topic_domain": "Government Operations", "countries": None,
     "patterns": [r"decentrali|dezentralis|décentralis|descentraliz|decentralizac|"
                  r"децентраліз|децентрализ|decentralizáció|hajauttaminen"]},
    {"id": "referendum", "question": "How were referendums and direct democracy discussed?",
     "topic_domain": "Government Operations", "countries": None,
     "patterns": [r"referendum|referéndum|volksabstimmung|référendum|népszavazás|референдум|"
                  r"kansanäänestys|folkomröstning|plebiscit"]},
    {"id": "constitutional_reform", "question": "What positions were taken on constitutional reform?",
     "topic_domain": "Government Operations", "countries": None,
     "patterns": [r"constitution|verfassung|\bustav|\bústav|конституці|конституци|alkotmány|"
                  r"anayasa|perustuslaki|grundlag|σύνταγμα"]},
    {"id": "whistleblower", "question": "How was whistleblower protection debated?",
     "topic_domain": "Law and Crime", "countries": None,
     "patterns": [r"whistleblow|hinweisgeber|lanceur.{0,8}alerte|denunciante|žvižgač|zviždač|"
                  r"викривач|информатор|visselblås|ilmiantaja|közérdekű bejelent"]},
    # --- Foreign / defence ---
    {"id": "sanctions", "question": "What was debated about economic sanctions?",
     "topic_domain": "International Affairs", "countries": None,
     "patterns": [r"sanction|sanktion|sankci|санкці|санкци|szankció|yaptırım|pakote|sanktioner"]},
    {"id": "development_aid", "question": "How was overseas development aid discussed?",
     "topic_domain": "Foreign Aid", "countries": None,
     "patterns": [r"development aid|development assistance|entwicklungshilfe|"
                  r"aide au développement|ayuda.{0,8}desarrollo|razvojna pomoč|razvojn.{0,4}pomoć|"
                  r"допомог.{0,6}розвитк|fejlesztési segély|kehitysapu|bistånd"]},
    {"id": "arms_exports", "question": "What positions were taken on arms exports?",
     "topic_domain": "Defense", "countries": None,
     "patterns": [r"arms export|weapons export|waffenexport|exportation.{0,10}arme|"
                  r"exportación.{0,10}arma|izvoz orožja|izvoz oružja|експорт зброї|"
                  r"экспорт оруж|fegyverexport|silah ihrac|asevienti|vapenexport"]},
    {"id": "eu_defence", "question": "How was European common defence and security policy discussed?",
     "topic_domain": "Defense", "countries": None,
     "patterns": [r"european defen[cs]e|common defen[cs]e|europäische verteidigung|"
                  r"défense européenne|defensa europea|evropska obramba|europska obran|"
                  r"європейськ.{0,4}оборон|közös védelem|\bpesco\b|common security and defen"]},
    {"id": "cyber_security", "question": "What was debated about cyber-security and cyber threats?",
     "topic_domain": "Defense", "countries": None,
     "patterns": [r"cyber|cybersecurity|cyber-security|cybersicherheit|cybersécurité|"
                  r"ciberseguridad|kibernetsk|кібербезпек|кибербезопасн|kiberbiztonság|"
                  r"siber güvenlik|kyberturvallisuus"]},
    {"id": "peacekeeping", "question": "How were peacekeeping and military missions abroad discussed?",
     "topic_domain": "Defense", "countries": None,
     "patterns": [r"peacekeeping|peace mission|friedensmission|maintien.{0,8}paix|"
                  r"mantenimiento.{0,8}paz|mirovna misija|mirovn.{0,6}operacij|миротворч|"
                  r"békefenntart|rauhanturva|fredsbevarande"]},
    # --- Tech / culture / education ---
    {"id": "research_innovation", "question": "What was said about funding research and innovation?",
     "topic_domain": "Technology", "countries": None,
     "patterns": [r"research and (development|innovation)|\bforschung|\brecherche|investigación|"
                  r"raziskav|istraživanj|дослідженн|исследован|\bkutatás|araştırma|\btutkimus|"
                  r"\bforskning"]},
    {"id": "data_digital_economy", "question": "How were digitalisation and the digital economy discussed?",
     "topic_domain": "Technology", "countries": None,
     "patterns": [r"digitali[sz]|digitalisierung|numérique|digital economy|digitalno gospodarstvo|"
                  r"digitaln.{0,6}ekonom|цифров.{0,6}економ|цифров.{0,6}эконом|digitális gazdaság|"
                  r"dijital ekonomi|digitalisering"]},
    {"id": "minority_language", "question": "What was debated about minority and regional languages?",
     "topic_domain": "Civil Rights", "countries": None,
     "patterns": [r"minority language|regional language|minderheitensprache|"
                  r"langue.{0,10}minoritaire|lengua.{0,10}minoritari|manjšinski jezik|"
                  r"manjinski jezik|мов.{0,6}меншин|kisebbségi nyelv|vähemmistökiel"]},
    {"id": "public_broadcasting", "question": "How was public-service broadcasting funding discussed?",
     "topic_domain": "Technology", "countries": None,
     "patterns": [r"public broadcast|public service broadcast|öffentlich-rechtlich|"
                  r"service public.{0,14}audiovisuel|radiotelevisión pública|"
                  r"javn.{0,4}radiotelevizij|javni servis|суспільне мовленн|общественн.{0,6}вещан|"
                  r"közmédia|közszolgálati média|yleisradio|public service media"]},
    {"id": "cultural_heritage", "question": "What was said about protecting cultural heritage?",
     "topic_domain": "Culture", "countries": None,
     "patterns": [r"cultural heritage|kulturerbe|patrimoine culturel|patrimonio cultural|"
                  r"kulturna dediščina|kulturn.{0,4}baštin|культурн.{0,4}спадщин|"
                  r"культурн.{0,4}наслед|kulturális örökség|kulttuuriperintö|kulturarv"]},
    # --- Infrastructure ---
    {"id": "public_transport", "question": "How was public transport funding and policy discussed?",
     "topic_domain": "Transportation", "countries": None,
     "patterns": [r"public transport|öffentlicher (personen)?(nah)?verkehr|transport.{0,6}public|"
                  r"transports en commun|transporte público|javni promet|javni prijevoz|"
                  r"громадськ.{0,6}транспорт|общественн.{0,6}транспорт|tömegközlekedés|"
                  r"joukkoliikenne|kollektivtrafik"]},
    {"id": "railways", "question": "What was debated about railways and rail investment?",
     "topic_domain": "Transportation", "countries": None,
     "patterns": [r"railway|\brail\b|eisenbahn|chemin de fer|ferrocarril|železnic|željeznic|"
                  r"залізниц|железн.{0,4}дорог|\bvasút|demiryolu|rautatie|järnväg"]},
    {"id": "broadband_internet", "question": "How were broadband rollout and internet access discussed?",
     "topic_domain": "Technology", "countries": None,
     "patterns": [r"broadband|breitband|haut débit|banda ancha|širokopasovn|širokopojasn|"
                  r"широкосмугов|широкополосн|szélessáv|laajakaista|bredband|fiber optic|"
                  r"internet access"]},
    {"id": "road_safety", "question": "What was debated about road and traffic safety?",
     "topic_domain": "Transportation", "countries": None,
     "patterns": [r"road safety|traffic safety|verkehrssicherheit|sécurité routière|"
                  r"seguridad vial|prometna varnost|sigurnost.{0,6}promet|безпек.{0,6}дорож|"
                  r"közlekedésbiztonság|trafik(en)?säkerhet|liikenneturvallisuus"]},
    {"id": "housing_social", "question": "What was said about social and affordable housing programmes?",
     "topic_domain": "Housing", "countries": None,
     "patterns": [r"social housing|affordable housing|sozialwohnung|sozialer wohnungsbau|"
                  r"logement social|vivienda social|neprofitn.{0,6}stanovanj|socijaln.{0,6}stanov|"
                  r"соціальн.{0,4}житл|социальн.{0,4}жиль|szociális lakás|allmännyttig"]},
    # --- Country-anchored (native terms, single parliament) ---
    {"id": "fr_yellow_vests", "question": "How were the Yellow Vests protests discussed in the French parliament?",
     "topic_domain": "Government Operations", "countries": ["FR"],
     "patterns": [r"gilets jaunes"], "date_from": "2018-11-01", "date_to": "2019-12-31"},
    {"id": "gb_scottish_independence", "question": "How was Scottish independence debated in the UK Parliament?",
     "topic_domain": "Government Operations", "countries": ["GB"],
     "patterns": [r"scottish independence|independence referendum|scotland.{0,25}independ|indyref"]},
    {"id": "es_catalan_independence", "question": "How was Catalan independence debated in Spain?",
     "topic_domain": "Government Operations", "countries": ["ES", "ES-CT"],
     "patterns": [r"independencia.{0,12}catal|cataluñ.{0,18}independ|catalan.{0,18}independ|"
                  r"procés|referéndum.{0,12}catal|1-o\b|puigdemont"]},
    {"id": "gb_nhs", "question": "What did MPs say about the National Health Service (NHS)?",
     "topic_domain": "Health", "countries": ["GB"],
     "patterns": [r"\bnhs\b|national health service"]},
    {"id": "it_reddito_cittadinanza", "question": "How was the citizens' income debated in the Italian parliament?",
     "topic_domain": "Social Welfare", "countries": ["IT"],
     "patterns": [r"reddito di cittadinanza"]},
    {"id": "pl_abortion_protest", "question": "How was the tightening of abortion law debated in Poland?",
     "topic_domain": "Civil Rights", "countries": ["PL"],
     "patterns": [r"aborcj|strajk kobiet"]},

    # --- Final batch to reach ~100 (non-capturing groups; broad multilingual stems) ---
    {"id": "eurozone_membership", "question": "What was debated about adopting the euro and eurozone membership?",
     "topic_domain": "Macroeconomics", "countries": None,
     "patterns": [r"eurozone|euro area|euro-?zone|eurozona|evroobmočje|єврозон|еврозон|"
                  r"euroövezet|euroalue|euron käyttöönotto|adopt.{0,8}euro|introduc.{0,8}euro"]},
    {"id": "fishing_policy", "question": "How were fisheries and fishing policy discussed?",
     "topic_domain": "Environment", "countries": None,
     "patterns": [r"fisher|\bfishing|fischerei|\bpêche|\bpesca|rybołów|rybactw|fiskve|sjavarutveg|"
                  r"ribištvo|ribarstvo|рибальств|рыболовств|halászat|balıkçılık|kalastus|\bfiske"]},
    {"id": "tourism", "question": "What was said about the tourism sector?",
     "topic_domain": "Domestic Commerce", "countries": None,
     "patterns": [r"tourism|tourismus|tourisme|turismo|turizem|turizam|туризм|turizmus|turizm|"
                  r"matkailu|\bturism"]},
    {"id": "small_business_sme", "question": "How were small and medium-sized enterprises supported?",
     "topic_domain": "Domestic Commerce", "countries": None,
     "patterns": [r"small.{0,4}business|\bsme\b|small and medium|mittelstand|"
                  r"kleine.{0,14}unternehmen|petites.{0,12}entreprises|pequeñas.{0,12}empresas|"
                  r"mala.{0,4}podjetja|mali.{0,8}poduzetni|малий бізнес|малого бизнеса|\bkkv\b|"
                  r"pk-yrit|små.{0,8}företag"]},
    {"id": "flood_disasters", "question": "What was debated about floods and natural disasters?",
     "topic_domain": "Environment", "countries": None,
     "patterns": [r"\bflood|hochwasser|überschwemmung|inondation|inundación|\bpoplav|\bповін|"
                  r"наводнен|árvíz|\bsel\b|tulva|översvämning|natural disaster|naturkatastrophe"]},
    {"id": "animal_welfare", "question": "What positions were taken on animal welfare and protection?",
     "topic_domain": "Agriculture", "countries": None,
     "patterns": [r"animal welfare|animal protection|tierschutz|bien-être animal|bienestar animal|"
                  r"dobrobit živali|dobrobit životinja|захист тварин|защит.{0,8}животн|"
                  r"állatvédelem|hayvan refah|djurskydd|eläinten hyvinvoint"]},
    {"id": "childcare", "question": "How was childcare and early-years provision discussed?",
     "topic_domain": "Social Welfare", "countries": None,
     "patterns": [r"childcare|child care|kinderbetreuung|\bkita\b|garde d'enfant|guardería|"
                  r"\bvrtec|\bvrtić|дитяч.{0,8}садоч|детск.{0,6}сад|\bóvoda|päivähoito|barnomsorg"]},
    {"id": "food_security", "question": "What was said about food security and food prices?",
     "topic_domain": "Agriculture", "countries": None,
     "patterns": [r"food (?:price|security|supply)|lebensmittel(?:preis|versorgung)|"
                  r"ernährungssicher|sécurité alimentaire|seguridad alimentaria|"
                  r"prehrambena sigurnost|sigurnost hrane|продовольч.{0,8}безпек|"
                  r"élelmiszer(?:ár|biztonság)|ruokaturva|livsmedel"]},
    {"id": "aging_population", "question": "How was the ageing of the population and its consequences discussed?",
     "topic_domain": "Social Welfare", "countries": None,
     "patterns": [r"age?ing population|überalterung|alterung der|vieillissement|envejecimiento|"
                  r"staranje prebivalstva|starenje stanovništva|старіння насел|"
                  r"старение насел|elöregedés|väestön ikäänty|åldrande befolkning|demografisk"]},
    {"id": "public_sector_wages", "question": "What was debated about public-sector pay?",
     "topic_domain": "Labor", "countries": None,
     "patterns": [r"public sector (?:pay|wage|salar)|öffentlich(?:er|en) dienst|fonction publique|"
                  r"funcionario|javni sektor.{0,14}plač|državn.{0,8}služb.{0,10}plač|"
                  r"державн.{0,10}службовц|közalkalmazott|kamu çalışan|julkisen sektorin palk"]},
]
