import re


def extract_features(description):

    text = description.lower()

    features = {
        "gender": None,
        "face": None,
        "eyes": None,
        "nose": None,
        "eyebrows": None,
        "lips": None,
        "hair": None,
        "skin_tone": None,
        "beard": None,
        "moustache": None,
        "age_group": None
    }

    # ---------------------------------------------------------
    # GENDER
    # ---------------------------------------------------------

    if re.search(r"\b(male|man|boy|gentleman)\b", text):
        features["gender"] = "male"

    elif re.search(r"\b(female|woman|girl|lady)\b", text):
        features["gender"] = "female"


    # ---------------------------------------------------------
    # AGE
    # ---------------------------------------------------------

    if re.search(r"\b(child|kid|boy|girl)\b", text):
        features["age_group"] = "child"

    elif re.search(r"\b(teen|teenager)\b", text):
        features["age_group"] = "teen"

    elif re.search(r"\b(young adult|young)\b", text):
        features["age_group"] = "young adult"

    elif re.search(r"\b(middle aged|middle-age|middle age)\b", text):
        features["age_group"] = "middle aged"

    elif re.search(r"\b(old|elderly|senior)\b", text):
        features["age_group"] = "elderly"


    # ---------------------------------------------------------
    # FACE SHAPE
    # ---------------------------------------------------------

    if re.search(r"\b(oval)\b", text):
        features["face"] = "oval"

    elif re.search(r"\b(round)\b", text):
        features["face"] = "round"

    elif re.search(r"\b(square)\b", text):
        features["face"] = "square"

    elif re.search(r"\b(heart)\b", text):
        features["face"] = "heart"

    elif re.search(r"\b(oblong|rectangle|rectangular)\b", text):
        features["face"] = "oblong"

    elif re.search(r"\b(long face|long)\b", text):
        features["face"] = "long"

    elif re.search(r"\b(triangle|triangular|pear)\b", text):
        features["face"] = "triangle"

    elif re.search(r"\b(short face|short)\b", text):
        features["face"] = "short"


    # ---------------------------------------------------------
    # EYES
    # ---------------------------------------------------------

    if re.search(r"\b(almond eyes|almond eye)\b", text):
        features["eyes"] = "almond eyes"

    elif re.search(r"\b(narrow eyes|narrow eye|small narrow eyes)\b", text):
        features["eyes"] = "narrow eyes"

    elif re.search(r"\b(large eyes|large eye|big eyes|big eye)\b", text):
        features["eyes"] = "large eyes"

    elif re.search(r"\b(round eyes|round eye)\b", text):
        features["eyes"] = "round eyes"

    elif re.search(r"\b(small eyes|small eye)\b", text):
        features["eyes"] = "small eyes"

    elif re.search(r"\b(hooded eyes|hooded eye)\b", text):
        features["eyes"] = "hooded eyes"


    # ---------------------------------------------------------
    # EYEBROWS
    # ---------------------------------------------------------

    if re.search(r"\b(high arch|high arched)\b", text):
        features["eyebrows"] = "high arch"

    elif re.search(r"\b(hard angled|hard angle)\b", text):
        features["eyebrows"] = "hard angled"

    elif re.search(r"\b(soft angled|soft angle)\b", text):
        features["eyebrows"] = "soft angled eyebrows"

    elif re.search(r"\b(straight eyebrows|straight eyebrow)\b", text):
        features["eyebrows"] = "straight eyebrows"

    elif re.search(r"\b(thin eyebrows|thin eyebrow)\b", text):
        features["eyebrows"] = "thin eyebrows"

    elif re.search(r"\b(round eyebrows|round eyebrow)\b", text):
        features["eyebrows"] = "round eyebrows"

    elif re.search(r"\b(short eyebrows|short eyebrow)\b", text):
        features["eyebrows"] = "short eyebrows"

    elif re.search(r"\b(arched eyebrows|arched eyebrow|arched)\b", text):
        features["eyebrows"] = "arched eyebrows"

    elif re.search(r"\b(common eyebrows|common eyebrow)\b", text):
        features["eyebrows"] = "common eyebrows"


    # ---------------------------------------------------------
    # NOSE
    # ---------------------------------------------------------

    if re.search(r"\b(long straight nose)\b", text):
        features["nose"] = "long straight nose"

    elif re.search(r"\b(long nose)\b", text):
        features["nose"] = "long nose"

    elif re.search(r"\b(straight nose)\b", text):
        features["nose"] = "straight nose"

    elif re.search(r"\b(narrow nose)\b", text):
        features["nose"] = "narrow nose"

    elif re.search(r"\b(small nose)\b", text):
        features["nose"] = "small nose"

    elif re.search(r"\b(broad nose|wide nose)\b", text):
        features["nose"] = "broad nose"

    elif re.search(r"\b(rounded nose|round nose)\b", text):
        features["nose"] = "rounded nose"


    # ---------------------------------------------------------
    # LIPS
    # ---------------------------------------------------------

    if re.search(r"\b(thin lips|thin lip)\b", text):
        features["lips"] = "thin lips"

    elif re.search(r"\b(full lips|full lip|thick lips)\b", text):
        features["lips"] = "full lips"

    elif re.search(r"\b(medium lips|medium lip|normal lips)\b", text):
        features["lips"] = "medium lips"


    # ---------------------------------------------------------
    # HAIR
    # ---------------------------------------------------------

    if re.search(r"\b(short black hair)\b", text):
        features["hair"] = "short black hair"

    elif re.search(r"\b(long black hair)\b", text):
        features["hair"] = "long black hair"

    elif re.search(r"\b(short hair)\b", text):
        features["hair"] = "short hair"

    elif re.search(r"\b(long hair)\b", text):
        features["hair"] = "long hair"

    elif re.search(r"\b(curly hair)\b", text):
        features["hair"] = "curly hair"

    elif re.search(r"\b(wavy hair)\b", text):
        features["hair"] = "wavy hair"

    elif re.search(r"\b(straight hair)\b", text):
        features["hair"] = "straight hair"


    # ---------------------------------------------------------
    # SKIN TONE
    # ---------------------------------------------------------

    if re.search(r"\b(fair skin|fair complexion)\b", text):
        features["skin_tone"] = "fair skin"

    elif re.search(r"\b(light brown skin)\b", text):
        features["skin_tone"] = "light brown skin"

    elif re.search(r"\b(medium brown skin|medium brown)\b", text):
        features["skin_tone"] = "medium brown skin"

    elif re.search(r"\b(dark brown skin|dark brown)\b", text):
        features["skin_tone"] = "dark brown skin"

    elif re.search(r"\b(dark skin)\b", text):
        features["skin_tone"] = "dark skin"


    # ---------------------------------------------------------
    # BEARD
    # ---------------------------------------------------------

    if re.search(r"\b(light stubble|stubble)\b", text):
        features["beard"] = "light stubble"

    elif re.search(r"\b(short beard)\b", text):
        features["beard"] = "short beard"

    elif re.search(r"\b(goatee)\b", text):
        features["beard"] = "goatee"

    elif re.search(r"\b(chin strap)\b", text):
        features["beard"] = "chin strap"

    elif re.search(r"\b(soul patch)\b", text):
        features["beard"] = "soul patch"

    elif re.search(r"\b(circular beard)\b", text):
        features["beard"] = "circular beard"

    elif re.search(r"\b(van dyke|ven dyke)\b", text):
        features["beard"] = "van dyke"


    # ---------------------------------------------------------
    # MOUSTACHE
    # ---------------------------------------------------------

    if re.search(r"\b(thick moustache|thick mustache)\b", text):
        features["moustache"] = "moustache"

    elif re.search(r"\b(thin moustache|thin mustache)\b", text):
        features["moustache"] = "thin moustache"

    elif re.search(r"\b(pencil moustache|pencil mustache)\b", text):
        features["moustache"] = "pencil moustache"

    elif re.search(r"\b(moustache|mustache)\b", text):
        features["moustache"] = "moustache"


    # ---------------------------------------------------------
    # RETURN
    # ---------------------------------------------------------

    return features