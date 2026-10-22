"""Farmer-friendly text in English / Gujarati / Hindi.
Guidance is deliberately cautious: cultural practices + 'ask an expert'.
No pesticide names or doses are prescribed."""

LANGS = {"en": "English", "gu": "ગુજરાતી", "hi": "हिन्दी"}

UI = {
    "title": {"en": "LeafGuard – Edge AI Leaf Disease & Pest Forecaster",
              "gu": "લીફગાર્ડ – એજ AI પાન રોગ અને જીવાત આગાહી",
              "hi": "लीफगार्ड – एज AI पत्ती रोग एवं कीट पूर्वानुमान"},
    "tab_leaf": {"en": "1. Leaf check (offline)", "gu": "૧. પાનની તપાસ (ઑફલાઇન)", "hi": "१. पत्ती जाँच (ऑफ़लाइन)"},
    "tab_pest": {"en": "2. Local pest risk", "gu": "૨. સ્થાનિક જીવાત જોખમ", "hi": "२. स्थानीय कीट जोखिम"},
    "upload": {"en": "Upload a leaf photo", "gu": "પાનનો ફોટો અપલોડ કરો", "hi": "पत्ती की फोटो अपलोड करें"},
    "camera": {"en": "Or take a photo with the camera", "gu": "અથવા કેમેરાથી ફોટો લો", "hi": "या कैमरे से फोटो लें"},
    "likely": {"en": "Likely condition", "gu": "સંભવિત સ્થિતિ", "hi": "संभावित स्थिति"},
    "confidence": {"en": "Confidence", "gu": "વિશ્વાસ સ્તર", "hi": "विश्वास स्तर"},
    "uncertain_title": {"en": "UNCERTAIN – the model is not sure", "gu": "અનિશ્ચિત – મોડેલને ખાતરી નથી", "hi": "अनिश्चित – मॉडल को पक्का नहीं पता"},
    "uncertain_msg": {
        "en": "Retake a clear, close photo of ONE leaf in daylight, or consult an agriculture expert.",
        "gu": "દિવસના પ્રકાશમાં એક જ પાનનો નજીકથી સ્પષ્ટ ફોટો ફરીથી લો, અથવા કૃષિ નિષ્ણાતની સલાહ લો.",
        "hi": "दिन की रोशनी में एक ही पत्ती की पास से साफ फोटो दोबारा लें, या कृषि विशेषज्ञ से सलाह लें."},
    "other_options": {"en": "Other possibilities", "gu": "અન્ય શક્યતાઓ", "hi": "अन्य संभावनाएँ"},
    "why": {"en": "Why this result? Highlighted areas influenced the model most",
            "gu": "આ પરિણામ કેમ? હાઇલાઇટ કરેલા ભાગોએ મોડેલને સૌથી વધુ અસર કરી",
            "hi": "यह परिणाम क्यों? हाइलाइट किए गए हिस्सों ने मॉडल को सबसे ज्यादा प्रभावित किया"},
    "next_steps": {"en": "Suggested next steps", "gu": "સૂચવેલા આગળના પગલાં", "hi": "सुझाए गए अगले कदम"},
    "expert": {
        "en": "Always confirm with your local Krishi Vigyan Kendra (KVK) or agriculture officer before spraying any chemical. Kisan Call Centre: 1800-180-1551 (toll free).",
        "gu": "કોઈપણ દવા છાંટતા પહેલા તમારા સ્થાનિક કૃષિ વિજ્ઞાન કેન્દ્ર (KVK) અથવા ખેતીવાડી અધિકારી પાસે ખાતરી કરો. કિસાન કોલ સેન્ટર: 1800-180-1551 (ટોલ ફ્રી).",
        "hi": "कोई भी दवा छिड़कने से पहले अपने स्थानीय कृषि विज्ञान केंद्र (KVK) या कृषि अधिकारी से पुष्टि करें। किसान कॉल सेंटर: 1800-180-1551 (टोल फ्री)."},
    "district": {"en": "District / location", "gu": "જિલ્લો / સ્થળ", "hi": "जिला / स्थान"},
    "crop": {"en": "Crop", "gu": "પાક", "hi": "फसल"},
    "mode": {"en": "Weather data source", "gu": "હવામાન ડેટા સ્ત્રોત", "hi": "मौसम डेटा स्रोत"},
    "mode_live": {"en": "Live weather (needs internet)", "gu": "લાઇવ હવામાન (ઇન્ટરનેટ જરૂરી)", "hi": "लाइव मौसम (इंटरनेट जरूरी)"},
    "mode_manual": {"en": "Offline – enter values", "gu": "ઑફલાઇન – મૂલ્યો દાખલ કરો", "hi": "ऑफ़लाइन – मान दर्ज करें"},
    "forecast_btn": {"en": "Forecast pest risk", "gu": "જીવાત જોખમની આગાહી કરો", "hi": "कीट जोखिम का पूर्वानुमान लगाएँ"},
    "risk_level": {"en": "Pest-risk level (next 3 days)", "gu": "જીવાત જોખમ સ્તર (આગામી ૩ દિવસ)", "hi": "कीट जोखिम स्तर (अगले 3 दिन)"},
    "factors": {"en": "Main factors behind this estimate", "gu": "આ અંદાજ પાછળના મુખ્ય કારણો", "hi": "इस अनुमान के मुख्य कारण"},
    "provenance": {"en": "Data used", "gu": "વપરાયેલ ડેટા", "hi": "उपयोग किया गया डेटा"},
    "conservative": {
        "en": "Confidence is low, so the level was raised to stay on the safe side.",
        "gu": "વિશ્વાસ સ્તર ઓછું હોવાથી સાવચેતી માટે સ્તર વધારવામાં આવ્યું છે.",
        "hi": "विश्वास स्तर कम होने के कारण सावधानी के लिए स्तर बढ़ाया गया है."},
    "no_model": {"en": "Leaf model not found. Run train_disease.py first (see README).",
                 "gu": "પાન મોડેલ મળ્યું નથી. પહેલા train_disease.py ચલાવો (README જુઓ).",
                 "hi": "पत्ती मॉडल नहीं मिला। पहले train_disease.py चलाएँ (README देखें)."},
}

LEVELS = {
    "Low": {"en": "LOW", "gu": "ઓછું", "hi": "कम"},
    "Moderate": {"en": "MODERATE", "gu": "મધ્યમ", "hi": "मध्यम"},
    "High": {"en": "HIGH", "gu": "વધારે", "hi": "अधिक"},
}

PEST_ADVICE = {
    "Low": {"en": "Routine scouting once a week.",
            "gu": "અઠવાડિયામાં એક વાર નિયમિત ખેતર નિરીક્ષણ કરો.",
            "hi": "सप्ताह में एक बार नियमित खेत निरीक्षण करें."},
    "Moderate": {"en": "Scout every 2–3 days, check the underside of leaves, remove weeds and infected leaves.",
                 "gu": "દર ૨–૩ દિવસે ખેતરનું નિરીક્ષણ કરો, પાનની નીચેની બાજુ તપાસો, નીંદણ અને રોગગ્રસ્ત પાન દૂર કરો.",
                 "hi": "हर 2–3 दिन में निरीक्षण करें, पत्तियों की निचली सतह देखें, खरपतवार और रोगग्रस्त पत्ते हटाएँ."},
    "High": {"en": "Scout daily, avoid over-irrigation, and contact KVK / agriculture officer early for confirmed advice.",
             "gu": "રોજ નિરીક્ષણ કરો, વધુ પિયત ટાળો અને ખાતરીબદ્ધ સલાહ માટે વહેલી તકે KVK / ખેતીવાડી અધિકારીનો સંપર્ક કરો.",
             "hi": "रोज निरीक्षण करें, अधिक सिंचाई से बचें और पुष्ट सलाह के लिए जल्दी KVK / कृषि अधिकारी से संपर्क करें."},
}

DISEASE_ADVICE = {
    "healthy": {
        "en": ["No clear disease sign found. Keep monitoring weekly.", "Maintain balanced irrigation and nutrition."],
        "gu": ["રોગનું કોઈ સ્પષ્ટ ચિહ્ન મળ્યું નથી. દર અઠવાડિયે નિરીક્ષણ ચાલુ રાખો.", "સંતુલિત પિયત અને પોષણ જાળવો."],
        "hi": ["रोग का कोई स्पष्ट लक्षण नहीं मिला। हर सप्ताह निगरानी जारी रखें।", "संतुलित सिंचाई और पोषण बनाए रखें।"]},
    "fungal": {
        "en": ["Remove badly affected leaves and dispose of them safely (do not compost).",
               "Avoid overhead watering, keep leaves dry and improve air flow.",
               "Do not spray any fungicide without expert confirmation of the disease and dose."],
        "gu": ["વધુ અસરગ્રસ્ત પાન કાઢીને સુરક્ષિત રીતે નાશ કરો.",
               "ઉપરથી પિયત ટાળો, પાન સૂકા રાખો અને હવાની અવરજવર સુધારો.",
               "રોગ અને માત્રાની નિષ્ણાત પાસે ખાતરી કર્યા વિના કોઈ ફૂગનાશક છાંટશો નહીં."],
        "hi": ["ज्यादा प्रभावित पत्तियाँ तोड़कर सुरक्षित तरीके से नष्ट करें।",
               "ऊपर से सिंचाई न करें, पत्ते सूखे रखें और हवा का आवागमन बढ़ाएँ।",
               "रोग और मात्रा की विशेषज्ञ से पुष्टि किए बिना कोई फफूंदनाशक न छिड़कें।"]},
    "bacterial": {
        "en": ["Remove infected leaves and avoid working in the field when plants are wet.",
               "Clean tools between plants.",
               "Do not apply any chemical without expert advice."],
        "gu": ["રોગગ્રસ્ત પાન દૂર કરો અને છોડ ભીના હોય ત્યારે ખેતરમાં કામ ન કરો.",
               "ઓજારો દરેક છોડ વચ્ચે સાફ કરો.",
               "નિષ્ણાતની સલાહ વિના કોઈ દવા છાંટશો નહીં."],
        "hi": ["रोगग्रस्त पत्तियाँ हटाएँ और पौधे गीले होने पर खेत में काम न करें।",
               "औजारों को हर पौधे के बाद साफ करें।",
               "विशेषज्ञ की सलाह के बिना कोई दवा न छिड़कें।"]},
    "viral": {
        "en": ["Viral diseases are often spread by insects such as whitefly; remove and destroy badly infected plants.",
               "Keep weeds under control.",
               "Ask an expert about insect management."],
        "gu": ["વાયરસજન્ય રોગ સફેદમાખી જેવી જીવાતથી ફેલાય છે; વધુ અસરગ્રસ્ત છોડ દૂર કરીને નાશ કરો.",
               "નીંદણ નિયંત્રણમાં રાખો.",
               "જીવાત નિયંત્રણ માટે નિષ્ણાતની સલાહ લો."],
        "hi": ["विषाणु रोग अक्सर सफेद मक्खी जैसे कीटों से फैलते हैं; ज्यादा प्रभावित पौधों को हटाकर नष्ट करें।",
               "खरपतवार नियंत्रित रखें।",
               "कीट प्रबंधन के लिए विशेषज्ञ से सलाह लें।"]},
    "mite": {
        "en": ["Check the underside of leaves for tiny mites or fine webbing.", "Avoid dust and water stress.",
               "Ask an expert before using any miticide."],
        "gu": ["પાનની નીચેની બાજુ નાની કથીરી અથવા ઝીણા જાળા તપાસો.", "ધૂળ અને પાણીની તાણ ટાળો.",
               "કોઈ કથીરીનાશક વાપરતા પહેલા નિષ્ણાતની સલાહ લો."],
        "hi": ["पत्तियों की निचली सतह पर छोटे घुन या महीन जाले देखें।", "धूल और पानी की कमी से बचें।",
               "कोई माइटनाशक उपयोग करने से पहले विशेषज्ञ से सलाह लें।"]},
}


def disease_category(class_name: str) -> str:
    n = class_name.lower()
    if "healthy" in n:
        return "healthy"
    if any(k in n for k in ("virus", "curl", "mosaic")):
        return "viral"
    if "mite" in n:
        return "mite"
    if "bacterial" in n or "haunglongbing" in n:
        return "bacterial"
    return "fungal"


def pretty_class(class_name: str) -> str:
    if "___" in class_name:
        crop, dis = class_name.split("___", 1)
        crop = crop.replace("_", " ").replace(",", "")
        return f"{crop} — {dis.replace('_', ' ')}"
    return class_name.replace("_", " ")
