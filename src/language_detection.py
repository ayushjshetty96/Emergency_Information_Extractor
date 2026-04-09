from langdetect import detect


def detect_language(text):

    hindi_words = [
        "hai", "hua", "par", "ke", "ka", "ki",
        "bada", "chota", "pass", "yaha", "waha"
    ]

    # Common Kannada words written in English (transliteration)
    kannada_words = [
        "aagide", "ide", "iddare", "nalli", "hatra", "hattira",
        "dodda", "chikka"
    ]

    words = text.lower().split()

    if any(w in words for w in hindi_words):
        return "Hindi"

    if any(w in words for w in kannada_words):
        return "Kannada"

    try:
        lang = detect(text)
    except Exception:
        return "Unknown"

    if lang == "en":
        return "English"
    elif lang == "hi":
        return "Hindi"
    elif lang == "kn":
        return "Kannada"
    else:
        return "English"