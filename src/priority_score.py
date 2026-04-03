def calculate_priority(text, severity):

    score = 50
    text = text.lower()

    keywords = ["explosion","collapse","trapped","huge","major"]

    for word in keywords:
        if word in text:
            score += 10

    if severity == "high":
        score += 20

    if severity == "low":
        score -= 10

    return max(0, min(score, 100))