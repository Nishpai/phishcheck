def top_phish_words(model, text, n=6):
    """Words that pushed this message toward phish.

    contribution = tf-idf of the word in this message x the word's coefficient.
    """
    tfidf = model.named_steps["tfidf"]
    logreg = model.named_steps["logreg"]
    # classes_ sort as ["legit", "phish"], so coef_ already points toward phish.
    weights = logreg.coef_[0]
    words = tfidf.get_feature_names_out()

    row = tfidf.transform([text])
    scores = []
    for i, value in zip(row.indices, row.data):
        contribution = value * weights[i]
        if contribution > 0:
            scores.append((words[i], contribution))

    scores.sort(key=lambda pair: pair[1], reverse=True)
    return [{"word": str(w), "contribution": round(float(c), 3)} for w, c in scores[:n]]
