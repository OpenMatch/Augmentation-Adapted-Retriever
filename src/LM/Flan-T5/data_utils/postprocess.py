OPTION_POST_FN = {
    ("imdb", "Negation template for positive and negative"): lambda x: ["negative review.", "positive review."],
    ("imdb_pseudo", "Negation template for positive and negative"): lambda x: ["negative review.", "positive review."],
    ("wiqa", "which_of_the_following_is_the_supposed_perturbation"): lambda x: ['directly impacting a step of the process', 'indirectly impacting a step of the process', 'not impacting any step of the process']
}

ANSWER_POST_FN = {
    ("quoref", "Squad"): lambda labels, preds: ([[l] for l in labels], preds),
    ("quoref_pseudo", "Squad"): lambda labels, preds: ([[l] for l in labels], preds),
    ("ropes", "Squad"): lambda labels, preds: ([[l] for l in labels], preds),
    ("ropes_pseudo", "Squad"): lambda labels, preds: ([[l] for l in labels], preds),
}