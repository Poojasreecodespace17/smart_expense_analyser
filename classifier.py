"""
classifier.py
─────────────
Two-stage classification:
  Stage 1 — Rule engine  : deterministic keyword rules that override the model
                           for cases where BigBasket's labelling is wrong or
                           biased (e.g. handwash=Beauty 135x, Cleaning 4x).
  Stage 2 — SVM model    : TF-IDF + LinearSVC trained on BigBasket products.
                           Used only when no rule matches.
"""

import re
import os
import pickle

# ── Load model ────────────────────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
_pipeline = pickle.load(open(os.path.join(_DIR, "model.pkl"), "rb"))


# ── Stage 1: Rule engine ──────────────────────────────────────────────────────
# Each rule is (compiled_regex, category).
# Rules are checked IN ORDER — first match wins.
# Add more rules here for any new product type that the model gets wrong.

_RULES = [
    # ── Cleaning & Household ──────────────────────────────────────────────────
    # Must come BEFORE soap/beauty rules because "handwash" contains "wash"
    (re.compile(r'\b(dishwash|dish\s*wash|dishwashing)\b',         re.I), "Cleaning & Household"),
    (re.compile(r'\b(hand\s*wash|handwash)\b',                      re.I), "Cleaning & Household"),
    (re.compile(r'\b(floor\s*clean|toilet\s*clean|surface\s*clean)',re.I), "Cleaning & Household"),
    (re.compile(r'\b(mop|mopping|floor\s*block|fresh\s*block)\b',   re.I), "Cleaning & Household"),
    (re.compile(r'\b(vim|pitambari|harpic|lizol|domex|collin)\b',   re.I), "Cleaning & Household"),
    (re.compile(r'\b(dishwash\s*(bar|liquid|powder|gel))\b',        re.I), "Cleaning & Household"),
    (re.compile(r'\bgodrej\s+protekt\b',                            re.I), "Cleaning & Household"),

    # ── Foodgrains, Oil & Masala ──────────────────────────────────────────────
    # Vermicelli is 23:2 Snacks in BigBasket — override it
    (re.compile(r'\b(vermicelli|sevai|semiya)\b',                   re.I), "Foodgrains, Oil & Masala"),
    (re.compile(r'\b(hing|asafoetida|compounded\s*hing)\b',         re.I), "Foodgrains, Oil & Masala"),
    (re.compile(r'\b(masala|spice|turmeric|jeera|cumin|coriander)\b',re.I),"Foodgrains, Oil & Masala"),
    (re.compile(r'\b(rice|wheat|atta|flour|dal|lentil|pulses?)\b',  re.I), "Foodgrains, Oil & Masala"),
    (re.compile(r'\b(cooking\s*oil|sunflower\s*oil|palm\s*oil)\b',  re.I), "Foodgrains, Oil & Masala"),

    # ── Bakery, Cakes & Dairy ─────────────────────────────────────────────────
    (re.compile(r'\b(condensed\s*milk|milkmaid)\b',                 re.I), "Bakery, Cakes & Dairy"),
    (re.compile(r'\b(milkshake|milk\s*shake)\b',                    re.I), "Bakery, Cakes & Dairy"),
    (re.compile(r'\b(butter|cheese|paneer|curd|yogurt|ghee)\b',     re.I), "Bakery, Cakes & Dairy"),

    # ── Snacks & Branded Foods ────────────────────────────────────────────────
    (re.compile(r'\b(chocolate|choco|cadbury|kitkat|dairy\s*milk)\b',re.I),"Snacks & Branded Foods"),
    (re.compile(r'\b(biscuit|cookie|chips|namkeen|wafer|cracker)\b',re.I), "Snacks & Branded Foods"),

    # ── Beauty & Hygiene (personal care) ─────────────────────────────────────
    # soap/shampoo AFTER handwash/dishwash rules so those aren't caught here
    (re.compile(r'\b(shampoo|conditioner|hair\s*oil|hair\s*color|hair\s*colour)\b', re.I), "Beauty & Hygiene"),
    (re.compile(r'\b(indica|parachute|dove\s*shampoo|head\s*shoulder)\b', re.I), "Beauty & Hygiene"),
    (re.compile(r'\b(toothpaste|toothbrush|mouthwash|colgate|pepsodent)\b', re.I), "Beauty & Hygiene"),
    (re.compile(r'\b(body\s*soap|bathing\s*soap|cinthol|lux\s*soap|lifebuoy)\b', re.I), "Beauty & Hygiene"),
    (re.compile(r'\b(deo|deodorant|perfume|body\s*wash|face\s*wash)\b', re.I), "Beauty & Hygiene"),
]


def classify_item(item_name: str) -> str:
    """
    Classify a grocery product name.

    Stage 1: Check deterministic rules (handles BigBasket label bias).
    Stage 2: Fall back to trained SVM model.
    """
    name = item_name.strip()

    # Stage 1 — rules
    for pattern, category in _RULES:
        if pattern.search(name):
            return category

    # Stage 2 — model
    try:
        return _pipeline.predict([name.lower()])[0]
    except Exception:
        return "Uncategorised"