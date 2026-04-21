from app.commerce.grounding import build_grounding_candidates


def test_build_grounding_candidates_scores_dom_candidates_for_current_goal():
    dom_summary = """
    [1]<input>Search Amazon</input>
    [2]<a>Apple iPhone 16 128GB unlocked product card</a>
    [3]<button>$729.99 Buy now</button>
    [4]<button>Customer reviews</button>
    """

    candidates = build_grounding_candidates(
        dom_summary,
        "Collect current iPhone 16 price and seller details.",
    )

    assert candidates
    assert candidates[0].source == "dom"
    assert any(item.kind == "price_area" for item in candidates)
    assert any(item.selector_hint == "interactive_index=3" for item in candidates)


def test_build_grounding_candidates_uses_layout_fallback_without_dom():
    candidates = build_grounding_candidates(
        "",
        "Find reviews and recurring complaints for this product.",
    )

    assert candidates
    assert candidates[0].source == "layout"
    assert any(item.kind == "review_anchor" for item in candidates)
