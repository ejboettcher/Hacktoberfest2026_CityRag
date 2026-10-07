"""Civic synonym map for query expansion.

Council minutes use formal terms; residents ask in plain language.
BM25 is exact-term matching, so expanding the query with these
synonyms recovers chunks the raw question would miss.

Add terms as you spot them in test questions. Keep the formal
term first — it appears in the minutes.
"""

SYNONYMS = {
    "pothole": ["street repair", "road repair", "pavement", "resurfacing"],
    "trash": ["waste collection", "garbage", "refuse", "rubbish", "dumping"],
    "sidewalk": ["footpath", "walkway", "pedestrian path"],
    "speeding": ["traffic enforcement", "speed limit", "traffic calming"],
    "park": ["parks", "recreation", "green space", "playground"],
    "pickleball": ["pickleball court", "recreation center"],
    "budget": ["appropriation", "spending", "fund", "tax levy", "mill levy"],
    "tax": ["levy", "property tax", "income tax", "tax rate"],
    "vote": ["ordinance", "resolution", "approved", "passed", "motion"],
    "street light": ["street lighting", "streetlight", "lighting program"],
    "water": ["water line", "water main", "utility"],
    "sewer": ["sanitary sewer", "storm sewer", "drainage"],
    "police": ["public safety", "law enforcement", "dispatch"],
    "fire": ["fire department", "EMS", "emergency services"],
    "zoning": ["rezoning", "land use", "planned development", "variance"],
    "minutes": ["meeting record", "council record", "journal"],
    "clerk": ["city clerk", "clerk of council"],
}


def expand(query, synonyms=SYNONYMS):
    """Return the query plus synonyms for any known civic term found in it."""
    q = query.lower()
    extra = [syn for term, syns in synonyms.items() if term in q for syn in syns]
    return " ".join([query, *extra])
