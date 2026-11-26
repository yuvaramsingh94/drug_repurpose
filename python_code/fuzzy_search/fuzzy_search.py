"""
About
Here are the codes for performing fuzzy search
"""

from rapidfuzz import fuzz, process


def match_finder(query_word: str, list_of_words: list[str]) -> list[tuple]:
    """Find the best candidates for the match using fuzzy search.
    If there is an exact match, then we will return that match.
    If not then we will return the best matches that have partial match.

    Args:
        query_word (Str): The user query word to be matched.
        list_of_words (List[Str]): The list of words to be matched against.

    Returns:
        List[Tuple]: A list of tuples containing the matched word and its score.
    """
    good_matches = process.extract(query_word, list_of_words, scorer=fuzz.ratio)
    for idx, match in enumerate(good_matches):

        ## Consider only the string which has length +- 2 of the query word

        score = fuzz.partial_ratio(match[0], query_word)
        if idx == 0:
            if score == 100:
                ## end the exploration as we have an exact match
                return good_matches[idx]
            top_value = score
        else:
            if score != top_value:
                break
    # sel = good_matches[:idx]
    return good_matches[:idx]
