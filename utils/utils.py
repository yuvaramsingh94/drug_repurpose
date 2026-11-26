import re
import os
import sys

sys.path.append("..")
import pandas as pd

## Pattern
A_pattern = r"\[(?:[A-Z]\d+(?:,\s*)?)+\]"
url_pattern = r"https?://\S+"
html_tag_pattern = r"<[^>]+>.*?</[^>]+>"  # Matches HTML tags
EC_pattern = r"\(EC [0-9.\-]+\)"
ECO_pattern = r"\{[^}]*\}"
step_pattern = r"step \d+/\d+\."
PubMed_pattern = r"\(PubMed:\d+(?:, PubMed:\d+)*\)"
GO_pattern = r"\[GO:\d+\]"
MIM_pattern = r"\[MIM:\d+\]"
TC_pattern = r"\(TC [0-9A-Z.]+\)"
orphan_dot_pattern = r"(?<=\s|\A)\.(?=\s|\Z)"
kegg_pattern = r"\(.*?\)"  # r"\( *([^()]*) *\)|\[.*$|\s+|,"
bracket_pattern = r"\[.*?\]"  # r"\[HSA:\d+\]"
flower_bracket_pattern = r"\{.*?\}"  # r"\[HSA:\d+\]"
KO_pattern = r"\[KO:(?:K\d+\s*)+\]"  # r"\[KO:K\d+\]"
crlf_pattern = r"\r\n"  # Matches Windows line endings (carriage return + line feed)


def remove_pattern(text: str, pattern: str, replace_with: str = "") -> str:
    """
    Remove the given pattern from the text

    Args:
        text (str): Input text
        pattern (str): Pattern
        replace_with (str): Text to replace

    Returns:
        str: Cleaned text
    """
    return re.sub(pattern, replace_with, text)


sys.path.append(os.path.join(".."))
sys.path.append(os.path.join("..", ".."))
from API_key import (
    Gemini_api,
    LANGSMITH_API_KEY,
    LANGSMITH_ENDPOINT,
    LANGSMITH_TRACING,
)


def API_setup():
    """
    A function to set the required Environemnt variable with the API key.
    """
    if not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = Gemini_api
    if not os.environ.get("LANGSMITH_TRACING"):
        os.environ["LANGSMITH_TRACING"] = LANGSMITH_TRACING
    if not os.environ.get("LANGSMITH_ENDPOINT"):
        os.environ["LANGSMITH_ENDPOINT"] = LANGSMITH_ENDPOINT
    if not os.environ.get("LANGSMITH_API_KEY"):
        os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY


def convert_columns_to_lower(
    df: pd.DataFrame, columns_to_convert: list
) -> pd.DataFrame:
    """Converts strings in specified columns of a DataFrame to lowercase.

    Args:
        df (pd.DataFrame): The DataFrame to modify.
        columns_to_convert (list): List of column names to convert.

    Returns:
        pd.DataFrame: The modified DataFrame.
    """
    for column in columns_to_convert:
        if (
            df[column].dtype == "object"
        ):  # Check if the column is of type object (string)
            df[column] = df[column].str.lower()
    return df


def filter_dataframe_by_selected_ids(df: pd.DataFrame, selected_db_ids: dict):
    """Filter the query database using the pair of query and selected Drug Bank id

    Args:
        df (pd.DataFrame): The Query dataframe with drugbank id, statement and query
        selected_db_ids (dict): Dictionary of query and list of drugbank id

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    # Start with a mask of all False values
    final_mask = pd.Series([False] * len(df), index=df.index)

    # Loop through each query-ID pair
    for q, id_list in zip(selected_db_ids["query"], selected_db_ids["selected_DB_ids"]):
        # Create a mask for the current pair
        pair_mask = (df["query"] == q) & (df["drugbank-id"].isin(id_list))

        # Add the rows matching this pair to our final mask using logical OR
        final_mask = final_mask | pair_mask

    # Apply the final combined mask to the DataFrame
    return df[final_mask]
