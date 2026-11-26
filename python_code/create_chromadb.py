"""
About
This code is to create a drug vector db using MoA, pharmacodynamics, and indication
"""

import os

## This is for the issue with chromadb
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import sys

## For server
# sys.path.insert(0, os.path.abspath(os.path.join("utils")))
# sys.path.insert(0, os.getcwd())
# from utils.utils import remove_pattern, KO_pattern, bracket_pattern
import yaml
import torch
import numpy as np
import pandas as pd
import chromadb
from chromadb.config import Settings
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer, AutoModel
from chromadb.utils.batch_utils import create_batches
from chromadb.api.types import Documents, Embeddings, EmbeddingFunction
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,  # <-- Force logs to stdout
    force=True,  # <-- Overwrite any previous logging config
)
logger = logging.getLogger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device = "cpu" if not torch.cuda.is_available() else torch.device("cuda:0")

CONFIG_FOLDER = os.path.join("configuration", "vector")
VECTORDB_CONFIG_PATH = os.path.join(
    CONFIG_FOLDER, "vectordb_BioLinkBERT_small_mol.yaml"
)

with open(VECTORDB_CONFIG_PATH, "r") as f:
    vectorDB = yaml.safe_load(f)

MODEL_WT = vectorDB["MODEL_WT"]
MODEL_DATABASE = vectorDB["MODEL_DATABASE"]
MODEL_NAME = vectorDB["MODEL_NAME"]
logger.info(f"Using model: {MODEL_NAME}")
logger.info(f"Using Database from: {MODEL_DATABASE}")

# Initialize BioBERT
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=MODEL_NAME, cache_dir=MODEL_WT
)
model = AutoModel.from_pretrained(
    pretrained_model_name_or_path=MODEL_NAME,
    cache_dir=MODEL_WT,
    # device_map="cuda:0",
    # torch_dtype=torch.bfloat16,
).to(device)

character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=vectorDB["chunk_size"],
    chunk_overlap=vectorDB["chunk_overlap"],
)

## Drug database
drug_db = pd.read_csv(vectorDB["DRUG_PATH"])
disease_db = pd.read_csv(vectorDB["DISEASE_PATH"])


def text_split(col_to_use: str, db: pd.DataFrame, uniq_id_col: str = "drugbank-id"):
    ## make a list of text that is split
    splitted_text_list = []
    DB_id_matching_splitted_text_list = []
    ## drop the row with nan for the columns [col_to_use, uniq_id_col]
    db = db.dropna(subset=[col_to_use, uniq_id_col])[[col_to_use, uniq_id_col]]
    for id, row in db.iterrows():
        text_ = row[col_to_use]
        if not isinstance(text_, str):
            continue  # Skip if the text is not a string
        character_split_texts = [
            t
            for t in character_splitter.split_text(text_)
            if len(t) > vectorDB["chunk_overlap"] + 5
        ]

        DB_id_matching_splitted_text_list.extend(
            [
                {uniq_id_col: i, "feature": col_to_use}
                for i in [row[uniq_id_col]] * len(character_split_texts)
            ]
        )
        splitted_text_list.extend(character_split_texts)
        # if id == 5:
        #    break
    generate_ids = [str(i) for i in range(len(DB_id_matching_splitted_text_list))]

    return splitted_text_list, DB_id_matching_splitted_text_list, generate_ids


def get_biobert_embedding(text):
    """Generate BioBERT embeddings with GPU optimization"""
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)[0].cpu().float().numpy()


class BioBERTEmbeddingFunction(EmbeddingFunction[Documents]):
    def __call__(self, input: Documents) -> Embeddings:
        # input is a list of strings (documents)
        embeddings = [get_biobert_embedding(text) for text in input]
        return normalize(np.array(embeddings)).tolist()

    def name(self):
        return "biobert"


def main(
    df: pd.DataFrame,
    collection_name_list: list[str] = None,
    uniq_id_col: str = "drugbank-id",
):

    ## Clean the database
    # drug_db["GENE"] = drug_db["GENE"].apply(
    #    lambda x: remove_pattern(x, KO_pattern, replace_with=" ")
    # )

    ## Initialize the embedding function
    embedding_function = BioBERTEmbeddingFunction()

    ## Iterate through the collection names
    for collection_name in collection_name_list:
        print(f"Processing collection: {collection_name}")

        chroma_client = chromadb.PersistentClient(path=MODEL_DATABASE)
        collection = chroma_client.get_or_create_collection(
            name=collection_name, embedding_function=embedding_function
        )

        ## Get the splitted text and metadata
        splitted_text_list, DB_id_matching_splitted_text_list, generate_ids = (
            text_split(col_to_use=collection_name, db=df, uniq_id_col=uniq_id_col)
        )
        print(
            f"Number of splitted texts for {collection_name}: {len(splitted_text_list)}"
        )
        batches = create_batches(
            api=chroma_client,
            ids=generate_ids,
            documents=splitted_text_list,
            metadatas=DB_id_matching_splitted_text_list,
        )

        for batch in batches:
            print(f"Adding a batch of size {len(batch[0])}")
            collection.add(ids=batch[0], documents=batch[3], metadatas=batch[2])

    print("All collections processed successfully.")


if __name__ == "__main__":
    main(df=drug_db, collection_name_list=vectorDB["collection_name_drug"])
    main(
        df=disease_db,
        collection_name_list=vectorDB["collection_name_disease"],
        uniq_id_col="Database_ID",
    )
