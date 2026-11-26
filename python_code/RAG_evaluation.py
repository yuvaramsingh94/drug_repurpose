import os

## This is for the issue with chromadb
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import chromadb
from chromadb.api.types import Documents, Embeddings, EmbeddingFunction
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer, AutoModel

import sys

## For server
sys.path.insert(0, os.path.abspath(os.path.join("utils")))
sys.path.insert(0, os.getcwd())
from utils.utils import API_setup
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from state_definition import Query_structured_output
import pandas as pd
import yaml
from system_message import query_sys_msg
import os
import torch
import numpy as np
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0")


API_setup()

CONFIG_FOLDER = os.path.join("configuration")
DATABASE_PATH_FOLDER = os.path.join(CONFIG_FOLDER, "database_path.yaml")
LLM_SETTINGS = os.path.join(CONFIG_FOLDER, "LLM.yaml")
with open(LLM_SETTINGS, "r") as f:
    LLM_settings = yaml.safe_load(f)
with open(DATABASE_PATH_FOLDER, "r") as f:
    DATABASE_PATH = yaml.safe_load(f)

MODEL_WT = os.path.join("weights")
MODEL_DATABASE = os.path.join("database")

COLLECTION_NAME = "drug_mechanism_of_action"
MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

drug_df = pd.read_csv(DATABASE_PATH["DRUG_PATH"])

# Initialize BioBERT only once
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=MODEL_NAME, cache_dir=MODEL_WT
)
model = AutoModel.from_pretrained(
    pretrained_model_name_or_path=MODEL_NAME,
    cache_dir=MODEL_WT,
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
).to(device)


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


def filter_dataframe_by_selected_ids(df, selected_db_ids):
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


def retrieve_and_format_queries(
    queries, MoA_collection, PD_collection, llm, n_results=5
):
    """
    For each query, retrieve relevant docs from both MoA_collection and PD_collection,
    build a dataframe with query, RAG system, ids, documents, metadatas,
    and generate a markdown summary.
    """
    results_list = []
    markdown_lines = []
    selected_db_ids = {"query": [], "selected_DB_ids": []}
    for count, query in enumerate(queries):
        db_id_list = []
        docs_list = []
        for rag_name, collection in [("MoA", MoA_collection), ("PD", PD_collection)]:
            res = collection.query(query_texts=[query], n_results=n_results)
            ids = res.get("ids", [[]])[0]

            docs = res.get("documents", [[]])[0]
            docs_list.extend(docs)
            metadatas = res.get("metadatas", [[]])[0]
            drugbank_ids = [meta.get("drugbank-id", "") for meta in metadatas]
            db_id_list.extend(drugbank_ids)
            for i, (id_, doc, meta) in enumerate(zip(ids, docs, metadatas)):
                results_list.append(
                    {
                        "query": query,
                        "RAG": rag_name,
                        "id": id_,
                        "document": doc,
                        # "metadata": meta,
                        "drugbank-id": drugbank_ids[i] if i < len(drugbank_ids) else "",
                    }
                )
        # Markdown formatting
        markdown_lines.append([])
        # markdown_lines[count] = []
        markdown_lines[count].append(f"## Query: {query}")
        for id_, doc in zip(db_id_list, docs_list):
            markdown_lines[count].append(f"- [{id_}] {doc}")

        # markdown_text_temp = "\n".join(markdown_lines[count])

        # history_msg = query_sys_msg + [HumanMessage(content=markdown_text_temp)]
        # op = llm.invoke(
        #     history_msg,
        # )
        # selected_db_ids["query"].append(query)
        # selected_db_ids["selected_DB_ids"].append(op.selected_DB_ids)
        # print("Hi")
    # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    # flatten markdown_lines and create a single markdown text
    markdown_text_temp = "\n".join(
        [line for sublist in markdown_lines for line in sublist]
    )
    # print(markdown_text_temp)
    history_msg = query_sys_msg + [HumanMessage(content=markdown_text_temp)]
    op = llm.invoke(history_msg)
    for op_item in op.query_list:
        selected_db_ids["query"].append(op_item.query)
        selected_db_ids["selected_DB_ids"].append(op_item.db_ids)

    df = pd.DataFrame(results_list)
    print("before filtering", df.shape)
    df = filter_dataframe_by_selected_ids(df, selected_db_ids)
    print("after filtering", df.shape)
    filtered_df = df[["query", "drugbank-id"]].drop_duplicates(inplace=False)
    drug_bank_filtered = drug_df[
        drug_df["drugbank-id"].isin(filtered_df["drugbank-id"])
    ]
    filtered_df = filtered_df.merge(
        drug_bank_filtered,
        on="drugbank-id",
        how="inner",
    )
    markdown_text = ""
    for query in filtered_df["query"].unique():
        sub_df = filtered_df[filtered_df["query"] == query]
        lines = sub_df.apply(
            lambda row: f'- [{row["drugbank-id"]}] MoA: {row["mechanism-of-action"]} Pharmacodynamics: {row["pharmacodynamics"]} Indication:  {row["indication"]}',
            axis=1,
        )
        markdown_text += f"## Query: {query}\n" + "\n".join(lines) + "\n"
    print(markdown_text)
    return filtered_df, markdown_text  # , markdown_text


def main():
    # queries_list = [
    #     "drug that inhibits thrombin",
    #     "find drugs that binds to insulin receptor",
    #     "drug that separates plasminogen",
    # ]
    queries_list = [
        "Which drugs are known to inhibit BRCA1?",
        "Which drugs are known to inhibit BRCA2?",
        "Which drugs are known to activate BRCA1?",
    ]
    embedding_function = BioBERTEmbeddingFunction()

    chroma_client = chromadb.PersistentClient(path=MODEL_DATABASE)
    MoA_collection = chroma_client.get_collection(
        name="mechanism-of-action", embedding_function=embedding_function
    )

    PD_collection = chroma_client.get_collection(
        name="pharmacodynamics", embedding_function=embedding_function
    )

    llm = ChatGoogleGenerativeAI(
        model=LLM_settings["model"], temperature=LLM_settings["temperature"]
    )
    query_llm = llm.with_structured_output(Query_structured_output)

    retrieve_and_format_queries(
        queries_list, MoA_collection, PD_collection, query_llm, n_results=5
    )


if __name__ == "__main__":
    main()

"""
From the filtered_df, using the query, drugbank-id and document, generate a markdown text that can be used to display the results in a markdown format.
Example 
## Query: drug that inhibits thrombin\n- [DB11095] Desirudin is a direct, highly selective thrombin inhibitor. Reversibly binds to the active thrombin site of free and clot-associated thrombin\n- [DB00569] . Neutralization of Factor Xa interrupts the blood coagulation cascade and thus inhibits thrombin formation and thrombus development\n- [DB06005] flexible administration (intravenous and subcutaneous), and both potent and predictable anti-Xa (aXa) and anti-IIa (aIIa) activity to provide anticoagulant therapy to patients with ACS\n- [DB05777] .  Thrombomodulin alfa attenuates the extension of the clot by inhibiting further thrombin generation on clots, while other anticoagulants inhibit the initiation of clot formation

use this code as reference to generate the markdown text
markdown_lines = pathway_desc_sub_df.apply(
                    lambda row: f'- Pathway name: "{row["pathway_name"]}". '
                    f'Pathway function: "{row["Description"]}". ',
                    axis=1,
                )
"""
