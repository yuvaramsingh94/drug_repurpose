## About
## This is a Multi agent system that connects experts from three different domains.
## It involves agents such as drug, protein, and disease. This model uses a planner agent
## setup where the planner. Idea from [ai-langgraph-multi-agent](https://github.com/botextractai/ai-langgraph-multi-agent)


##TO RUN python code/python_code/multi_agent.py

## Imports
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import chromadb
from chromadb.api.types import Documents, Embeddings, EmbeddingFunction
import sys
import time
from sklearn.preprocessing import normalize
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from questions import disease_question_temp as disease_question  ##! Change

## For server
sys.path.insert(0, os.path.abspath(os.path.join("utils")))
sys.path.insert(0, os.getcwd())
from tools import (
    medlineplus_dis_property_initializer,
    drug_initializer,
)
from utils.utils import (
    API_setup,
    filter_dataframe_by_selected_ids,
)
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode

from tools import (
    drug_retrieve_and_format_queries_agent,
    disease_retrieve_and_format_queries_agent,
)

# from IPython.display import Image, display
import pandas as pd
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import yaml
from state_definition import (
    Multi_agent_state_v2,
    Planner_structured_output_new,
)

from state_definition import (
    RAG_questions,
    Query_structured_output,
    candidate_extraction,
    recommendation_pro,
)
from agents import (
    planner_agent_initializer,
    disease_agent_initializer,
    drug_agent_initializer,
    drug_disease_condition_initializer,
)
from system_message import (
    planner_sys_msg,
    disease_treatment_summary_sys_msg,
    disease_protein_summary_sys_msg,
    disease_pathway_summary_sys_msg,
    disease_protein_RAG_sys_msg,
    disease_pathway_RAG_sys_msg,
    disease_retrever_sys_msg,
    disease_expert_with_drug_struc_op_sys_msg,
    disease_expert_with_drug_sys_msg,  # The final Pro model prompt
    drug_protein_summary_sys_msg,
    drug_pathway_summary_sys_msg,
    drug_indication_summary_sys_msg,
    drug_pathway_RAG_sys_msg,
    drug_protein_RAG_sys_msg,
    drug_retrever_sys_msg,
    drug_expert_with_disease_struc_op_sys_msg,
    drug_expert_with_disease_sys_msg,  # The final Pro model prompt
    db_extractor_sys_msg,  ##Extract the drug and disease candidates name and ID
    drug_RAG_query_augmentation_sys_msg,
    disease_RAG_query_augmentation_sys_msg,
)
from langchain_core.messages import HumanMessage

## TODO: Setup the API
API_setup()
## TODO: Load the required files
device = "cpu"  # torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

RESULT_NAME = "2_1_26_augmentation_Q3"  ##! Change
CONFIG_FOLDER = os.path.join("configuration")
RAG_CONFIG_PATH = os.path.join(CONFIG_FOLDER, "RAG.yaml")
UniProtKB_CONFIG_PATH = os.path.join(CONFIG_FOLDER, "UniProtKB.yaml")
DB_config_PATH = os.path.join(CONFIG_FOLDER, "DrugBank.yaml")
KEGG_m_CONFIG_PATH = os.path.join(CONFIG_FOLDER, "KEGG_medicus.yaml")
DATABASE_PATH = os.path.join(CONFIG_FOLDER, "database_path.yaml")
LLM_SETTINGS = os.path.join(CONFIG_FOLDER, "LLM.yaml")
SYSTEM_SETTINGS = os.path.join(CONFIG_FOLDER, "system_settings.yaml")
SQLPATH = os.path.join("memory", f"{RESULT_NAME}.sqlite")
RESULT_FOLDER_PATH = os.path.join("results", RESULT_NAME)
os.makedirs(RESULT_FOLDER_PATH, exist_ok=True)
MODEL_WT = os.path.join("weights")
MODEL_DATABASE = os.path.join(
    "database", "database_BioLinkBERT_small_mol"
)  # vectordb_BioLinkBERT_small_mol
# COLLECTION_NAME = "drug_mechanism_of_action"
MODEL_NAME = "michiyasunaga/BioLinkBERT-large"  # "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
with open(DATABASE_PATH, "r") as f:
    DATABASE_PATH = yaml.safe_load(f)
## Load the RAG configuation
with open(RAG_CONFIG_PATH, "r") as f:
    RAG_CONFIG = yaml.safe_load(f)
## Load the important column
with open(UniProtKB_CONFIG_PATH, "r") as f:
    UniProtKB = yaml.safe_load(f)
## Load the important column
with open(KEGG_m_CONFIG_PATH, "r") as f:
    KEGG_m = yaml.safe_load(f)
## Load the LLM settings
with open(LLM_SETTINGS, "r") as f:
    LLM_settings = yaml.safe_load(f)
## Load the system settings
with open(SYSTEM_SETTINGS, "r") as f:
    system_settings = yaml.safe_load(f)
## Load the DrugBank settings
with open(DB_config_PATH, "r") as f:
    DB_settings = yaml.safe_load(f)


UniProtKB_columns = UniProtKB["UniProtKB_columns"]
KEGG_m_columns = KEGG_m["KEGG_Medicus_columns"]

## Load the dataframes
drug_df = pd.read_csv(os.path.join(DATABASE_PATH["DRUG_PATH"])).fillna("")
drug_df["name"] = drug_df["name"].str.lower()
## Load the medlineplus genetic disease
medlineplus_disease = pd.read_csv(
    os.path.join(DATABASE_PATH["MEDLINEPLUS_DISEASE_PATH"])
).fillna("")
## TODO: Can we use DOID instead of the name as the database ID?
## TODO: use the medlineplus csv with the Database_ID
## use name as the database ID.
## check if "name" column is unique
# if not medlineplus_disease["name"].is_unique:
#     raise ValueError("The 'name' column in medlineplus_disease must be unique.")
# ## generate unique database ID with DIS:xxxxxx number starting from 000001
# medlineplus_disease["Database_ID"] = "DIS:" + (medlineplus_disease.index + 1).astype(
#     str
# ).str.zfill(6)
# medlineplus_disease["Database_ID"] = medlineplus_disease["name"].astype(str)
## Load the medlineplus gene
medlineplus_gene = pd.read_csv(
    os.path.join(DATABASE_PATH["MEDLINEPLUS_GENE_PATH"])
).fillna("")
## Load the medlineplus gene disease mapping
medlineplus_gene_disease_mapping = pd.read_csv(
    os.path.join(DATABASE_PATH["MEDLINEPLUS_GENE_DIS_MAPPING_PATH"])
).fillna("")
## Load the medlineplus gene pathway mapping
medlineplus_gene_pathway_mapping = pd.read_csv(
    os.path.join(DATABASE_PATH["MEDLINEPLUS_GENE_PATHWAY_MAPPING_PATH"])
).fillna("")
## Load the pathway description
senor_pathway_description = pd.read_csv(
    os.path.join(DATABASE_PATH["SIGNOR_PATHWAY_PATH"])
).fillna("")
disease_treatments_relation = pd.read_csv(
    os.path.join(DATABASE_PATH["DISEASE_TREATMENTS_RELATION_PATH"])
).fillna("")

## Make a dictionary of the dataframes and use it for passing to the initializers.
## This will make it clean and uniform

dataframes_dict = {
    "drug_df": drug_df,
    "medlineplus_disease": medlineplus_disease,
    "medlineplus_gene": medlineplus_gene,
    "medlineplus_gene_disease_mapping": medlineplus_gene_disease_mapping,
    "medlineplus_gene_pathway_mapping": medlineplus_gene_pathway_mapping,
    "senor_pathway_description": senor_pathway_description,
    "disease_treatments_relation": disease_treatments_relation,
}

## Load RAG models and tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=RAG_CONFIG["RAG_MODEL"], cache_dir=MODEL_WT
)
model = AutoModel.from_pretrained(
    pretrained_model_name_or_path=RAG_CONFIG["RAG_MODEL"],
    cache_dir=MODEL_WT,
    device_map=device,
    torch_dtype=torch.float16,
).to(device)


def get_biobert_embedding(text):
    """Generate BioBERT embeddings with GPU optimization"""
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=384
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


embedding_function = BioBERTEmbeddingFunction()

chroma_client = chromadb.PersistentClient(path=MODEL_DATABASE)
MoA_collection = chroma_client.get_collection(
    name=RAG_CONFIG["MoA_collection"], embedding_function=embedding_function
)

PD_collection = chroma_client.get_collection(
    name=RAG_CONFIG["PD_collection"], embedding_function=embedding_function
)
## Load disease chroma collections
cause_collection = chroma_client.get_collection(
    name=RAG_CONFIG["cause_collection"], embedding_function=embedding_function
)

gene_fn_collection = chroma_client.get_collection(
    name=RAG_CONFIG["gene_function_collection"], embedding_function=embedding_function
)


disease_property = medlineplus_dis_property_initializer(dataframes_dict=dataframes_dict)

drugs_information = drug_initializer(
    dataframes_dict=dataframes_dict, property=DB_settings["Property"]
)

## Model initialization
llm = ChatGoogleGenerativeAI(
    model=LLM_settings["model"], temperature=LLM_settings["temperature"]
)
llm_pro = ChatGoogleGenerativeAI(
    model=LLM_settings["pro_model"], temperature=LLM_settings["temperature"]
)
llm_lite = ChatGoogleGenerativeAI(
    model=LLM_settings["lite_model"], temperature=LLM_settings["temperature"]
)

drug_retrieve_and_format_queries = drug_retrieve_and_format_queries_agent(
    query_filter_sys_msg=drug_retrever_sys_msg,
    dataframes_dict=dataframes_dict,
    filter_dataframe_by_selected_ids=filter_dataframe_by_selected_ids,
)

disease_retrieve_and_format_queries = disease_retrieve_and_format_queries_agent(
    query_filter_sys_msg=disease_retrever_sys_msg,
    dataframes_dict=dataframes_dict,
    filter_dataframe_by_selected_ids=None,
    disease_property=disease_property,  ## Makes it easier to construct the markdown text. Can take list of disease names
)

##* drug
drug_tools = [drugs_information]
drug_tool_node = ToolNode(tools=drug_tools)
drug_llm = llm.bind_tools(
    drug_tools,
)

## Planner
planner_llm = llm.with_structured_output(Planner_structured_output_new)
## Reflectiondisease_sys_msg
reflection_llm = llm


# Initialise the graph with the agent state

planner_agent = planner_agent_initializer(
    sys_msg={"planner_human_input_processing": planner_sys_msg}, llm=planner_llm
)

drug_agent = drug_agent_initializer(
    sys_msg={
        # "disease_agent": disease_sys_msg,
        "protein_expert_drug_summary": drug_protein_summary_sys_msg,
        "pathway_expert_drug_summary": drug_pathway_summary_sys_msg,
        "indication_expert_drug_summary": drug_indication_summary_sys_msg,
        "protein_expert_drug_RAG": drug_protein_RAG_sys_msg,
        "pathway_expert_drug_RAG": drug_pathway_RAG_sys_msg,
        "drug_retriever": drug_retrever_sys_msg,
        "drug_pro_prompt": drug_expert_with_disease_struc_op_sys_msg,
        "db_extractor": db_extractor_sys_msg,
        "Drug_RAG_query_augmentation": drug_RAG_query_augmentation_sys_msg,
    },
    state_definition={
        "RAG_structure": RAG_questions,
        "Query_selection": Query_structured_output,
        "db_extractor": candidate_extraction,
        "recommendation_pro": recommendation_pro,
    },
    llm=llm,
    llm_pro=llm_pro,
    llm_lite=llm_lite,
    drug_df_tool=drugs_information,
    # drug_RAG_tool,
    RAG_collection={"MoA_collection": MoA_collection, "PD_collection": PD_collection},
    retrieve_and_format_queries=drug_retrieve_and_format_queries,
    dataframes_dict=dataframes_dict,
    configuration=system_settings["system_settings"],
    result_path=RESULT_FOLDER_PATH,
)
disease_agent = disease_agent_initializer(
    sys_msg={
        # "disease_agent": disease_sys_msg,
        "treatment_expert_disease_summary": disease_treatment_summary_sys_msg,
        "protein_expert_disease_summary": disease_protein_summary_sys_msg,
        "pathway_expert_disease_summary": disease_pathway_summary_sys_msg,
        "protein_expert_disease_RAG": disease_protein_RAG_sys_msg,
        "pathway_expert_disease_RAG": disease_pathway_RAG_sys_msg,
        "disease_retriever": disease_retrever_sys_msg,
        "disease_pro_prompt": disease_expert_with_drug_struc_op_sys_msg,
        "db_extractor": db_extractor_sys_msg,
        "Disease_RAG_query_augmentation": disease_RAG_query_augmentation_sys_msg,
    },
    ## TODO: Finish this
    state_definition={
        "RAG_structure": RAG_questions,
        "Query_selection": Query_structured_output,
        "db_extractor": candidate_extraction,
        "recommendation_pro": recommendation_pro,
    },  # {"disease_query":,"disease_query_selection":},
    llm=llm,
    llm_pro=llm_pro,
    llm_lite=llm_lite,
    disease_df_tool=disease_property,
    # disease_RAG_tool,
    RAG_collection={
        "cause_collection": cause_collection,
        "gene_fn_collection": gene_fn_collection,
    },
    retrieve_and_format_queries=disease_retrieve_and_format_queries,
    dataframes_dict=dataframes_dict,
    configuration=system_settings["system_settings"],
)

drug_disease_condition = drug_disease_condition_initializer(
    next_agent_information="next_agent_information",
    next_agent_key="agent_name",
    max_agent_count=2,
    max_reflection_agent_count=None,
)

builder = StateGraph(Multi_agent_state_v2)
builder.add_node("planner", planner_agent)
builder.add_node("drug_agent", drug_agent)
builder.add_node("disease_agent", disease_agent)
# builder.add_node("research_critique", research_critique_node)

## TODO: Build the graph
builder.add_edge(START, "planner")
builder.add_conditional_edges("planner", drug_disease_condition)
## Add edge from tools back to agent
builder.add_edge("drug_agent", "planner")
builder.add_edge("disease_agent", "planner")

# memory = MemorySaver()
MEMORY_SAVER = os.path.join(SQLPATH)
conn = sqlite3.connect(MEMORY_SAVER, check_same_thread=False)
memory = SqliteSaver(conn)
CureFinderAI_graph = builder.compile(checkpointer=memory)
# CureFinderAI_graph.get_graph().draw_mermaid_png(
#    output_file_path=os.path.join("figure", "bidir_agent.png")
# )


threads = [str(i) for i in range(292, 292 + len(disease_question))]

for i in range(len(disease_question)):
    print("Thread", i)
    config = {"configurable": {"thread_id": threads[i]}}
    # input_message = HumanMessage(content="How can I implement a neural network in a quantum computer (with code)?")
    print("QUESTION:", disease_question[i])
    messages = {"messages": [disease_question[i]]}
    # Invoke the graph with robust error handling
    try:
        print("Starting graph execution...")
        print("Question", messages)
        response = CureFinderAI_graph.invoke(messages, config=config)
        print("Response received successfully")
        for m in response["messages"]:
            if m and hasattr(m, "pretty_print"):
                m.pretty_print()
            elif m and hasattr(m, "content"):
                print(f"Message content: {m.content}")
            else:
                print(f"Unknown message format: {m}")
    except Exception as e:
        print(f"Error occurred in main execution: {str(e)}")
    print("Sleep for 65 Sec....")
    time.sleep(65)  ## Sleep for 65 sec to avoid overrunning RPM
