from langchain_core.messages import AnyMessage
from typing_extensions import TypedDict
from typing import Annotated
from operator import add
from langgraph.graph import add_messages
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List

## Define the entity extraction structure
class entity_structure_item(BaseModel):

    entity: str = Field(
        ..., description="The name of the entity like a drug or disease name."
    )
    category: str = Field(
        ...,
        description="The category of the entity. Does it belong to a drug or a disease?.",
    )

## Define the main output model that expects a list of the items above
class entity_structure(BaseModel):

    entity_list: List[entity_structure_item] = Field(
        ..., description="A list of extracted entity and its category."
    )

## Define a model for a single query and its associated IDs
class QueryItem(BaseModel):
    """A single query and its corresponding database IDs."""

    query: str = Field(..., description="The user's query string.")
    db_ids: List[str] = Field(
        ..., description="A list of selected database IDs for the query."
    )

# Define the main output model that expects a list of the items above
class Query_structured_output(BaseModel):
    """A list of queries and their selected database IDs."""

    query_list: List[QueryItem] = Field(
        ..., description="A list containing query-ID pairs."
    )

class RAG_questions(BaseModel):
    """Information for RAG query generator"""

    list_of_queries: List[str] = Field(
        ..., description="The list of queries to be answered"
    )

class candidate_extraction(BaseModel):
    """Extract the details of the recommended and not recommended entities"""

    recommended_drug_IDs: List[str] = Field(
        ..., description="The list of recommended drug IDs"
    )
    recommended_drug_names: List[str] = Field(
        ..., description="The list of recommended drug names"
    )
    recommended_disease_IDs: List[str] = Field(
        ..., description="The list of recommended disease IDs"
    )
    recommended_disease_names: List[str] = Field(
        ..., description="The list of recommended disease names"
    )
    not_recommended_drug_IDs: List[str] = Field(
        ..., description="The list of not_recommended drug IDs"
    )
    not_recommended_drug_names: List[str] = Field(
        ..., description="The list of not_recommended drug names"
    )
    not_recommended_disease_IDs: List[str] = Field(
        ..., description="The list of not_recommended disease IDs"
    )
    not_recommended_disease_names: List[str] = Field(
        ..., description="The list of not_recommended disease names"
    )

class recommendation_pro(BaseModel):
    """Detailed explanation of recommended and not recommended entities with structured information"""

    detailed_explanation_of_recommended_and_not_recommended_entities: str = Field(
        ...,
        description="A detailed explanation of the recommended and not recommended drug or disease",
    )
    recommended_drug_IDs: List[str] = Field(
        ..., description="The list of recommended drug IDs"
    )
    recommended_drug_names: List[str] = Field(
        ..., description="The list of recommended drug names"
    )
    recommended_disease_IDs: List[str] = Field(
        ..., description="The list of recommended disease IDs"
    )
    recommended_disease_names: List[str] = Field(
        ..., description="The list of recommended disease names"
    )
    not_recommended_drug_IDs: List[str] = Field(
        ..., description="The list of not_recommended drug IDs"
    )
    not_recommended_drug_names: List[str] = Field(
        ..., description="The list of not_recommended drug names"
    )
    not_recommended_disease_IDs: List[str] = Field(
        ..., description="The list of not_recommended disease IDs"
    )
    not_recommended_disease_names: List[str] = Field(
        ..., description="The list of not_recommended disease names"
    )

class Planner_structured_output(BaseModel):
    """Information for a planner agent."""

    think: str = Field(..., description="The thinking step for solving the problem")
    sub_problem: str = Field(
        ..., description="The ordered list of sub problem breakdown"
    )
    next_agent_name: str = Field(
        ..., description="The name of the next agent to be called"
    )
    next_agent_query: str = Field(
        ..., description="The query which is to be passed to the next agent"
    )
    final_answer: str = Field(
        ...,
        description="The final response to the user query. Populate this only if you have reached a conclusion. else return an empty string",
    )

class Planner_structured_output_new(BaseModel):
    """Information for a planner agent."""

    think: str = Field(..., description="The thinking step for solving the problem")
    sub_problem: str = Field(
        ..., description="The ordered list of sub problem breakdown"
    )
    next_agent_name: str = Field(
        ..., description="The name of the next agent to be called"
    )
    next_agent_query: str = Field(
        ..., description="The query which is to be passed to the next agent"
    )
    final_answer: str = Field(
        ...,
        description="The final response to the user query. Populate this only if you have reached a conclusion. else return an empty string",
    )
    sub_task: str = Field(
        ...,
        description="Type of the task that is to be carried out. Select one from 'disease_summary','find_diseases_for_a_drug','drug_summary','find_drugs_for_a_disease'",
    )
    disease_name: list[str] = Field(
        ..., description="The name of the disease if mentioned in the text"
    )
    drug_name: str = Field(
        ..., description="The name of the drug if mentioned in the text"
    )

## This is a dict that has information about which node to go next and what is the query.
class Next_agent(TypedDict):

    agent_name: str
    query: str

class Multi_agent_state_v1(TypedDict):

    messages: Annotated[list[AnyMessage], add_messages]
    reflection_count: Annotated[int, add] = 0
    protein_count: Annotated[int, add] = 0
    drug_count: Annotated[int, add] = 0
    disease_count: Annotated[int, add] = 0
    next_agent: Next_agent

class selected_query_ids_dict(TypedDict):

    query: list[str]
    selected_query_ids: list[list[str]]

class Multi_agent_state_v2(TypedDict):

    messages: Annotated[list[AnyMessage], add_messages]
    sub_problems: Annotated[list[str], add]
    owner: Annotated[list[str], add]
    disease_name: list[list[str]]  
    drug_name: list[str]  
    next_agent_information: Annotated[list[dict], add]
    drug_count: Annotated[int, add] = 0
    disease_count: Annotated[int, add] = 0
    disease_query: list[str]
    drug_query: list[str]
    drug_intermediate_queries: list[str]
    disease_intermediate_queries: list[str]
    disease_primary_selected_ids: list[str]
    drug_primary_selected_ids: list[str]
    disease_final_selected_ids: selected_query_ids_dict
    drug_final_selected_ids: selected_query_ids_dict
    disease_query_selection_message: str
    drug_query_selection_message: str
    ## Store the extracted IDs and Names
    recommended_drug_IDs: List[str]
    recommended_drug_names: List[str]
    recommended_disease_IDs: List[str]
    recommended_disease_names: List[str]

    not_recommended_drug_IDs: List[str]
    not_recommended_drug_names: List[str]
    not_recommended_disease_IDs: List[str]
    not_recommended_disease_names: List[str]
