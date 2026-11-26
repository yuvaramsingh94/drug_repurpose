"""
About
This file contains the code for the bidirectional agents
"""

import os
import re
import pandas as pd
from fuzzy_search import match_finder
from state_definition import Multi_agent_state_v2
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Literal, Any, Callable
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    filter_messages,
    ToolMessage,
)
from chromadb.api.models.Collection import Collection
from langchain_core.pydantic_v1 import BaseModel


def planner_agent_initializer(sys_msg: dict, llm: ChatGoogleGenerativeAI):
    """Initializer for the planner agent.

    Args:
        sys_msg (dict): dictionary containing system messages for the planner.
        llm (ChatGoogleGenerativeAI): The language model to be used by the planner with structured output.
    """

    def planner_agent(state: Multi_agent_state_v2) -> Multi_agent_state_v2:
        """Planner agent for managing the multi-agent state.

        Args:
            state (Multi_agent_state_v2): The current state of the multi-agent system.

        Returns:
            Multi_agent_state_v2: The updated state of the multi-agent system.
        """

        ## TODO: Check if the state's last message is an human query
        ## TODO: if the message is the startign message, then decide which agent to call
        # Check if the last message is from human or AI
        # Preprocess the message accordingly
        if state["messages"][-1].type == "human":

            history_msg = sys_msg["planner_human_input_processing"] + state["messages"]

            op = llm.invoke(history_msg)

            next_agent_information = {
                "agent_name": op.next_agent_name,
                "query": op.next_agent_query,
                "sub_task": op.sub_task,
            }

        elif state["messages"][-1].type == "ai":

            # Filter the tool calling messages as they have empty content and it causes issue with gemini
            messages_list = [i for i in state["messages"] if len(i.content) > 0]

            ## Filter the tool messages as planner does not have any tool by itself
            history_msg = sys_msg["planner_human_input_processing"] + filter_messages(
                messages_list, include_types=[HumanMessage, AIMessage]
            )

            ##* Solution
            ##* Repeated calling of same agent
            ##* If the message is labled as AI, the planner is repeatedly calling the same agent
            ##* Recast the AIMessage to HumanMessage
            history_msg[-1] = HumanMessage(content=history_msg[-1].content)

            op = llm.invoke(history_msg)

            next_agent_information = {
                "agent_name": op.next_agent_name,
                "query": op.next_agent_query,
                "AI_report_previous_agent": messages_list[-1].content,
                "sub_task": op.sub_task,
            }

        ## Construct a AI message using the planner's think
        response_msg = AIMessage(content=f"Response: {op.think}", name="planner")

        return {
            "messages": [response_msg],
            "Think": [op.think],
            "sub_problems": [op.sub_problem],
            "next_agent_information": [next_agent_information],
            "disease_name": [op.disease_name],
            "drug_name": [op.drug_name],
        }

    return planner_agent


# sys_msg "disease_classify" is used to classify if the query is about a disease or to find diseases that can match a drug
def disease_agent_initializer(
    sys_msg: dict,
    state_definition: dict[str:BaseModel],
    llm: ChatGoogleGenerativeAI,
    llm_pro: ChatGoogleGenerativeAI,
    llm_lite: ChatGoogleGenerativeAI,
    disease_df_tool: Callable[[list[str], list[str]], str],
    RAG_collection: dict[str:Collection],
    dataframes_dict: dict[str : pd.DataFrame],
    retrieve_and_format_queries: Callable[..., Any],
    configuration: dict,
):
    """Initialize the disease agent.

    Args:
        sys_msg (dict): System messages for the agent.
        state_definition (dict[str:BaseModel]): State definition for the agent.
        llm (ChatGoogleGenerativeAI): Language model for the agent.
        llm_pro (ChatGoogleGenerativeAI): Pro version of the language model for the agent.
        disease_df_tool (Callable[[list[str], list[str]], str]): Tool for retrieving disease information from a dataframe.
        RAG_collection (dict[str:Collection]): Collection of RAG documents.
        dataframes_dict (dict[str:pd.DataFrame]): Dictionary of dataframes for various data sources.
        retrieve_and_format_queries (Callable[[list[str]], list[str]]): RAG function to retrieve from queries.
    """

    def disease_agent(state: Multi_agent_state_v2) -> Multi_agent_state_v2:
        """Disease agent for processing disease-related queries.

        Args:
            state (Multi_agent_state_v2): Current state of the multi-agent system.

        Raises:
            ValueError: If the state does not have information about drug when asking to find diseases.

        Returns:
            Multi_agent_state_v2: Updated state after processing the disease query.
        """

        ## Check if the sub task is to find diseases for a drug or to get a summary of a disease
        if state["next_agent_information"][-1]["sub_task"] == "disease_summary":

            ## Get the name of the disease from the state
            disease_name = state["disease_name"][-1]

            ## Get the detailed information about the disease
            disease_property = disease_df_tool(disease_name)

            ## Split the disease property into disease info and treatment info if treatment info is present
            if "## Known treatments for the disease" in disease_property:
                disease_property, treatment_info = disease_property.split(
                    "## Known treatments for the disease"
                )
                disease_property = disease_property.strip()
                treatment_info = treatment_info.strip()
            else:
                treatment_info = ""
                disease_property = disease_property.strip()

            ## pass the response through panel of experts
            response_list = []
            for panel_agent in [
                "protein_expert_disease_summary",
                "pathway_expert_disease_summary",
                "treatment_expert_disease_summary",
            ]:
                ## For protein and pathway expert, only pass the disease property
                history_msg = sys_msg[panel_agent] + [
                    HumanMessage(content=disease_property)
                ]

                ## For treatment expert, pass both treatment info if available
                if panel_agent == "treatment_expert_disease_summary":

                    ## End the loop if we dont want treatment expert
                    if not configuration["use_treatment_expert"]:
                        break

                    ## If we have the treatment information
                    if len(treatment_info) > 0:
                        history_msg = sys_msg[panel_agent] + [
                            HumanMessage(content=treatment_info)
                        ]
                    else:
                        ## We dont have treatment info, so skip the treatment expert
                        break

                op = llm.invoke(history_msg)

                response_list.append(op)

            ## Combine the responses from the panel of experts into a Q&A format
            contents = "\n\n".join([res.content for res in response_list])
            contents = (
                f"Detailed summary of {disease_name[0]} in the question and answer format. "
                + contents
            )

            ## Cast it as an AI message.
            op_msg = AIMessage(content=contents, name="disease_agent")

            ## Wrap the disease property as a Tool message for keeping track
            disease_property = ToolMessage(
                content=disease_property,
                tool_call_id="disease_agent_tool_summary",
            )

            return {
                "messages": [disease_property, op_msg],
            }

        elif (
            state["next_agent_information"][-1]["sub_task"]
            == "find_diseases_for_a_drug"
        ):

            ## Keep the drug summary from the `drug_agent`
            filtered_msg = filter_messages(
                state["messages"], include_names=["drug_agent"]
            )

            ## Recasting to avoid the error: Invalid argument provided to Gemini: 400 Please ensure that single turn requests end with a user role or the role field is empty.
            if len(filtered_msg) == 1:
                filtered_msg = [HumanMessage(content=filtered_msg[0].content)]
            elif len(filtered_msg) == 0:
                raise ValueError("No information from drug agent to find diseases")

            ## Generate queries based on the drug summary document
            response_list = []
            for panel_agent in [
                "protein_expert_disease_RAG",
                "pathway_expert_disease_RAG",
            ]:
                ## Messages with expert specific system prompt
                history_msg = sys_msg[panel_agent] + filtered_msg

                ## Use the structured output
                llm_disease_query = llm.with_structured_output(
                    state_definition["RAG_structure"]
                )

                ## Get the list of queries from the LLM
                op = llm_disease_query.invoke(history_msg)
                response_list.extend(op.list_of_queries)

            ## LLM for selecting the best query and answer pairs
            ## ? Should we also augment the RAG queries to get better results?
            ## ? Can this be done using local LLM?
            query_selection_llm = llm.with_structured_output(
                state_definition["Query_selection"]
            )

            ## Retrieve the relevant documents from the RAG collection and filter them
            markdown_text, db_id_list, selected_db_ids, query_selection_prompt = (
                retrieve_and_format_queries(
                    response_list, RAG_collection, query_selection_llm, n_results=5
                )
            )

            history_msg = (
                sys_msg["disease_pro_prompt"]
                + filtered_msg
                + [HumanMessage(content=markdown_text)]
            )

            ## Final recommendation from the drug RAG. This is done using PRO model to get the best results.
            llm_pro_struct = llm_pro.with_structured_output(
                state_definition["recommendation_pro"]
            )
            op = llm_pro_struct.invoke(history_msg)

            return {
                "messages": [
                    ToolMessage(
                        content=markdown_text,
                        name="disease_agent_tool",
                        tool_call_id="disease_agent_tool_RAG",
                    ),
                    AIMessage(
                        content=op.detailed_explanation_of_recommended_and_not_recommended_entities,
                        name="disease_agent",
                    ),
                ],
                "disease_query": response_list,
                "disease_primary_selected_ids": db_id_list,
                "disease_final_selected_ids": selected_db_ids,
                "disease_query_selection_message": query_selection_prompt,
                "recommended_disease_IDs": op.recommended_disease_IDs,
                "recommended_disease_names": op.recommended_disease_names,
                "not_recommended_disease_IDs": op.not_recommended_disease_IDs,
                "not_recommended_disease_names": op.not_recommended_disease_names,
            }

    return disease_agent


def drug_agent_initializer(
    sys_msg: dict,
    state_definition: dict[str:BaseModel],
    llm: ChatGoogleGenerativeAI,
    llm_pro: ChatGoogleGenerativeAI,
    llm_lite: ChatGoogleGenerativeAI,
    drug_df_tool: Callable,
    RAG_collection: dict[str:Collection],
    retrieve_and_format_queries,
    dataframes_dict: dict,
    configuration: dict,
    result_path: str,
):
    """Initialize the drug agent.

    Args:
        sys_msg (dict): System messages for the agent.
        state_definition (dict[str:BaseModel]): State definitions for the agent.
        llm (ChatGoogleGenerativeAI): Language model for the agent.
        llm_pro (ChatGoogleGenerativeAI): Pro version of the language model for the agent.
        drug_df_tool (Callable): Tool for retrieving drug information.
        RAG_collection (dict[str:Collection]): RAG collection for retrieving relevant documents.
        retrieve_and_format_queries (Callable): Function for retrieving and formatting queries.
        dataframes_dict (dict): Dictionary of dataframes for the agent.
    """

    def drug_agent(state: Multi_agent_state_v2) -> Multi_agent_state_v2:
        """Drug agent for processing drug-related queries.

        Args:
            state (Multi_agent_state_v2): Current state of the multi-agent system.

        Raises:
            ValueError: _description_

        Returns:
            Multi_agent_state_v2: Updated state after processing the drug query.
        """

        ## Check if the sub task is to find drug for a disease or to get a summary of a drug
        if state["next_agent_information"][-1]["sub_task"] == "drug_summary":

            ## Get the name of the drug from the state
            drug_name = state["drug_name"][-1]

            ## Get the detailed information about the drug
            drug_property = drug_df_tool(drug_name)

            ## pass the response through panel of experts
            response_list = []
            for panel_agent in [
                "protein_expert_drug_summary",
                "pathway_expert_drug_summary",
                "indication_expert_drug_summary",
            ]:

                history_msg = sys_msg[panel_agent] + [
                    HumanMessage(content=drug_property)
                ]
                op = llm.invoke(history_msg)
                response_list.append(op)

            ## Combine the responses from the panel of experts into a Q&A format
            contents = "\n\n".join([res.content for res in response_list])

            ## Cast it as tool message for keeping track
            drug_property_msg = ToolMessage(
                content=drug_property,
                tool_call_id="drug_agent_tool_summary",
            )
            op_msg = AIMessage(content=contents, name="drug_agent")

            return {
                "messages": [drug_property_msg, op_msg],  # "owner": ["disease_agent"]
            }

        elif (
            state["next_agent_information"][-1]["sub_task"]
            == "find_drugs_for_a_disease"
        ):

            ## Keep the disease summary from the `disease_agent`
            filtered_msg = filter_messages(
                state["messages"], include_names=["disease_agent"]
            )

            ## Recasting to avoid the error: Invalid argument provided to Gemini: 400 Please ensure that single turn requests end with a user role or the role field is empty.
            if len(filtered_msg) == 1:
                filtered_msg = [HumanMessage(content=filtered_msg[0].content)]
            elif len(filtered_msg) == 0:
                raise ValueError("No information from disease agent to find drugs")

            ## Generate queries based on the disease summary document
            intermediate_queries = []
            response_list = []
            for panel_agent in [
                "protein_expert_drug_RAG",
                "pathway_expert_drug_RAG",
            ]:

                ## Messages with expert specific system prompt
                history_msg = sys_msg[panel_agent] + filtered_msg

                ## LLM with structured output for the RAG queries
                llm_drug_query = llm.with_structured_output(
                    state_definition["RAG_structure"]
                )
                op = llm_drug_query.invoke(history_msg)
                intermediate_queries.extend(op.list_of_queries)

                ## TODO add the query augmentation and reformatting here
                if configuration["augment_drug_RAG_queries"]:

                    ## Reformat the message to have list of queries
                    reformatted_msg = ""
                    for query_ in op.list_of_queries:
                        reformatted_msg += f"- {query_}\n "

                    history_msg = sys_msg["Drug_RAG_query_augmentation"] + [
                        HumanMessage(content=reformatted_msg)
                    ]

                    ## Use lite model with structured output for query augmentation
                    llm_drug_augmented_query = llm_lite.with_structured_output(
                        state_definition["RAG_structure"]
                    )
                    op = llm_drug_augmented_query.invoke(history_msg)

                response_list.extend(op.list_of_queries)

            ## LLM for selecting the best query and answer pairs
            ## ? Should we also augment the RAG queries to get better results?
            ## ? Can this be done using local LLM?
            query_selection_llm = llm.with_structured_output(
                state_definition["Query_selection"]
            )

            ## Modify the queries. Improve the quality of the generated queries
            ## Append the other names for the genes in the query
            ## Pattern to find gene ids
            pattern = r"\b[A-Z0-9]{2,}\b"

            ## Iterate through the queries and find gene ids to augment with alternative gene names
            for idx, query_ in enumerate(response_list):

                gene_ids = re.findall(pattern, query_)

                if len(gene_ids) > 0:

                    gene_ids = [gid.strip() for gid in gene_ids]
                    gene_other_names = dataframes_dict["medlineplus_gene"].loc[
                        dataframes_dict["medlineplus_gene"]["gene_name"].isin(gene_ids)
                    ]["other_names"]
                    gene_other_names = [
                        name.strip().replace("\n", ", ") for name in gene_other_names
                    ]

                    ## Add the other names to the query
                    # Commented out now.
                    ## TODO: find a better way to augment the other_name
                    # updated_query = query_
                    # for other_name in gene_other_names:
                    #     updated_query += f" other names: {other_name}"
                    # query_ = updated_query

                    response_list[idx] = query_

            ## Retrieve the relevant documents from the RAG collection and filter them
            (
                _,
                markdown_text,
                db_id_list,
                selected_db_ids,
                query_selection_prompt,
            ) = retrieve_and_format_queries(
                response_list, RAG_collection, query_selection_llm, n_results=5
            )

            history_msg = (
                sys_msg["drug_pro_prompt"]
                + filtered_msg
                + [HumanMessage(content=markdown_text)]
            )
            ## TODO: Save this filtered_msg contents, markdown_text as a MD file locally for future reference
            with open(os.path.join(result_path, "drug_recommendations.md"), "w") as f:
                f.write("# Filtered Message\n")
                f.write(filtered_msg[0].content)
                f.write("\n# Informations about drugs\n")
                f.write(markdown_text)
            ## Final recommendation from the drug RAG.
            # This is done using PRO model to get the best results.
            llm_pro_struct = llm_pro.with_structured_output(
                state_definition["recommendation_pro"]
            )
            op = llm_pro_struct.invoke(history_msg)

            return {
                "messages": [
                    ToolMessage(
                        content=markdown_text,
                        name="drug_agent_tool",
                        tool_call_id="drug_agent_tool_RAG",
                    ),
                    AIMessage(
                        content=op.detailed_explanation_of_recommended_and_not_recommended_entities,
                        name="drug_agent",
                    ),
                ],
                "drug_query": response_list,
                "drug_intermediate_queries": intermediate_queries,
                "drug_primary_selected_ids": db_id_list,
                "drug_final_selected_ids": selected_db_ids,
                "drug_query_selection_message": query_selection_prompt,
                "recommended_drug_IDs": op.recommended_drug_IDs,
                "recommended_drug_names": op.recommended_drug_names,
                "not_recommended_drug_IDs": op.not_recommended_drug_IDs,
                "not_recommended_drug_names": op.not_recommended_drug_names,
            }

    return drug_agent


def drug_disease_condition_initializer(
    next_agent_information: str = "next_agent_information",
    next_agent_key: str = "agent_name",
    max_agent_count: int = None,
    max_reflection_agent_count: int = None,
) -> Callable[..., Literal["drug_agent", "disease_agent", "__end__"]]:
    """Conditional edge function to decide which agent to call next.

    Args:
        next_agent_information (str, optional): Name of the key in the updated state. Defaults to "next_agent_information".
        next_agent_key (str, optional): Name of the Key in the updated state. Defaults to "agent_name".
        max_agent_count (int, optional): maximum agent count. Defaults to None.
        max_reflection_agent_count (int, optional): Maximum reflection count. Defaults to None.

    Returns:
        Callable[..., Literal["drug_agent", "disease_agent", "__end__"]]: The conditional edge function.
    """

    def drug_disease_condition(
        state: dict[str, Any],
        next_agent_information: str = next_agent_information,
        next_agent_key: str = next_agent_key,
        reflection_key: str = "reflection_count",
        max_agent_count: int = max_agent_count,
    ) -> Literal["drug_agent", "disease_agent", "__end__"]:
        """Function to decide which agent to call next.

        Args:
            state (dict[str, Any]): The current state of the multi-agent system.
            next_agent_information (str, optional): Name of the key in the updated state. Defaults to next_agent_information.
            next_agent_key (str, optional): Name of the Key in the updated state. Defaults to next_agent_key.
            reflection_key (str, optional): Name of the key for reflection count. Defaults to "reflection_count".
            max_agent_count (int, optional): maximum agent count. Defaults to max_agent_count.

        Returns:
            Literal["drug_agent", "disease_agent", "__end__"]: The next agent to be called or end the process.
        """

        ## Get the latest next agent information
        next_agent = state.get(next_agent_information, [""])[-1]

        ## Get the count of the agents called so far
        drug_agent_count = state.get("drug_count", 0)
        disease_agent_count = state.get("disease_count", 0)

        ## Decide which agent to call next based on the next_agent information and the max_agent_count
        if (
            next_agent[next_agent_key] == "drug_agent"
            and drug_agent_count < max_agent_count
        ):
            return "drug_agent"

        elif (
            next_agent[next_agent_key] == "disease_agent"
            and disease_agent_count < max_agent_count
        ):
            return "disease_agent"

        ## Call end
        elif next_agent[next_agent_key] == "END":
            return "__end__"
        else:
            return "__end__"

    return drug_disease_condition
