## About
## This python file holds the tool definitions.
import re
import pandas as pd
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from chromadb.api.models.Collection import Collection
from typing import Callable
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI


def drug_gene_initializer(drug_df: pd.DataFrame):
    """
    An initialize for drug property function

    Args:
        drug_df (pd.DataFrame): The drug database created from DrugBank

    Returns:
        _type_: The drug property tool
    """

    @tool
    def get_drugs_related_to_gene_or_protein(gene_or_protein_list: list[str]) -> str:
        """The tool returns information about the drugs that can target
        gene or protein that are provided. This gives information such as drug name, description,
        mechanism of action, indication etc.

        Args:
            gene_or_protein_list (list[str]): This is the list of gene name / protein name for which related drugs information is required.

        Returns:
            str: The drug properties formatted as a markdown list.
        """

        if len(gene_or_protein_list) == 0 or type(gene_or_protein_list) != list:
            return "Please provide the gene_or_protein_list as a list of string or gene_or_protein_list is not provided as a list of string"

        gene_or_protein_list = [i.lower() for i in gene_or_protein_list]
        rows_list = []
        for gene in gene_or_protein_list:

            # Pattern: match only if 'gene name' is surrounded by whitespace or string boundaries, not part of a longer token
            pattern = rf"(?<![\w-]){re.escape(gene)}(?![\w-])"

            # Filter rows where 'gene' is a separate word
            sub_df = drug_df[
                drug_df["mechanism-of-action"].str.contains(
                    pattern, case=False, na=False
                )
            ]
            rows_list.append(sub_df)

        ## Remove `\n` from the description, mechanism-of-action and indication columns
        recovered_df = pd.concat(rows_list, axis=0).drop_duplicates()
        recovered_df["description"] = recovered_df["description"].str.replace(
            "\n\n", " ", regex=False
        )
        recovered_df["mechanism-of-action"] = recovered_df[
            "mechanism-of-action"
        ].str.replace("\n\n", " ", regex=False)
        recovered_df["indication"] = recovered_df["indication"].str.replace(
            "\n\n", " ", regex=False
        )

        ## Make the markdown text as a row for each drug
        if len(recovered_df) > 0:
            markdown_lines = recovered_df.apply(
                lambda row: f'- Drug name: "{row["name"]}". '
                f'Drug type: "{row["type"]}". '
                f'description: "{row["description"]}". '
                f'mechanism-of-action: "{row["mechanism-of-action"]}". '
                f'indication: "{row["indication"]}". ',
                axis=1,
            )

            # Join all lines into a single Markdown block
            markdown_text = "\n".join(markdown_lines)
            return markdown_text

        else:
            return "Did not find any informations on drugs that have know relation with the gene provided."

    return get_drugs_related_to_gene_or_protein


def drug_initializer(dataframes_dict: dict, property: list[str]):
    """
    An initialize for drug property function

    Args:
        drug_df (pd.DataFrame): The drug database created from DrugBank
        property (list[str]): A list of drug properties that are to be extracted

    Returns:
        _type_: The drug property tool
    """

    drug_df = dataframes_dict["drug_df"]
    property = property

    # @tool
    def drug_property(drug_name: str) -> str:
        """
        This tool retrives the requested property about the drug.
        In scenarious if the required information is not available, the tool returns on of the following.
        - If the drug is not available, it return "Drug not found".
        - If the property is not avaiable In the data, it returns "Property not found".
        - If the property is dosen't exist for the drug, it returns "Property does not exist for the drug".

        Parameters
        ----------
        drug_name : str
            Name of the drug.

        Returns
        -------
        str
            Property
        """

        drug_name = drug_name.lower()

        drug_row = drug_df.query("name == @drug_name")
        if len(drug_row) > 0:
            lines = drug_row.apply(
                lambda row: f'- [{row["drugbank-id"]}] MoA: {row["mechanism-of-action"]} Pharmacodynamics: {row["pharmacodynamics"]} Indication:  {row["indication"]}',
                axis=1,
            )
            ## convert the lines to a string
            lines = "\n".join(lines)
            # print(property_)
            return lines
        else:
            return "Drug not found"

    return drug_property


def drug_property_initializer(drug_df: pd.DataFrame):
    """
    An initialize for drug property function

    Args:
        drug_df (pd.DataFrame): The drug database created from DrugBank

    Returns:
        _type_: The drug property tool
    """

    @tool
    def drug_property(drug_name: str, property: str) -> str:
        """
        This tool retrives the requested property about the drug.
        In scenarious if the required information is not available, the tool returns on of the following.
        - If the drug is not available, it return "Drug not found".
        - If the property is not avaiable In the data, it returns "Property not found".
        - If the property is dosen't exist for the drug, it returns "Property does not exist for the drug".

        Parameters
        ----------
        drug_name : str
            Name of the drug.
        property : str
            The property that is required to know from the drug. This should be one of the column name of the `drug database`. It should be one of the following.
            "`drugbank-id`: Unique identifier of the drug"
            "`type`: The type of the drug. Like `biotech` or `small molecule`"
            "`description`: A short description about the drug."
            "`indication`: A short description about various health conditions that the provided drug is used as a treatment."
            "`pharmacodynamics`: The pharmacodynamic characterestics of the drug."
            "`mechanism-of-action`: A short description about how the drug works inside the human body to provide the theraputic benifits"
            "`toxicity`: The toxicity details of the drug."
            "`metabolism`: A short description on how the drug will get metabolised."
            "`absorption`: A short description about the absorption of the drug."
            "`half-life`: A short description about the half life of the drug."
            "`protein-binding`: A short description about the know interctions between the drug and proteins."
            "`route-of-elimination`: A short description about the elemination of the drug."
            "`category`: Various categories that the drug falls under."
            "`food_interaction`: The documented interactions between the drug and the foods. Here it also talks about adverse effect."
            "`drug_interaction`: The documented interactions between the drug and other drugs. Here it also talks about adverse effect."

        Returns
        -------
        str
            Property
        """
        drug_name = drug_name.lower()
        if not property in drug_df.columns:
            return "Property not found"
        drug_row = drug_df.query("name == @drug_name")
        if len(drug_row) > 0:
            property_ = drug_row[property].values
            # print(property_)
            if not pd.isnull(property_):
                return property_
            else:
                return "Property does not exist for the drug"
        else:
            return "Drug not found"

    return drug_property


def protein_property_initializer(
    protein_df: pd.DataFrame,
):
    """
    An initialize for protein property function

    Args:
        protein_df (pd.DataFrame): The protein database created from UniProtKB

    Returns:
        _type_: The protein property tool
    """

    ## For now lets not use protein name as it is matching with multiple rows
    @tool
    def protein_and_gene_property(gene_name: list[str]) -> str:
        """Retrieves biological information for a provided list of protein or gene names.

        This function returns detailed annotations including:
        - Function
        - Gene Ontology: Biological Process
        - Gene Ontology: Cellular Component
        - Gene Ontology (GO) terms
        - Involvement in disease
        - Protein families
        - Tissue specificity

        This information is useful for establishing connections between health conditions and drug responses.

        Args:
            gene_name (list[str]): This is the list of gene name / gene symbols for which properties are required.

        Returns:
            str: The gene/protein properties formatted as a markdown list.
        """

        # if protein_name == None and gene_name == None:
        if len(gene_name) == 0 or type(gene_name) != list:
            return "Please provide the gene_name as a list of string or gene_name is not provided as a list of string"

        # protein_name = [i.lower() for i in protein_name]
        gene_name = [i.lower() for i in gene_name]
        gene_rows_list = []
        for gene in gene_name:
            # Pattern: match only if 'gene name' is surrounded by whitespace or string boundaries, not part of a longer token
            pattern = rf"(?<![\w-]){re.escape(gene)}(?![\w-])"
            # Filter rows where 'gene' is a separate word
            gene_rows_list.append(
                protein_df[
                    protein_df["Gene Names"].str.contains(pattern, case=False, na=False)
                ]
            )
        recovered_df = pd.concat(gene_rows_list, axis=0)

        if len(recovered_df) > 0:
            markdown_lines = recovered_df.apply(
                lambda row: f'- Protein name: "{row["Protein names"]}". '
                f'Gene name: "{row["Gene Names"]}". '
                f'Function: "{row["Function [CC]"]}". '
                f'Pathway: "{row["Pathway"]}". '
                f'Gene Ontology (biological process): "{row["Gene Ontology (biological process)"]}". '
                f'Gene Ontology (cellular component): "{row["Gene Ontology (cellular component)"]}". '
                f'Gene Ontology (GO): "{row["Gene Ontology (GO)"]}". '
                f'Involvement in disease: "{row["Involvement in disease"]}". '
                f'Protein families: "{row["Protein families"]}". '
                f'Tissue specificity: "{row["Tissue specificity"]}"',
                axis=1,
            )

            # Join all lines into a single Markdown block
            markdown_text = "\n".join(markdown_lines)
            return markdown_text

        else:
            return "Did not find any informations for the provided gene list"

    return protein_and_gene_property


def disease_property_initializer(disease_df: pd.DataFrame):
    """
    An initialize for disease property function

    Args:
        disease_df (pd.DataFrame): The disease database created from KEGG_medicus

    Returns:
        _type_: The disease property tool
    """

    @tool
    def disease_property(disease_name: list[str]) -> str:
        """Retrieves biological information for a provided list of disease name.

        This function returns detailed annotations including:
        - Description
        - Category
        - GENE
        - Drugs
        - Name of the pathogen caused the disease
        - Sub group
        - Super group

        This information is useful for establishing connections between health conditions and drug responses.

        Args:
            disease_name (list[str]): This is the list of disease name.

        Returns:
            str: The properties formatted as a markdown list.
        """

        if len(disease_name) == 0 or type(disease_name) != list:
            return "Please provide the disease_name as a list of string or disease_name is not provided as a list of string"

        # protein_name = [i.lower() for i in protein_name]
        disease_name = [i.lower() for i in disease_name]
        disease_rows_list = []
        for disease in disease_name:
            # Pattern: match only if 'disease name' is surrounded by whitespace or string boundaries, not part of a longer token
            pattern = rf"(?<![\w-]){re.escape(disease)}(?![\w-])"
            # Filter rows where 'disease' is a separate word
            disease_rows_list.append(
                disease_df[
                    disease_df["Name"].str.contains(pattern, case=False, na=False)
                ]
            )
        recovered_df = pd.concat(disease_rows_list, axis=0)

        if len(recovered_df) > 0:
            markdown_lines = recovered_df.apply(
                lambda row: f'- Disease name: "{row["Name"]}". '
                f'Description: "{row["DESCRIPTION"]}". '
                f'Category: "{row["Category"]}". '
                f'GENES: "{row["GENE"]}". '
                f'DRUGS: "{row["DRUG"]}". '
                f'Name of the pathogen caused the disease: "{row["PATHOGEN"]}". '
                f'SUBGROUP: "{row["SUBGROUP"]}". '
                f'SUPERGRP: "{row["SUPERGRP"]}". ',
                axis=1,
            )

            # Join all lines into a single Markdown block
            markdown_text = "\n".join(markdown_lines)
            return markdown_text

        else:
            return "Did not find any informations for the provided disease list"

    return disease_property


"""

    Args:
        disease_df (pd.DataFrame): The disease database created from MedlinePlus
        gene_df (pd.DataFrame): The gene database created from MedlinePlus
        dis_gene_map_df (pd.DataFrame): The map between disease and genes
        gene_pathway_map_df (pd.DataFrame): The map between genes and pathways
        pathway_df (pd.DataFrame): The dataframe with pathway definition
        summarize_LLM (ChatGoogleGenerativeAI, optional): The LLM used for summarization. This is currently not implemented. Defaults to None.
        summarizer_sysmsg (list[tuple], optional): System message for summarization. Defaults to None.

    Returns:
        function: The medlineplus disease property tool
    """


def medlineplus_dis_property_initializer(
    dataframes_dict: dict,
    summarize_LLM: ChatGoogleGenerativeAI = None,
    summarizer_sysmsg: list[tuple] = None,
) -> Callable:
    """An initialize for medlineplus disease property function initialization

    Args:
        dataframes_dict (dict): dictionary of all the dataframes required for the function
        summarize_LLM (ChatGoogleGenerativeAI, optional): summarization LLM. Defaults to None.
        summarizer_sysmsg (list[tuple], optional): System message for summarization. Defaults to None.

    Returns:
        Callable: disease property tool
    """

    # Initialize dataframes
    disease_df = dataframes_dict["medlineplus_disease"]
    gene_df = dataframes_dict["medlineplus_gene"]
    dis_gene_map_df = dataframes_dict["medlineplus_gene_disease_mapping"]
    gene_pathway_map_df = dataframes_dict["medlineplus_gene_pathway_mapping"]
    pathway_df = dataframes_dict["senor_pathway_description"]
    disease_treatments_relation = dataframes_dict["disease_treatments_relation"]
    drug_bank_df = dataframes_dict["drug_df"]

    ## Convert the disease name to lower so it works with the searching without any issue
    disease_df["name"] = disease_df["name"].apply(lambda x: x.strip().lower())
    dis_gene_map_df["disease_name"] = dis_gene_map_df["disease_name"].apply(
        lambda x: x.strip().lower()
    )
    disease_treatments_relation["disease_name"] = disease_treatments_relation[
        "disease_name"
    ].apply(lambda x: x.strip().lower())

    ## Just strip the names to remove accidental witespace
    dis_gene_map_df["gene_name"] = dis_gene_map_df["gene_name"].apply(
        lambda x: x.strip()
    )
    gene_df["gene_name"] = gene_df["gene_name"].apply(lambda x: x.strip())
    gene_pathway_map_df["gene_name"] = gene_pathway_map_df["gene_name"].apply(
        lambda x: x.strip()
    )
    gene_pathway_map_df["pathways"] = gene_pathway_map_df["pathways"].apply(
        lambda x: x.strip()
    )
    pathway_df["Gene Set Key"] = pathway_df["Gene Set Key"].apply(lambda x: x.strip())

    def disease_property(disease_name: list[str], disease_id: list[str] = None) -> str:
        """Retrieves biological information for a provided list of disease name.

        This function returns detailed annotations including:
        - Cause of the disease
        - Proteins involved in the disease
        - What is the normal function of those proteins
        - Inheritance property of the disease

        Args:
            disease_name (list[str]): This is the list of disease name.
            disease_id (list[str], optional): This is the list of disease IDs. Defaults to None.

        Returns:
            str: The properties formatted as a markdown text.
        """

        if len(disease_name) == 0 or type(disease_name) != list:
            return "Please provide the disease_name as a list of string or disease_name is not provided as a list of string"

        disease_name = [i.lower() for i in disease_name]
        disease_rows_list = []

        ## Filter the disease dataframe based on the disease name provided
        ## TODO enable disease search with the synonym names also
        for disease in disease_name:
            # Pattern: match only if 'disease name' is surrounded by whitespace or string boundaries, not part of a longer token
            pattern = rf"(?<![\w-]){re.escape(disease)}(?![\w-])"
            # Filter rows where 'disease' is a separate word
            disease_rows_list.append(
                disease_df[
                    disease_df["name"].str.contains(pattern, case=False, na=False)
                ]
            )
        recovered_df = pd.concat(disease_rows_list, axis=0)

        ## Make a dictionary of key disease_name and value disease_id
        if disease_id is not None:
            disease_dict = dict(zip(disease_name, disease_id))
            recovered_df["Disease_ID"] = recovered_df["name"].map(disease_dict)

        if len(recovered_df) > 0:
            ## TODO: summarizer is not implimented correctly. the summarized result is not looking good
            if summarize_LLM != None:
                desx = recovered_df["description"].values[0]
                history_msg = summarizer_sysmsg + [HumanMessage(content=desx)]

                op = summarize_LLM.invoke(history_msg)
                if disease_id is not None:
                    markdown_lines = recovered_df.apply(
                        lambda row: f'- Disease name: "{row["name"]}". '
                        f'- Disease ID: "{row["Disease_ID"]}". '
                        f'\n - Description: "{row["description"]}". '
                        f'\n - Cause: "{row["causes"]}". '
                        f'\n - Inheritance: "{row["inheritance"]}". ',
                        axis=1,
                    )
                else:
                    markdown_lines = recovered_df.apply(
                        lambda row: f'- Disease name: "{row["name"]}". '
                        f'\n - Description: "{row["description"]}". '
                        f'\n - Cause: "{row["causes"]}". '
                        f'\n - Inheritance: "{row["inheritance"]}". ',
                        axis=1,
                    )
            else:

                if disease_id is not None:

                    markdown_lines = recovered_df.apply(
                        lambda row: f'- Disease name: "{row["name"]}". '
                        f'- database ID: ["{row["Disease_ID"]}]". '
                        # f'\n - Description: "{row["description"]}". '
                        f'\n - Cause: "{row["causes"]}". '
                        f'\n - Inheritance: "{row["inheritance"]}". ',
                        axis=1,
                    )

                else:

                    markdown_lines = recovered_df.apply(
                        lambda row: f'- Disease name: "{row["name"]}". '
                        # f'\n - Description: "{row["description"]}". '
                        f'\n - Cause: "{row["causes"]}". '
                        f'\n - Inheritance: "{row["inheritance"]}". ',
                        axis=1,
                    )

            # Join all lines into a single Markdown block
            markdown_disease = "\n".join(markdown_lines)

            ## Get the treatment details for the disease from disease_treatments_relation and drug_bank_df
            drug_bank_id = disease_treatments_relation[
                disease_treatments_relation["disease_name"].isin(disease_name)
            ]["drugbank-id"].values

            if len(drug_bank_id) > 0:
                drug_sub_df = drug_bank_df[
                    drug_bank_df["drugbank-id"].isin(drug_bank_id)
                ]
                markdown_lines = drug_sub_df.apply(
                    lambda row: f'- Drug name: "{row["name"]}". '
                    f'indication: "{row["indication"]}". '
                    f'mechanism-of-action: "{row["mechanism-of-action"]}". ',
                    axis=1,
                )
                # Join all lines into a single Markdown block
                markdown_treatment = "\n".join(markdown_lines)
            final_markdown = ""
            ## get the list of disease name and from it get the list of gene names
            ## drop duplicates
            gene_names = list(
                set(
                    list(
                        dis_gene_map_df[
                            dis_gene_map_df["disease_name"].isin(disease_name)
                        ]["gene_name"].values
                    )
                )
            )

            if len(gene_names) > 0:
                gene_sub_df = gene_df[gene_df["gene_name"].isin(gene_names)]
                ## Remove \n in normal function of gene_sub_df
                gene_sub_df["normal_function"] = gene_sub_df[
                    "normal_function"
                ].str.replace("\n", " ")
                markdown_lines = gene_sub_df.apply(
                    lambda row: f'- Gene name: "{row["gene_name"]}". '
                    f'The Normal Function of the Gene: "{row["normal_function"]}". ',
                    axis=1,
                )
                # Join all lines into a single Markdown block
                markdown_gene = "\n".join(markdown_lines)

                ## Pathway information
                pathway_list = [
                    path_.split(",")
                    for path_ in list(
                        gene_pathway_map_df[
                            gene_pathway_map_df["gene_name"].isin(gene_names)
                        ]["pathways"].values
                    )
                ]
                pathway_names = []
                for path_ in pathway_list:
                    path_ = [i.strip() for i in path_]
                    for i in path_:
                        pathway_names.append(i)
                pathway_names = set(pathway_names)
                pathway_desc_sub_df = pathway_df[
                    pathway_df["Gene Set Key"].isin(list(pathway_names))
                ]

                markdown_lines = pathway_desc_sub_df.apply(
                    lambda row: f'- Pathway name: "{row["pathway_name"]}". '
                    f'Pathway function: "{row["Description"]}". ',
                    axis=1,
                )
                # Join all lines into a single Markdown block
                markdown_pathway = "\n".join(markdown_lines)
            #     if len(pathway_list) > 0:
            #         return (
            #             "## Explaination about the disease \n"
            #             + markdown_disease
            #             + "\n## Explaination about the genes that are involved in the disease \n"
            #             + markdown_gene
            #             + "\n## Explaination about the pathways the genes are involved in \n"
            #             + markdown_pathway
            #         )
            #     else:  # If no pathway info found in signor
            #         return (
            #             "## Explaination about the disease \n"
            #             + markdown_disease
            #             + "\n## Explaination about the genes that are involved in the disease \n"
            #             + markdown_gene
            #         )

            # else:

            #     return markdown_disease

            ## Final markdown update
            ## Add disease info
            final_markdown += (
                "## Explaination about the disease \n" + markdown_disease + "\n"
            )
            ## Add gene info
            if len(gene_names) > 0:
                final_markdown += (
                    "\n## Explaination about the genes that are involved in the disease \n"
                    + markdown_gene
                )
            ## Add pathway info
            if len(pathway_list) > 0:
                final_markdown += (
                    "\n## Explaination about the pathways the genes are involved in \n"
                    + markdown_pathway
                )
            ## Add treatment info
            if len(drug_bank_id) > 0:
                final_markdown += (
                    "\n## Known treatments for the disease \n" + markdown_treatment
                )

            return final_markdown

        else:
            return "Did not find any informations for the provided disease list"

    return disease_property


def drug_retrieve_and_format_queries_agent(
    query_filter_sys_msg: list[tuple[str]],
    dataframes_dict: dict[str : pd.DataFrame],
    filter_dataframe_by_selected_ids,
):
    """Retrieve drug informations based on the queries using Bio NLP model.

    Args:
        query_filter_sys_msg (_type_): The sys msg for LLM that acts as a query and doc matcher/filter.
        dataframes_dict (dict): A dictionary of drug databases from which the information will be populated
        filter_dataframe_by_selected_ids (_type_): A function that is used to filter query database
    """

    def drug_retrieve_and_format_queries(
        queries: list,
        RAG_collection: dict[str:Collection],
        llm: ChatGoogleGenerativeAI,
        n_results: int = 5,
    ) -> tuple[pd.DataFrame, str]:
        """For each query, retrieve relevant docs from both MoA_collection and PD_collection,
        build a dataframe with query, RAG system, ids, documents, metadatas,
        and generate a markdown summary.

        Args:
            queries (list): List of queries that are generated
            RAG_collection (dict): Dictionary of RAG collections
            llm (ChatGoogleGenerativeAI): LLM to select relevant docs for the queries.
            n_results (int, optional): number of docs to retrieve in the initial phase. . Defaults to 5.

        Returns:
            tuple[pd.DataFrame, str]: Filtered doc dataframe and markdown generated from the docs.
        """

        results_list = []
        markdown_lines = []
        selected_db_ids = {"query": [], "selected_DB_ids": []}
        all_db_id_selected = []

        for count, query in enumerate(queries):
            db_id_list = []
            docs_list = []

            ## Iterate through both the MoA nd PD collections
            for rag_name, collection in RAG_collection.items():

                res = collection.query(query_texts=[query], n_results=n_results)
                ids = res.get("ids", [[]])[0]
                docs = res.get("documents", [[]])[0]
                docs_list.extend(docs)
                metadatas = res.get("metadatas", [[]])[0]
                ## For the new chroma db version compatibility
                if "drugbank-id" in metadatas[0].keys():
                    drugbank_ids = [meta.get("drugbank-id", "") for meta in metadatas]
                ## For the older chroma db version compatibility
                else:
                    drugbank_ids = [meta.get("drugbank_id", "") for meta in metadatas]
                db_id_list.extend(drugbank_ids)

                for i, (id_, doc, meta) in enumerate(zip(ids, docs, metadatas)):

                    results_list.append(
                        {
                            "query": query,
                            "RAG": rag_name,
                            "id": id_,
                            "document": doc,
                            # "metadata": meta,
                            "drugbank-id": (
                                drugbank_ids[i] if i < len(drugbank_ids) else ""
                            ),
                        }
                    )

            markdown_lines.append([])
            markdown_lines[count].append(f"## Query: {query}")
            all_db_id_selected.extend(db_id_list)
            for id_, doc in zip(db_id_list, docs_list):
                markdown_lines[count].append(f"- [{id_}] {doc}")

        markdown_query_selection = "\n".join(
            [line for sublist in markdown_lines for line in sublist]
        )
        history_msg = query_filter_sys_msg + [
            HumanMessage(content=markdown_query_selection)
        ]
        op = llm.invoke(history_msg)

        for op_item in op.query_list:
            selected_db_ids["query"].append(op_item.query)
            selected_db_ids["selected_DB_ids"].append(op_item.db_ids)

        df = pd.DataFrame(results_list)
        # print("before filtering", df.shape)
        df = filter_dataframe_by_selected_ids(df, selected_db_ids)
        # * Since all the query are for the same disease, we can drop repeatind drug entries.
        filtered_df = df[["drugbank-id"]].drop_duplicates(inplace=False)
        drug_bank_filtered = dataframes_dict["drug_df"][
            dataframes_dict["drug_df"]["drugbank-id"].isin(filtered_df["drugbank-id"])
        ]

        markdown_text = ""
        ## Append all the queries first
        queries_l = df["query"].unique()
        markdown_text = "# Queries\n"
        for i, query in enumerate(queries_l):
            markdown_text += f"{i+1}. {query}\n"
        lines = drug_bank_filtered.apply(
            lambda row: f'- database ID: [{row["drugbank-id"]}] MoA: {row["mechanism-of-action"]} Pharmacodynamics: {row["pharmacodynamics"]} Indication:  {row["indication"]}',
            axis=1,
        )
        ## Append all the selected
        markdown_text += "\n# Answers\n"
        markdown_text += "\n".join(lines) + "\n"
        ## Keep only unique
        all_db_id_selected = list(set(all_db_id_selected))
        return (
            filtered_df,
            markdown_text,
            all_db_id_selected,
            selected_db_ids,
            markdown_query_selection,
        )

    return drug_retrieve_and_format_queries


def disease_retrieve_and_format_queries_agent(
    query_filter_sys_msg: list[tuple[str]],
    dataframes_dict: dict[str : pd.DataFrame],
    filter_dataframe_by_selected_ids,
    disease_property,
):
    """Retrieve disease informations based on the queries using Bio NLP model.

    Args:
        query_filter_sys_msg (_type_): The sys msg for LLM that acts as a query and doc matcher/filter.
        This will only return relevant docs for the query.
        dataframes_dict (dict): A dictionary of disease databases from which the information will be populated
        filter_dataframe_by_selected_ids (_type_): A function that is used to filter query database
        disease_property : Function that takes list of disease names and returns a string
    """

    def disease_retrieve_and_format_queries(
        queries: list[str],
        RAG_collection: dict[str:Collection],
        llm: ChatGoogleGenerativeAI,
        n_results: int = 5,
    ) -> tuple[pd.DataFrame, str]:
        """For each query, retrieve relevant docs from both MoA_collection and PD_collection,
        build a dataframe with query, RAG system, ids, documents, metadatas,
        and generate a markdown summary.

        Args:
            queries (list): List of queries that are generated
            RAG_collection (dict): Dictionary of RAG collections
            llm (ChatGoogleGenerativeAI): LLM to select relevant docs for the queries.
            n_results (int, optional): number of docs to retrieve in the initial phase. . Defaults to 5.

        Returns:
            tuple[pd.DataFrame, str]: Filtered doc dataframe and markdown generated from the docs.
        """
        # TODO: VALIDATE: Check if the initial retrieved drugs are correct. Even before query LLM filter.
        results_list = []
        markdown_lines = []
        selected_db_ids = {"query": [], "selected_DB_ids": []}
        all_db_id_selected = []
        for count, query in enumerate(queries):
            db_id_list = []
            docs_list = []
            ## Iterate through both the MoA nd PD collections
            for rag_name, collection in RAG_collection.items():
                res = collection.query(query_texts=[query], n_results=n_results)
                ids = res.get("ids", [[]])[0]

                docs = res.get("documents", [[]])[0]
                docs_list.extend(docs)
                metadatas = res.get("metadatas", [[]])[0]
                Database_IDs = [meta.get("Database_ID", "") for meta in metadatas]
                db_id_list.extend(Database_IDs)
                for i, (id_, doc, meta) in enumerate(zip(ids, docs, metadatas)):
                    results_list.append(
                        {
                            "query": query,
                            "RAG": rag_name,
                            "id": id_,
                            "document": doc,
                            # "metadata": meta,
                            "Database_ID": (
                                Database_IDs[i] if i < len(Database_IDs) else ""
                            ),
                        }
                    )
            markdown_lines.append([])
            markdown_lines[count].append(f"## Query: {query}")
            all_db_id_selected.extend(db_id_list)
            for id_, doc in zip(db_id_list, docs_list):
                markdown_lines[count].append(f"- [Database ID: {id_}] {doc}")
        markdown_query_selection = "\n".join(
            [line for sublist in markdown_lines for line in sublist]
        )

        ## TODO: VAL save `db_id_list``

        history_msg = query_filter_sys_msg + [
            HumanMessage(content=markdown_query_selection)
        ]
        op = llm.invoke(history_msg)
        for op_item in op.query_list:
            selected_db_ids["query"].append(op_item.query)
            selected_db_ids["selected_DB_ids"].append(op_item.db_ids)

        ## ## TODO go from selected_db_ids to name of the health condition
        id_lists = selected_db_ids["selected_DB_ids"][0]
        for id_list in selected_db_ids["selected_DB_ids"][1:]:
            id_lists.extend(id_list)
        id_lists = list(set(id_lists))  # Remove duplicates

        med_df = dataframes_dict["medlineplus_disease"]
        dis_subset_df = med_df[med_df["Database_ID"].isin(id_lists)]
        if len(dis_subset_df) > 0:
            selected_dis_names = list(set(dis_subset_df["name"].str.lower()))
            id_lists = list(set(dis_subset_df["Database_ID"]))

            markdown_disease = disease_property(
                disease_name=selected_dis_names, disease_id=id_lists
            )
        else:
            markdown_disease = "No relevant disease information found."
        ## TODO: INSPECTED TILL HERE
        ## formulate the query and answer setup
        markdown_text = "# Queries\n"
        for i, query in enumerate(selected_db_ids["query"]):
            markdown_text += f"{i}. {query}\n"

        ## Append all the selected
        markdown_text += "\n# Answers\n"
        ## Append all the selected
        markdown_text += markdown_disease
        all_db_id_selected = list(set(all_db_id_selected))
        return (
            markdown_text,
            all_db_id_selected,
            selected_db_ids,
            markdown_query_selection,
        )

    return disease_retrieve_and_format_queries
