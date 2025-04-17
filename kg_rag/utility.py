import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Memory
import json
import openai
import os
import sys
from tenacity import retry, stop_after_attempt, wait_random_exponential
import time
from dotenv import load_dotenv, find_dotenv
import torch
from langchain import HuggingFacePipeline
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, TextStreamer, GPTQConfig
from kg_rag.config_loader import *
import ast
import requests

from  kg_rag.gpt_request import *
import logging

# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='kg_rag/test/retriever.log',   # Tên file log bạn muốn lưu
    filemode='w'                # 'w' để ghi đè mỗi lần chạy, hoặc 'a' để ghi thêm
)

memory = Memory("cachegpt", verbose=0)

# Config openai library
config_file = config_data['GPT_CONFIG_FILE']
load_dotenv(config_file)
api_key = os.environ.get('API_KEY')
api_version = os.environ.get('API_VERSION')
resource_endpoint = os.environ.get('RESOURCE_ENDPOINT')
openai.api_type = config_data['GPT_API_TYPE']
openai.api_key = api_key
if resource_endpoint:
    openai.api_base = resource_endpoint
if api_version:
    openai.api_version = api_version

torch.cuda.empty_cache()
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def get_spoke_api_resp(base_uri, end_point, params=None):
    uri = base_uri + end_point
    if params:
        return requests.get(uri, params=params)
    else:
        return requests.get(uri)

@retry(wait=wait_random_exponential(min=10, max=30), stop=stop_after_attempt(5))
def get_context_using_spoke_api(node_type, node_value):
    type_end_point = "/api/v1/types"
    result = get_spoke_api_resp(config_data['BASE_URI'], type_end_point)
    data_spoke_types = result.json()
    node_types = list(data_spoke_types["nodes"].keys())
    edge_types = list(data_spoke_types["edges"].keys())
    node_types_to_remove = ["DatabaseTimestamp", "Version"]
    filtered_node_types = [node_type for node_type in node_types if node_type not in node_types_to_remove]
    api_params = {
        'node_filters' : filtered_node_types,
        'edge_filters': edge_types,
        'cutoff_Compound_max_phase': config_data['cutoff_Compound_max_phase'],
        'cutoff_Protein_source': config_data['cutoff_Protein_source'],
        'cutoff_DaG_diseases_sources': config_data['cutoff_DaG_diseases_sources'],
        'cutoff_DaG_textmining': config_data['cutoff_DaG_textmining'],
        'cutoff_CtD_phase': config_data['cutoff_CtD_phase'],
        'cutoff_PiP_confidence': config_data['cutoff_PiP_confidence'],
        'cutoff_ACTeG_level': config_data['cutoff_ACTeG_level'],
        'cutoff_DpL_average_prevalence': config_data['cutoff_DpL_average_prevalence'],
        'depth' : config_data['depth']
    }
    # node_type = "Disease"
    attribute = "name"
    nbr_end_point = "/api/v1/neighborhood/{}/{}/{}".format(node_type, attribute, node_value)
    result = get_spoke_api_resp(config_data['BASE_URI'], nbr_end_point, params=api_params)
    node_context = result.json()
    nbr_nodes = []
    nbr_edges = []
    for item in node_context:
        if "_" not in item["data"]["neo4j_type"]:
            try:
                if item["data"]["neo4j_type"] == "Protein":
                    nbr_nodes.append((item["data"]["neo4j_type"], item["data"]["id"], item["data"]["properties"]["description"]))
                else:
                    nbr_nodes.append((item["data"]["neo4j_type"], item["data"]["id"], item["data"]["properties"]["name"]))
            except:
                nbr_nodes.append((item["data"]["neo4j_type"], item["data"]["id"], item["data"]["properties"]["identifier"]))
        elif "_" in item["data"]["neo4j_type"]:
            try:
                provenance = ", ".join(item["data"]["properties"]["sources"])
            except:
                try:
                    provenance = item["data"]["properties"]["source"]
                    if isinstance(provenance, list):
                        provenance = ", ".join(provenance)                    
                except:
                    try:                    
                        preprint_list = ast.literal_eval(item["data"]["properties"]["preprint_list"])
                        if len(preprint_list) > 0:                                                    
                            provenance = ", ".join(preprint_list)
                        else:
                            pmid_list = ast.literal_eval(item["data"]["properties"]["pmid_list"])
                            pmid_list = map(lambda x:"pubmedId:"+x, pmid_list)
                            if len(pmid_list) > 0:
                                provenance = ", ".join(pmid_list)
                            else:
                                provenance = "Based on data from Institute For Systems Biology (ISB)"
                    except:                                
                        provenance = "SPOKE-KG"     
            try:
                evidence = item["data"]["properties"]
            except:
                evidence = None
            nbr_edges.append((item["data"]["source"], item["data"]["neo4j_type"], item["data"]["target"], provenance, evidence))
    nbr_nodes_df = pd.DataFrame(nbr_nodes, columns=["node_type", "node_id", "node_name"])
    nbr_edges_df = pd.DataFrame(nbr_edges, columns=["source", "edge_type", "target", "provenance", "evidence"])
    merge_1 = pd.merge(nbr_edges_df, nbr_nodes_df, left_on="source", right_on="node_id").drop("node_id", axis=1)
    merge_1.loc[:,"node_name"] = merge_1.node_type + " " + merge_1.node_name
    merge_1.drop(["source", "node_type"], axis=1, inplace=True)
    merge_1 = merge_1.rename(columns={"node_name":"source"})
    merge_2 = pd.merge(merge_1, nbr_nodes_df, left_on="target", right_on="node_id").drop("node_id", axis=1)
    merge_2.loc[:,"node_name"] = merge_2.node_type + " " + merge_2.node_name
    merge_2.drop(["target", "node_type"], axis=1, inplace=True)
    merge_2 = merge_2.rename(columns={"node_name":"target"})
    merge_2 = merge_2[["source", "edge_type", "target", "provenance", "evidence"]]
    merge_2.loc[:, "predicate"] = merge_2.edge_type.apply(lambda x:x.split("_")[0])
    merge_2.loc[:, "context"] =  merge_2.source + " " + merge_2.predicate.str.lower() + " " + merge_2.target + " and Provenance of this association is " + merge_2.provenance + "."
    context = merge_2.context.str.cat(sep=' ')
    context += node_value + " has a " + node_context[0]["data"]["properties"]["source"] + " identifier of " + node_context[0]["data"]["properties"]["identifier"] + " and Provenance of this is from " + node_context[0]["data"]["properties"]["source"] + "."
    return context, merge_2
        
#         if edge_evidence:
#             merge_2.loc[:, "context"] =  merge_2.source + " " + merge_2.predicate.str.lower() + " " + merge_2.target + " and Provenance of this association is " + merge_2.provenance + " and attributes associated with this association is in the following JSON format:\n " + merge_2.evidence.astype('str') + "\n\n"
#         else:
#             merge_2.loc[:, "context"] =  merge_2.source + " " + merge_2.predicate.str.lower() + " " + merge_2.target + " and Provenance of this association is " + merge_2.provenance + ". "
#         context = merge_2.context.str.cat(sep=' ')
#         context += node_value + " has a " + node_context[0]["data"]["properties"]["source"] + " identifier of " + node_context[0]["data"]["properties"]["identifier"] + " and Provenance of this is from " + node_context[0]["data"]["properties"]["source"] + "."
#     return context



def get_context_using_spoke_api_disease(node_value):    
    type_end_point = "/api/v1/types"
    result = get_spoke_api_resp(config_data['BASE_URI'], type_end_point)
    data_spoke_types = result.json()
    node_types = list(data_spoke_types["nodes"].keys())
    edge_types = list(data_spoke_types["edges"].keys())
    node_types_to_remove = ["DatabaseTimestamp", "Version"]
    filtered_node_types = [node_type for node_type in node_types if node_type not in node_types_to_remove]
    api_params = {
        'node_filters' : filtered_node_types,
        'edge_filters': edge_types,
        'cutoff_Compound_max_phase': config_data['cutoff_Compound_max_phase'],
        'cutoff_Protein_source': config_data['cutoff_Protein_source'],
        'cutoff_DaG_diseases_sources': config_data['cutoff_DaG_diseases_sources'],
        'cutoff_DaG_textmining': config_data['cutoff_DaG_textmining'],
        'cutoff_CtD_phase': config_data['cutoff_CtD_phase'],
        'cutoff_PiP_confidence': config_data['cutoff_PiP_confidence'],
        'cutoff_ACTeG_level': config_data['cutoff_ACTeG_level'],
        'cutoff_DpL_average_prevalence': config_data['cutoff_DpL_average_prevalence'],
        'depth' : config_data['depth']
    }
    node_type = "Disease"
    attribute = "name"
    nbr_end_point = "/api/v1/neighborhood/{}/{}/{}".format(node_type, attribute, node_value)
    result = get_spoke_api_resp(config_data['BASE_URI'], nbr_end_point, params=api_params)
    node_context = result.json()
    nbr_nodes = []
    nbr_edges = []
    for item in node_context:
        if "_" not in item["data"]["neo4j_type"]:
            try:
                if item["data"]["neo4j_type"] == "Protein":
                    nbr_nodes.append((item["data"]["neo4j_type"], item["data"]["id"], item["data"]["properties"]["description"]))
                else:
                    nbr_nodes.append((item["data"]["neo4j_type"], item["data"]["id"], item["data"]["properties"]["name"]))
            except:
                nbr_nodes.append((item["data"]["neo4j_type"], item["data"]["id"], item["data"]["properties"]["identifier"]))
        elif "_" in item["data"]["neo4j_type"]:
            try:
                provenance = ", ".join(item["data"]["properties"]["sources"])
            except:
                try:
                    provenance = item["data"]["properties"]["source"]
                    if isinstance(provenance, list):
                        provenance = ", ".join(provenance)                    
                except:
                    try:                    
                        preprint_list = ast.literal_eval(item["data"]["properties"]["preprint_list"])
                        if len(preprint_list) > 0:                                                    
                            provenance = ", ".join(preprint_list)
                        else:
                            pmid_list = ast.literal_eval(item["data"]["properties"]["pmid_list"])
                            pmid_list = map(lambda x:"pubmedId:"+x, pmid_list)
                            if len(pmid_list) > 0:
                                provenance = ", ".join(pmid_list)
                            else:
                                provenance = "Based on data from Institute For Systems Biology (ISB)"
                    except:                                
                        provenance = "SPOKE-KG"     
            try:
                evidence = item["data"]["properties"]
            except:
                evidence = None
            nbr_edges.append((item["data"]["source"], item["data"]["neo4j_type"], item["data"]["target"], provenance, evidence))
    nbr_nodes_df = pd.DataFrame(nbr_nodes, columns=["node_type", "node_id", "node_name"])
    nbr_edges_df = pd.DataFrame(nbr_edges, columns=["source", "edge_type", "target", "provenance", "evidence"])
    merge_1 = pd.merge(nbr_edges_df, nbr_nodes_df, left_on="source", right_on="node_id").drop("node_id", axis=1)
    merge_1.loc[:,"node_name"] = merge_1.node_type + " " + merge_1.node_name
    merge_1.drop(["source", "node_type"], axis=1, inplace=True)
    merge_1 = merge_1.rename(columns={"node_name":"source"})
    merge_2 = pd.merge(merge_1, nbr_nodes_df, left_on="target", right_on="node_id").drop("node_id", axis=1)
    merge_2.loc[:,"node_name"] = merge_2.node_type + " " + merge_2.node_name
    merge_2.drop(["target", "node_type"], axis=1, inplace=True)
    merge_2 = merge_2.rename(columns={"node_name":"target"})
    merge_2 = merge_2[["source", "edge_type", "target", "provenance", "evidence"]]
    merge_2.loc[:, "predicate"] = merge_2.edge_type.apply(lambda x:x.split("_")[0])
    merge_2.loc[:, "context"] =  merge_2.source + " " + merge_2.predicate.str.lower() + " " + merge_2.target + " and Provenance of this association is " + merge_2.provenance + "."
    context = merge_2.context.str.cat(sep=' ')
    context += node_value + " has a " + node_context[0]["data"]["properties"]["source"] + " identifier of " + node_context[0]["data"]["properties"]["identifier"] + " and Provenance of this is from " + node_context[0]["data"]["properties"]["source"] + "."
    return context, merge_2

def get_context_using_spoke_api_gene(node_value):
    type_end_point = "/api/v1/types"
    result = get_spoke_api_resp(config_data['BASE_URI'], type_end_point)
    data_spoke_types = result.json()
    node_types = list(data_spoke_types["nodes"].keys())
    edge_types = list(data_spoke_types["edges"].keys())
    node_types_to_remove = ["DatabaseTimestamp", "Version"]
    filtered_node_types = [nt for nt in node_types if nt not in node_types_to_remove]

    api_params = {
        'node_filters': filtered_node_types,
        'edge_filters': edge_types,
        'cutoff_Compound_max_phase': config_data['cutoff_Compound_max_phase'],
        'cutoff_Protein_source': config_data['cutoff_Protein_source'],
        'cutoff_DaG_diseases_sources': config_data['cutoff_DaG_diseases_sources'],
        'cutoff_DaG_textmining': config_data['cutoff_DaG_textmining'],
        'cutoff_CtD_phase': config_data['cutoff_CtD_phase'],
        'cutoff_PiP_confidence': config_data['cutoff_PiP_confidence'],
        'cutoff_ACTeG_level': config_data['cutoff_ACTeG_level'],
        'cutoff_DpL_average_prevalence': config_data['cutoff_DpL_average_prevalence'],
        'depth': config_data['depth']
    }
    node_type = "Gene"
    attribute = "name"
    nbr_end_point = f"/api/v1/neighborhood/{node_type}/{attribute}/{node_value}"
    result = get_spoke_api_resp(config_data['BASE_URI'], nbr_end_point, params=api_params)
    node_context = result.json()

    nbr_nodes = []
    nbr_edges = []

    for item in node_context:
        if "_" not in item["data"]["neo4j_type"]:
            try:
                if item["data"]["neo4j_type"] == "Protein":
                    nbr_nodes.append((
                        item["data"]["neo4j_type"],
                        item["data"]["id"],
                        item["data"]["properties"]["description"]
                    ))
                else:
                    nbr_nodes.append((
                        item["data"]["neo4j_type"],
                        item["data"]["id"],
                        item["data"]["properties"]["name"]
                    ))
            except:
                nbr_nodes.append((
                    item["data"]["neo4j_type"],
                    item["data"]["id"],
                    item["data"]["properties"]["identifier"]
                ))
        elif "_" in item["data"]["neo4j_type"]:
            try:
                provenance = ", ".join(item["data"]["properties"]["sources"])
            except:
                try:
                    provenance = item["data"]["properties"]["source"]
                    if isinstance(provenance, list):
                        provenance = ", ".join(provenance)
                except:
                    try:
                        preprint_list = ast.literal_eval(item["data"]["properties"].get("preprint_list", "[]"))
                        if preprint_list:
                            provenance = ", ".join(preprint_list)
                        else:
                            pmid_list = ast.literal_eval(item["data"]["properties"].get("pmid_list", "[]"))
                            pmid_list = list(map(lambda x: "pubmedId:" + x, pmid_list))
                            provenance = ", ".join(pmid_list) if pmid_list else "Based on data from ISB"
                    except:
                        provenance = "SPOKE-KG"
            evidence = item["data"].get("properties", None)
            nbr_edges.append((
                item["data"]["source"],
                item["data"]["neo4j_type"],
                item["data"]["target"],
                provenance,
                evidence
            ))

    nbr_nodes_df = pd.DataFrame(nbr_nodes, columns=["node_type", "node_id", "node_name"])
    nbr_edges_df = pd.DataFrame(nbr_edges, columns=["source", "edge_type", "target", "provenance", "evidence"])

    merge_1 = pd.merge(nbr_edges_df, nbr_nodes_df, left_on="source", right_on="node_id").drop("node_id", axis=1)
    merge_1["node_name"] = merge_1["node_type"] + " " + merge_1["node_name"]
    merge_1.drop(["source", "node_type"], axis=1, inplace=True)
    merge_1.rename(columns={"node_name": "source"}, inplace=True)

    merge_2 = pd.merge(merge_1, nbr_nodes_df, left_on="target", right_on="node_id").drop("node_id", axis=1)
    merge_2["node_name"] = merge_2["node_type"] + " " + merge_2["node_name"]
    merge_2.drop(["target", "node_type"], axis=1, inplace=True)
    merge_2.rename(columns={"node_name": "target"}, inplace=True)

    merge_2 = merge_2[["source", "edge_type", "target", "provenance", "evidence"]]
    merge_2["predicate"] = merge_2["edge_type"].apply(lambda x: x.split("_")[0])
    merge_2["context"] = merge_2["source"] + " " + merge_2["predicate"].str.lower() + " " + merge_2["target"] + \
                         " and Provenance of this association is " + merge_2["provenance"] + "."

    # Lấy thêm thông tin riêng cho Gene
    root_node = next((item for item in node_context if item["data"].get("neo4j_root", 0) == 1), None)
    if root_node:
        props = root_node["data"]["properties"]
        gene_extra = []
        if "description" in props:
            gene_extra.append(f"{node_value} is described as: {props['description']}")
        if "ensembl" in props:
            gene_extra.append(f"Its Ensembl ID is {props['ensembl']}")
        if "chromosome" in props:
            gene_extra.append(f"It is located on chromosome {props['chromosome']}")
        if "chembl_id" in props:
            gene_extra.append(f"The ChEMBL ID is {props['chembl_id']}")
        if "identifier" in props:
            gene_extra.append(f"{node_value} has Entrez Gene ID {props['identifier']}")
        if "source" in props:
            gene_extra.append(f"Source: {props['source']}")

        extra_info = " ".join(gene_extra)
    else:
        extra_info = ""

    context = merge_2["context"].str.cat(sep=' ') + " " + extra_info

    return context, merge_2

def get_context_using_spoke_api_drugs(node_value):
    type_end_point = "/api/v1/types"
    result = get_spoke_api_resp(config_data['BASE_URI'], type_end_point)
    data_spoke_types = result.json()
    node_types = list(data_spoke_types["nodes"].keys())
    edge_types = list(data_spoke_types["edges"].keys())
    node_types_to_remove = ["DatabaseTimestamp", "Version"]
    filtered_node_types = [nt for nt in node_types if nt not in node_types_to_remove]

    api_params = {
        'node_filters': filtered_node_types,
        'edge_filters': edge_types,
        'cutoff_Compound_max_phase': config_data['cutoff_Compound_max_phase'],
        'cutoff_Protein_source': config_data['cutoff_Protein_source'],
        'cutoff_DaG_diseases_sources': config_data['cutoff_DaG_diseases_sources'],
        'cutoff_DaG_textmining': config_data['cutoff_DaG_textmining'],
        'cutoff_CtD_phase': config_data['cutoff_CtD_phase'],
        'cutoff_PiP_confidence': config_data['cutoff_PiP_confidence'],
        'cutoff_ACTeG_level': config_data['cutoff_ACTeG_level'],
        'cutoff_DpL_average_prevalence': config_data['cutoff_DpL_average_prevalence'],
        'depth': config_data['depth']
    }

    node_type = "Compound"
    attribute = "name"
    nbr_end_point = f"/api/v1/neighborhood/{node_type}/{attribute}/{node_value}"
    result = get_spoke_api_resp(config_data['BASE_URI'], nbr_end_point, params=api_params)
    node_context = result.json()

    nbr_nodes = []
    nbr_edges = []

    for item in node_context:
        if "_" not in item["data"]["neo4j_type"]:
            try:
                if item["data"]["neo4j_type"] == "Protein":
                    nbr_nodes.append((
                        item["data"]["neo4j_type"],
                        item["data"]["id"],
                        item["data"]["properties"]["description"]
                    ))
                else:
                    nbr_nodes.append((
                        item["data"]["neo4j_type"],
                        item["data"]["id"],
                        item["data"]["properties"]["name"]
                    ))
            except:
                nbr_nodes.append((
                    item["data"]["neo4j_type"],
                    item["data"]["id"],
                    item["data"]["properties"]["identifier"]
                ))
        elif "_" in item["data"]["neo4j_type"]:
            try:
                provenance = ", ".join(item["data"]["properties"]["sources"])
            except:
                try:
                    provenance = item["data"]["properties"]["source"]
                    if isinstance(provenance, list):
                        provenance = ", ".join(provenance)
                except:
                    try:
                        preprint_list = ast.literal_eval(item["data"]["properties"].get("preprint_list", "[]"))
                        if preprint_list:
                            provenance = ", ".join(preprint_list)
                        else:
                            pmid_list = ast.literal_eval(item["data"]["properties"].get("pmid_list", "[]"))
                            pmid_list = list(map(lambda x: "pubmedId:" + x, pmid_list))
                            provenance = ", ".join(pmid_list) if pmid_list else "Based on data from ISB"
                    except:
                        provenance = "SPOKE-KG"
            evidence = item["data"].get("properties", None)
            nbr_edges.append((
                item["data"]["source"],
                item["data"]["neo4j_type"],
                item["data"]["target"],
                provenance,
                evidence
            ))

    nbr_nodes_df = pd.DataFrame(nbr_nodes, columns=["node_type", "node_id", "node_name"])
    nbr_edges_df = pd.DataFrame(nbr_edges, columns=["source", "edge_type", "target", "provenance", "evidence"])

    merge_1 = pd.merge(nbr_edges_df, nbr_nodes_df, left_on="source", right_on="node_id").drop("node_id", axis=1)
    merge_1["node_name"] = merge_1["node_type"] + " " + merge_1["node_name"]
    merge_1.drop(["source", "node_type"], axis=1, inplace=True)
    merge_1.rename(columns={"node_name": "source"}, inplace=True)

    merge_2 = pd.merge(merge_1, nbr_nodes_df, left_on="target", right_on="node_id").drop("node_id", axis=1)
    merge_2["node_name"] = merge_2["node_type"] + " " + merge_2["node_name"]
    merge_2.drop(["target", "node_type"], axis=1, inplace=True)
    merge_2.rename(columns={"node_name": "target"}, inplace=True)

    merge_2 = merge_2[["source", "edge_type", "target", "provenance", "evidence"]]
    merge_2["predicate"] = merge_2["edge_type"].apply(lambda x: x.split("_")[0])
    merge_2["context"] = merge_2["source"] + " " + merge_2["predicate"].str.lower() + " " + merge_2["target"] + \
                         " and Provenance of this association is " + merge_2["provenance"] + "."

    # Thông tin riêng cho Compound (node gốc)
    root_node = next((item for item in node_context if item["data"].get("neo4j_root", 0) == 1), None)
    if root_node:
        props = root_node["data"]["properties"]
        compound_extra = []
        if "name" in props:
            compound_extra.append(f"Compound name: {props['name']}")
        if "identifier" in props:
            compound_extra.append(f"Identifier: {props['identifier']}")
        if "chembl_id" in props:
            compound_extra.append(f"ChEMBL ID: {props['chembl_id']}")
        if "standardized_smiles" in props:
            compound_extra.append("Standardized SMILES available.")
        if "synonyms" in props:
            synonyms = ", ".join(props["synonyms"])
            compound_extra.append(f"Synonyms: {synonyms}")
        if "max_phase" in props:
            compound_extra.append(f"Max clinical trial phase: {props['max_phase']}")
        if "sources" in props:
            compound_extra.append(f"Sources: {', '.join(props['sources'])}")
        extra_info = " ".join(compound_extra)
    else:
        extra_info = ""

    context = merge_2["context"].str.cat(sep=' ') + " " + extra_info

    return context, merge_2



def get_prompt(instruction, new_system_prompt):
    system_prompt = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + system_prompt + instruction + E_INST
    return prompt_template

def llama_model(model_name, branch_name, cache_dir, temperature=0, top_p=1, max_new_tokens=512, stream=False, method='method-1'):
    if method == 'method-1':
        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                 revision=branch_name,
                                                 cache_dir=cache_dir)
        # model = AutoModelForCausalLM.from_pretrained(model_name,                                             
        #                                     device_map='auto',
        #                                     torch_dtype=torch.float16,
        #                                     revision=branch_name,
        #                                     cache_dir=cache_dir
        #                                     )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if torch.cuda.is_available() else None,  # Sử dụng GPU nếu có
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  # Sử dụng float32 trên CPU
            revision=branch_name,
            cache_dir=cache_dir
        )
        
    elif method == 'method-2':
        import transformers
        tokenizer = transformers.LlamaTokenizer.from_pretrained(model_name, 
                                                                revision=branch_name, 
                                                                cache_dir=cache_dir, 
                                                                legacy=False)
        # model = transformers.LlamaForCausalLM.from_pretrained(model_name, 
        #                                                       device_map='auto', 
        #                                                       torch_dtype=torch.float16, 
        #                                                       revision=branch_name, 
        #                                                       cache_dir=cache_dir)        
        
        model = transformers.LlamaForCausalLM.from_pretrained(model_name, 
                                                              device_map="auto" if torch.cuda.is_available() else None,  # Sử dụng GPU nếu có
                                                              torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  # Sử dụng float32 trên CPU
                                                              revision=branch_name, 
                                                              cache_dir=cache_dir)        
        
    # if not stream:
    #     pipe = pipeline("text-generation",
    #                 model = model,
    #                 tokenizer = tokenizer,
    #                 torch_dtype = torch.bfloat16,
    #                 device_map = "auto",
    #                 max_new_tokens = max_new_tokens,
    #                 do_sample = True
    #                 )
    # else:
    #     streamer = TextStreamer(tokenizer)
    #     pipe = pipeline("text-generation",
    #                 model = model,
    #                 tokenizer = tokenizer,
    #                 torch_dtype = torch.bfloat16,
    #                 device_map = "auto",
    #                 max_new_tokens = max_new_tokens,
    #                 do_sample = True,
    #                 streamer=streamer
    #                 )   
    
    if not stream:
        pipe = pipeline("text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    torch_dtype=torch.float32,  # Sử dụng float32 thay vì bfloat16
                    device_map="auto",
                    max_new_tokens=max_new_tokens,
                    do_sample=True
                    )
    else:
        streamer = TextStreamer(tokenizer)
        pipe = pipeline("text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    torch_dtype=torch.float32,  # Sử dụng float32 thay vì bfloat16
                    device_map="auto",
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    streamer=streamer
                    )        
     
    llm = HuggingFacePipeline(pipeline = pipe,
                              model_kwargs = {"temperature":temperature, "top_p":top_p})
    return llm


@retry(wait=wait_random_exponential(min=10, max=30), stop=stop_after_attempt(5))
def fetch_GPT_response(instruction, system_prompt, chat_model_id, chat_deployment_id, temperature=0):
    # print('Calling OpenAI...')
    # response = openai.ChatCompletion.create(
    #     temperature=temperature,
    #     deployment_id=chat_deployment_id,
    #     model=chat_model_id,
    #     messages=[
    #         {"role": "system", "content": system_prompt},
    #         {"role": "user", "content": instruction}
    #     ]
    # )
    """
    Gọi GPT từ endpoint tùy chỉnh, bỏ qua OpenAI SDK.
    `chat_deployment_id` giữ lại cho tương thích nhưng không sử dụng.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instruction}
    ]
    
    response = request_api(messages=messages, temperature=temperature)

    
    if 'choices' in response \
       and isinstance(response['choices'], list) \
       and len(response) >= 0 \
       and 'message' in response['choices'][0] \
       and 'content' in response['choices'][0]['message']:
        return response['choices'][0]['message']['content']
    else:
        return 'Unexpected response'

@memory.cache
def get_GPT_response(instruction, system_prompt, chat_model_id, chat_deployment_id, temperature=0):
    return fetch_GPT_response(instruction, system_prompt, chat_model_id, chat_deployment_id, temperature)


def stream_out(output):
    CHUNK_SIZE = int(round(len(output)/50))
    SLEEP_TIME = 0.1
    for i in range(0, len(output), CHUNK_SIZE):
        print(output[i:i+CHUNK_SIZE], end='')
        sys.stdout.flush()
        time.sleep(SLEEP_TIME)
    print("\n")

def get_gpt35():
    chat_model_id = 'gpt-35-turbo' if openai.api_type == 'azure' else 'gpt-3.5-turbo'
    chat_deployment_id = chat_model_id if openai.api_type == 'azure' else None
    return chat_model_id, chat_deployment_id

def disease_entity_extractor(text):
    chat_model_id, chat_deployment_id = get_gpt35()
    resp = get_GPT_response(text, system_prompts["DISEASE_ENTITY_EXTRACTION"], chat_model_id, chat_deployment_id, temperature=0)
    try:
        entity_dict = json.loads(resp)
        return entity_dict["Diseases"]
    except:
        return None
    
def disease_entity_extractor_v2(text):
    chat_model_id, chat_deployment_id = get_gpt35()
    prompt_updated = system_prompts["DISEASE_ENTITY_EXTRACTION"] + "\n" + "Sentence : " + text
    resp = get_GPT_response(prompt_updated, system_prompts["DISEASE_ENTITY_EXTRACTION"], chat_model_id, chat_deployment_id, temperature=0)
    
    # print("thu - resp: ", resp)
    
    try:
        entity_dict = json.loads(resp)
        return entity_dict["Diseases"]
    except:
        return None

def load_sentence_transformer(sentence_embedding_model):
    return SentenceTransformerEmbeddings(model_name=sentence_embedding_model)

def load_chroma(vector_db_path, sentence_embedding_model):
    embedding_function = load_sentence_transformer(sentence_embedding_model)
    return Chroma(persist_directory=vector_db_path, embedding_function=embedding_function)

def biomedical_entity_extractor(text):
    chat_model_id, chat_deployment_id = get_gpt35()
    
    prompt = system_prompts["ENTITY_EXTRACTION_MULTI_LABEL"] + "\nSentence: " + text
    resp = get_GPT_response(
        prompt,
        system_prompts["ENTITY_EXTRACTION_MULTI_LABEL"],
        chat_model_id,
        chat_deployment_id,
        temperature=0
    )
    
    try:
        entity_dict = json.loads(resp)
        return entity_dict  # entity_dict có các key: "Diseases", "Drugs", "Genes"
    except Exception as e:
        print("Error parsing response:", e)
        print("Raw response:", resp)
        return None

def retrieve_context(question, vectorstore, embedding_function, node_context_df, context_volume, context_sim_threshold, context_sim_min_threshold, edge_evidence, api=True):
    # Trích xuất entity từ câu hỏi
    entity_dict = biomedical_entity_extractor(question)
    disease_entities = entity_dict.get("Diseases", []) if entity_dict else []
    drug_entities = entity_dict.get("Drugs", []) if entity_dict else []
    gene_entities = entity_dict.get("Genes", []) if entity_dict else []

    logging.debug(f"Extracted Entities: Diseases={disease_entities}, Drugs={drug_entities}, Genes={gene_entities}")

    node_context_extracted = ""
    question_embedding = embedding_function.embed_query(question)

    total_entities = len(disease_entities) + len(drug_entities) + len(gene_entities)
    max_context_per_node = max(1, int(context_volume / max(1, total_entities)))

    def process_entities(entity_list, get_context_fn, entity_type):
        nonlocal node_context_extracted
        for entity in entity_list:
            logging.debug(f"Processing {entity_type} entity: {entity}")

            # Bỏ qua bước tìm kiếm trong vectorstore, truy vấn trực tiếp SPOKE
            if api:
                try:
                    node_context, context_table = get_context_fn(entity)
                    logging.debug(f"Fetched context from API for {entity}")
                except Exception as e:
                    logging.warning(f"Failed to fetch context from SPOKE API for {entity}: {str(e)}")
                    continue
            else:
                # Nếu không dùng API (dùng node_context_df), giữ nguyên logic cũ
                node_search_result = vectorstore.similarity_search_with_score(entity, k=1)
                if not node_search_result:
                    logging.warning(f"No search result for entity: {entity}")
                    continue
                node_name = node_search_result[0][0].page_content
                node_context = node_context_df[node_context_df.node_name == node_name].node_context.values[0]
                context_table = None

            # Chia ngữ cảnh thành các câu và tính độ tương đồng
            node_context_list = node_context.split(". ")
            node_context_embeddings = embedding_function.embed_documents(node_context_list)

            similarities = [
                cosine_similarity(
                    np.array(question_embedding).reshape(1, -1),
                    np.array(ctx_emb).reshape(1, -1)
                ) for ctx_emb in node_context_embeddings
            ]
            similarities = sorted([(s[0], i) for i, s in enumerate(similarities)], reverse=True)

            logging.debug(f"Similarities for node {entity}: {similarities[:5]}")

            # Lọc các câu ngữ cảnh dựa trên ngưỡng độ tương đồng
            percentile_threshold = np.percentile([s[0] for s in similarities], context_sim_threshold)
            high_similarity_indices = [
                s[1] for s in similarities if s[0] > percentile_threshold and s[0] > context_sim_min_threshold
            ]
            logging.debug(f"Selected indices for context (>{context_sim_threshold} percentile): {high_similarity_indices}")

            if len(high_similarity_indices) > max_context_per_node:
                high_similarity_indices = high_similarity_indices[:max_context_per_node]

            high_similarity_context = [node_context_list[i] for i in high_similarity_indices]
            logging.debug(f"Selected context sentences: {high_similarity_context}")

            # Thêm thông tin cạnh (edge evidence) nếu cần
            if edge_evidence and context_table is not None:
                high_similarity_context = list(map(lambda x: x + '.', high_similarity_context))
                context_table = context_table[context_table.context.isin(high_similarity_context)]
                context_table.loc[:, "context"] = (
                    context_table.source + " " + context_table.predicate.str.lower() + " " +
                    context_table.target + " and Provenance of this association is " +
                    context_table.provenance + " and attributes associated with this association is in the following JSON format:\n " +
                    context_table.evidence.astype('str') + "\n\n"
                )
                node_context_extracted += context_table.context.str.cat(sep=' ')
            else:
                node_context_extracted += ". ".join(high_similarity_context) + ". "

    # Gọi API SPOKE trực tiếp cho từng loại entity
    process_entities(disease_entities, get_context_using_spoke_api_disease, "Disease")
    process_entities(drug_entities, get_context_using_spoke_api_drugs, "Drug")
    process_entities(gene_entities, get_context_using_spoke_api_gene, "Gene")

    # Fallback nếu không có entity
    if total_entities == 0:
        logging.info("No entities found, fallback to question-based search.")
        node_hits = vectorstore.similarity_search_with_score(question, k=5)
        max_context_per_node = max(1, int(context_volume / 5))

        for node in node_hits:
            node_name = node[0].page_content
            logging.debug(f"Fallback node: {node_name}")

            if not api:
                node_context = node_context_df[node_context_df.node_name == node_name].node_context.values[0]
                context_table = None
            else:
                node_context, context_table = get_context_using_spoke_api_disease(node_name)
                logging.debug(f"Fetched fallback context from API for {node_name}")

            node_context_list = node_context.split(". ")
            node_context_embeddings = embedding_function.embed_documents(node_context_list)

            similarities = [
                cosine_similarity(
                    np.array(question_embedding).reshape(1, -1),
                    np.array(ctx_emb).reshape(1, -1)
                ) for ctx_emb in node_context_embeddings
            ]
            similarities = sorted([(s[0], i) for i, s in enumerate(similarities)], reverse=True)

            logging.debug(f"Similarities for fallback node {node_name}: {similarities[:5]}")

            percentile_threshold = np.percentile([s[0] for s in similarities], context_sim_threshold)
            high_similarity_indices = [
                s[1] for s in similarities if s[0] > percentile_threshold and s[0] > context_sim_min_threshold
            ]

            if len(high_similarity_indices) > max_context_per_node:
                high_similarity_indices = high_similarity_indices[:max_context_per_node]

            high_similarity_context = [node_context_list[i] for i in high_similarity_indices]
            logging.debug(f"Selected fallback context: {high_similarity_context}")

            if edge_evidence and context_table is not None:
                high_similarity_context = list(map(lambda x: x + '.', high_similarity_context))
                context_table = context_table[context_table.context.isin(high_similarity_context)]
                context_table.loc[:, "context"] = (
                    context_table.source + " " + context_table.predicate.str.lower() + " " +
                    context_table.target + " and Provenance of this association is " +
                    context_table.provenance + " and attributes associated with this association is in the following JSON format:\n " +
                    context_table.evidence.astype('str') + "\n\n"
                )
                node_context_extracted += context_table.context.str.cat(sep=' ')
            else:
                node_context_extracted += ". ".join(high_similarity_context) + ". "

    return node_context_extracted

def retrieve_context_search_DB(question, vectorstore, embedding_function, node_context_df, context_volume, context_sim_threshold, context_sim_min_threshold, edge_evidence, api=True):
    entity_dict = biomedical_entity_extractor(question)
    disease_entities = entity_dict.get("Diseases", []) if entity_dict else []
    drug_entities = entity_dict.get("Drugs", []) if entity_dict else []
    gene_entities = entity_dict.get("Genes", []) if entity_dict else []

    logging.debug(f"Extracted Entities: Diseases={disease_entities}, Drugs={drug_entities}, Genes={gene_entities}")

    node_hits = []
    node_context_extracted = ""
    question_embedding = embedding_function.embed_query(question)

    total_entities = len(disease_entities) + len(drug_entities) + len(gene_entities)
    max_context_per_node = max(1, int(context_volume / max(1, total_entities)))

    def process_entities(entity_list, get_context_fn, entity_type):
        nonlocal node_context_extracted
        for entity in entity_list:
            logging.debug(f"Processing {entity_type} entity: {entity}")
            node_search_result = vectorstore.similarity_search_with_score(entity, k=1)
            if not node_search_result:
                logging.warning(f"No search result for entity: {entity}")
                continue

            node_name = node_search_result[0][0].page_content
            logging.debug(f"Top matched node for {entity}: {node_name}")

            if not api:
                node_context = node_context_df[node_context_df.node_name == node_name].node_context.values[0]
                context_table = None
            else:
                node_context, context_table = get_context_fn(node_name)
                logging.debug(f"Fetched context from API for {node_name}")

            node_context_list = node_context.split(". ")
            node_context_embeddings = embedding_function.embed_documents(node_context_list)

            similarities = [
                cosine_similarity(
                    np.array(question_embedding).reshape(1, -1),
                    np.array(ctx_emb).reshape(1, -1)
                ) for ctx_emb in node_context_embeddings
            ]
            similarities = sorted([(s[0], i) for i, s in enumerate(similarities)], reverse=True)

            logging.debug(f"Similarities for node {node_name}: {similarities[:5]}")

            percentile_threshold = np.percentile([s[0] for s in similarities], context_sim_threshold)
            high_similarity_indices = [
                s[1] for s in similarities if s[0] > percentile_threshold and s[0] > context_sim_min_threshold
            ]
            logging.debug(f"Selected indices for context (>{context_sim_threshold} percentile): {high_similarity_indices}")

            if len(high_similarity_indices) > max_context_per_node:
                high_similarity_indices = high_similarity_indices[:max_context_per_node]

            high_similarity_context = [node_context_list[i] for i in high_similarity_indices]
            logging.debug(f"Selected context sentences: {high_similarity_context}")

            if edge_evidence and context_table is not None:
                high_similarity_context = list(map(lambda x: x + '.', high_similarity_context))
                context_table = context_table[context_table.context.isin(high_similarity_context)]
                context_table.loc[:, "context"] = (
                    context_table.source + " " + context_table.predicate.str.lower() + " " +
                    context_table.target + " and Provenance of this association is " +
                    context_table.provenance + " and attributes associated with this association is in the following JSON format:\n " +
                    context_table.evidence.astype('str') + "\n\n"
                )
                node_context_extracted += context_table.context.str.cat(sep=' ')
            else:
                node_context_extracted += ". ".join(high_similarity_context) + ". "

    # Gọi API theo từng loại entity
    process_entities(disease_entities, get_context_using_spoke_api_disease, "Disease")
    process_entities(drug_entities, get_context_using_spoke_api_drugs, "Drug") #"Drug"
    process_entities(gene_entities, get_context_using_spoke_api_gene, "Gene")

    # Fallback nếu không có entity
    if total_entities == 0:
        logging.info("No entities found, fallback to question-based search.")
        node_hits = vectorstore.similarity_search_with_score(question, k=5)
        max_context_per_node = max(1, int(context_volume / 5))

        for node in node_hits:
            node_name = node[0].page_content
            logging.debug(f"Fallback node: {node_name}")

            if not api:
                node_context = node_context_df[node_context_df.node_name == node_name].node_context.values[0]
                context_table = None
            else:
                node_context, context_table = get_context_using_spoke_api("Disease", node_name)
                logging.debug(f"Fetched fallback context from API for {node_name}")

            node_context_list = node_context.split(". ")
            node_context_embeddings = embedding_function.embed_documents(node_context_list)

            similarities = [
                cosine_similarity(
                    np.array(question_embedding).reshape(1, -1),
                    np.array(ctx_emb).reshape(1, -1)
                ) for ctx_emb in node_context_embeddings
            ]
            similarities = sorted([(s[0], i) for i, s in enumerate(similarities)], reverse=True)

            logging.debug(f"Similarities for fallback node {node_name}: {similarities[:5]}")

            percentile_threshold = np.percentile([s[0] for s in similarities], context_sim_threshold)
            high_similarity_indices = [
                s[1] for s in similarities if s[0] > percentile_threshold and s[0] > context_sim_min_threshold
            ]

            if len(high_similarity_indices) > max_context_per_node:
                high_similarity_indices = high_similarity_indices[:max_context_per_node]

            high_similarity_context = [node_context_list[i] for i in high_similarity_indices]
            logging.debug(f"Selected fallback context: {high_similarity_context}")

            if edge_evidence and context_table is not None:
                high_similarity_context = list(map(lambda x: x + '.', high_similarity_context))
                context_table = context_table[context_table.context.isin(high_similarity_context)]
                context_table.loc[:, "context"] = (
                    context_table.source + " " + context_table.predicate.str.lower() + " " +
                    context_table.target + " and Provenance of this association is " +
                    context_table.provenance + " and attributes associated with this association is in the following JSON format:\n " +
                    context_table.evidence.astype('str') + "\n\n"
                )
                node_context_extracted += context_table.context.str.cat(sep=' ')
            else:
                node_context_extracted += ". ".join(high_similarity_context) + ". "

    return node_context_extracted
    
# def retrieve_context(question, vectorstore, embedding_function, node_context_df, context_volume, context_sim_threshold, context_sim_min_threshold, edge_evidence, api=True):
#     entity_dict = biomedical_entity_extractor(question)
#     disease_entities = entity_dict.get("Diseases", []) if entity_dict else []
#     drug_entities = entity_dict.get("Drugs", []) if entity_dict else []
#     gene_entities = entity_dict.get("Genes", []) if entity_dict else []

#     node_hits = []
#     node_context_extracted = ""
#     question_embedding = embedding_function.embed_query(question)

#     # Tổng số entity để chia đều context volume
#     total_entities = len(disease_entities) + len(drug_entities) + len(gene_entities)
#     max_context_per_node = max(1, int(context_volume / max(1, total_entities)))

#     def process_entities(entity_list, get_context_fn):
#         nonlocal node_context_extracted

#         for entity in entity_list:
#             node_search_result = vectorstore.similarity_search_with_score(entity, k=1)
#             if not node_search_result:
#                 continue
#             node_name = node_search_result[0][0].page_content

#             if not api:
#                 node_context = node_context_df[node_context_df.node_name == node_name].node_context.values[0]
#                 context_table = None
#             else:
#                 node_context, context_table = get_context_fn(node_name)

#             node_context_list = node_context.split(". ")
#             node_context_embeddings = embedding_function.embed_documents(node_context_list)

#             similarities = [
#                 cosine_similarity(
#                     np.array(question_embedding).reshape(1, -1),
#                     np.array(ctx_emb).reshape(1, -1)
#                 ) for ctx_emb in node_context_embeddings
#             ]
#             similarities = sorted([(s[0], i) for i, s in enumerate(similarities)], reverse=True)

#             percentile_threshold = np.percentile([s[0] for s in similarities], context_sim_threshold)
#             high_similarity_indices = [
#                 s[1] for s in similarities if s[0] > percentile_threshold and s[0] > context_sim_min_threshold
#             ]

#             if len(high_similarity_indices) > max_context_per_node:
#                 high_similarity_indices = high_similarity_indices[:max_context_per_node]

#             high_similarity_context = [node_context_list[i] for i in high_similarity_indices]

#             if edge_evidence and context_table is not None:
#                 high_similarity_context = list(map(lambda x: x + '.', high_similarity_context))
#                 context_table = context_table[context_table.context.isin(high_similarity_context)]
#                 context_table.loc[:, "context"] = (
#                     context_table.source + " " + context_table.predicate.str.lower() + " " +
#                     context_table.target + " and Provenance of this association is " +
#                     context_table.provenance + " and attributes associated with this association is in the following JSON format:\n " +
#                     context_table.evidence.astype('str') + "\n\n"
#                 )
#                 node_context_extracted += context_table.context.str.cat(sep=' ')
#             else:
#                 node_context_extracted += ". ".join(high_similarity_context) + ". "

#     # Gọi API tương ứng từng loại
#     process_entities(disease_entities, get_context_using_spoke_api_disease)
#     process_entities(drug_entities, get_context_using_spoke_api_drug)
#     process_entities(gene_entities, get_context_using_spoke_api_gene)

#     # Nếu không có entity nào → fallback search theo câu hỏi
#     if total_entities == 0:
#         node_hits = vectorstore.similarity_search_with_score(question, k=5)
#         max_context_per_node = max(1, int(context_volume / 5))

#         for node in node_hits:
#             node_name = node[0].page_content
#             if not api:
#                 node_context = node_context_df[node_context_df.node_name == node_name].node_context.values[0]
#                 context_table = None
#             else:
#                 node_context, context_table = get_context_using_spoke_api(node_name)

#             node_context_list = node_context.split(". ")
#             node_context_embeddings = embedding_function.embed_documents(node_context_list)

#             similarities = [
#                 cosine_similarity(
#                     np.array(question_embedding).reshape(1, -1),
#                     np.array(ctx_emb).reshape(1, -1)
#                 ) for ctx_emb in node_context_embeddings
#             ]
#             similarities = sorted([(s[0], i) for i, s in enumerate(similarities)], reverse=True)

#             percentile_threshold = np.percentile([s[0] for s in similarities], context_sim_threshold)
#             high_similarity_indices = [
#                 s[1] for s in similarities if s[0] > percentile_threshold and s[0] > context_sim_min_threshold
#             ]

#             if len(high_similarity_indices) > max_context_per_node:
#                 high_similarity_indices = high_similarity_indices[:max_context_per_node]

#             high_similarity_context = [node_context_list[i] for i in high_similarity_indices]

#             if edge_evidence and context_table is not None:
#                 high_similarity_context = list(map(lambda x: x + '.', high_similarity_context))
#                 context_table = context_table[context_table.context.isin(high_similarity_context)]
#                 context_table.loc[:, "context"] = (
#                     context_table.source + " " + context_table.predicate.str.lower() + " " +
#                     context_table.target + " and Provenance of this association is " +
#                     context_table.provenance + " and attributes associated with this association is in the following JSON format:\n " +
#                     context_table.evidence.astype('str') + "\n\n"
#                 )
#                 node_context_extracted += context_table.context.str.cat(sep=' ')
#             else:
#                 node_context_extracted += ". ".join(high_similarity_context) + ". "

#     return node_context_extracted

# def retrieve_context(question, vectorstore, embedding_function, node_context_df, context_volume, context_sim_threshold, context_sim_min_threshold, edge_evidence, api=True):
#     entities = disease_entity_extractor_v2(question)
#     node_hits = []
#     if entities:
#         max_number_of_high_similarity_context_per_node = int(context_volume/len(entities))
#         for entity in entities:
#             node_search_result = vectorstore.similarity_search_with_score(entity, k=1)
#             print("Thu print")
#             print(f"Search result for entity '{entity}': {node_search_result}")
            
#             if node_search_result:
#                 node_hits.append(node_search_result[0][0].page_content)
#             else:
#                 print(f"No results found for entity: {entity}")
                
#         question_embedding = embedding_function.embed_query(question)
#         node_context_extracted = ""
#         for node_name in node_hits:
#             if not api:
#                 node_context = node_context_df[node_context_df.node_name == node_name].node_context.values[0]
#             else:
#                 node_context,context_table = get_context_using_spoke_api(node_name)
#             node_context_list = node_context.split(". ")        
#             node_context_embeddings = embedding_function.embed_documents(node_context_list)
#             similarities = [cosine_similarity(np.array(question_embedding).reshape(1, -1), np.array(node_context_embedding).reshape(1, -1)) for node_context_embedding in node_context_embeddings]
#             similarities = sorted([(e, i) for i, e in enumerate(similarities)], reverse=True)
#             percentile_threshold = np.percentile([s[0] for s in similarities], context_sim_threshold)
#             high_similarity_indices = [s[1] for s in similarities if s[0] > percentile_threshold and s[0] > context_sim_min_threshold]
#             if len(high_similarity_indices) > max_number_of_high_similarity_context_per_node:
#                 high_similarity_indices = high_similarity_indices[:max_number_of_high_similarity_context_per_node]
#             high_similarity_context = [node_context_list[index] for index in high_similarity_indices]            
#             if edge_evidence:
#                 high_similarity_context = list(map(lambda x:x+'.', high_similarity_context)) 
#                 context_table = context_table[context_table.context.isin(high_similarity_context)]
#                 context_table.loc[:, "context"] =  context_table.source + " " + context_table.predicate.str.lower() + " " + context_table.target + " and Provenance of this association is " + context_table.provenance + " and attributes associated with this association is in the following JSON format:\n " + context_table.evidence.astype('str') + "\n\n"                
#                 node_context_extracted += context_table.context.str.cat(sep=' ')
#             else:
#                 node_context_extracted += ". ".join(high_similarity_context)
#                 node_context_extracted += ". "
#         return node_context_extracted
#     else:
#         node_hits = vectorstore.similarity_search_with_score(question, k=5)
#         max_number_of_high_similarity_context_per_node = int(context_volume/5)
#         question_embedding = embedding_function.embed_query(question)
#         node_context_extracted = ""
#         for node in node_hits:
#             node_name = node[0].page_content
#             if not api:
#                 node_context = node_context_df[node_context_df.node_name == node_name].node_context.values[0]
#             else:
#                 node_context, context_table = get_context_using_spoke_api(node_name)
#             node_context_list = node_context.split(". ")        
#             node_context_embeddings = embedding_function.embed_documents(node_context_list)
#             similarities = [cosine_similarity(np.array(question_embedding).reshape(1, -1), np.array(node_context_embedding).reshape(1, -1)) for node_context_embedding in node_context_embeddings]
#             similarities = sorted([(e, i) for i, e in enumerate(similarities)], reverse=True)
#             percentile_threshold = np.percentile([s[0] for s in similarities], context_sim_threshold)
#             high_similarity_indices = [s[1] for s in similarities if s[0] > percentile_threshold and s[0] > context_sim_min_threshold]
#             if len(high_similarity_indices) > max_number_of_high_similarity_context_per_node:
#                 high_similarity_indices = high_similarity_indices[:max_number_of_high_similarity_context_per_node]
#             high_similarity_context = [node_context_list[index] for index in high_similarity_indices]
#             if edge_evidence:
#                 high_similarity_context = list(map(lambda x:x+'.', high_similarity_context))
#                 context_table = context_table[context_table.context.isin(high_similarity_context)]
#                 context_table.loc[:, "context"] =  context_table.source + " " + context_table.predicate.str.lower() + " " + context_table.target + " and Provenance of this association is " + context_table.provenance + " and attributes associated with this association is in the following JSON format:\n " + context_table.evidence.astype('str') + "\n\n"                
#                 node_context_extracted += context_table.context.str.cat(sep=' ')
#             else:
#                 node_context_extracted += ". ".join(high_similarity_context)
#                 node_context_extracted += ". "
#         return node_context_extracted
    


# def retrieve_context(question, vectorstore, embedding_function, node_context_df, context_volume, context_sim_threshold, context_sim_min_threshold, edge_evidence, api=True):
#     # Extract entities using biomedical_entity_extractor
#     entity_dict = biomedical_entity_extractor(question)
#     node_hits = []
    
#     if entity_dict:
#         # Collect entities with their types
#         all_entities_with_types = []
#         for entity_type in ["Diseases", "Drugs", "Genes"]:
#             entities = entity_dict.get(entity_type, [])
#             # Pair each entity with its type
#             all_entities_with_types.extend([(entity, entity_type) for entity in entities])
        
#         # Avoid division by zero if no entities are found
#         if not all_entities_with_types:
#             all_entities_with_types = [(question, None)]  # Fallback to question with no type
#             max_number_of_high_similarity_context_per_node = context_volume // 5
#         else:
#             max_number_of_high_similarity_context_per_node = int(context_volume / len(all_entities_with_types))
        
#         # Perform similarity search for each entity
#         for entity, entity_type in all_entities_with_types:
#             # If entity_type is None (fallback case), use the question directly
#             search_term = entity if entity_type else question
#             node_search_result = vectorstore.similarity_search_with_score(search_term, k=1)
#             print("Thu print")
#             print(f"Search result for entity '{search_term}': {node_search_result}")
            
#             if node_search_result:
#                 # Store the node name along with the entity type
#                 node_hits.append((node_search_result[0][0].page_content, entity_type))
#             else:
#                 print(f"No results found for entity: {search_term}")
                
#         # Extract context for each node hit
#         question_embedding = embedding_function.embed_query(question)
#         node_context_extracted = ""
#         for node_name, entity_type in node_hits:
#             if not api:
#                 node_context = node_context_df[node_context_df.node_name == node_name].node_context.values[0]
#                 context_table = None  # Not used in non-API mode
#             else:
#                 # Call the appropriate API based on entity type
#                 if entity_type == "Diseases":
#                     node_context, context_table = get_context_using_spoke_api_disease(node_name)
#                 elif entity_type == "Drugs":
#                     node_context, context_table = get_context_using_spoke_api_drug(node_name)
#                 elif entity_type == "Genes":
#                     node_context, context_table = get_context_using_spoke_api_gene(node_name)
#                 else:
#                     # Fallback case (when entity_type is None)
#                     node_context, context_table = get_context_using_spoke_api_disease(node_name)  # Default to disease API for fallback
                
#             node_context_list = node_context.split(". ")        
#             node_context_embeddings = embedding_function.embed_documents(node_context_list)
#             similarities = [cosine_similarity(np.array(question_embedding).reshape(1, -1), np.array(node_context_embedding).reshape(1, -1)) for node_context_embedding in node_context_embeddings]
#             similarities = sorted([(e, i) for i, e in enumerate(similarities)], reverse=True)
#             percentile_threshold = np.percentile([s[0] for s in similarities], context_sim_threshold)
#             high_similarity_indices = [s[1] for s in similarities if s[0] > percentile_threshold and s[0] > context_sim_min_threshold]
#             if len(high_similarity_indices) > max_number_of_high_similarity_context_per_node:
#                 high_similarity_indices = high_similarity_indices[:max_number_of_high_similarity_context_per_node]
#             high_similarity_context = [node_context_list[index] for index in high_similarity_indices]            
#             if edge_evidence:
#                 high_similarity_context = list(map(lambda x: x + '.', high_similarity_context)) 
#                 context_table = context_table[context_table.context.isin(high_similarity_context)]
#                 context_table.loc[:, "context"] = context_table.source + " " + context_table.predicate.str.lower() + " " + context_table.target + " and Provenance of this association is " + context_table.provenance + " and attributes associated with this association is in the following JSON format:\n " + context_table.evidence.astype('str') + "\n\n"                
#                 node_context_extracted += context_table.context.str.cat(sep=' ')
#             else:
#                 node_context_extracted += ". ".join(high_similarity_context)
#                 node_context_extracted += ". "
#         return node_context_extracted
#     else:
#         # Fallback if no entities are extracted
#         node_hits = vectorstore.similarity_search_with_score(question, k=5)
#         max_number_of_high_similarity_context_per_node = int(context_volume / 5)
#         question_embedding = embedding_function.embed_query(question)
#         node_context_extracted = ""
#         for node in node_hits:
#             node_name = node[0].page_content
#             if not api:
#                 node_context = node_context_df[node_context_df.node_name == node_name].node_context.values[0]
#                 context_table = None
#             else:
#                 # Default to disease API for fallback case
#                 node_context, context_table = get_context_using_spoke_api_disease(node_name)
#             node_context_list = node_context.split(". ")        
#             node_context_embeddings = embedding_function.embed_documents(node_context_list)
#             similarities = [cosine_similarity(np.array(question_embedding).reshape(1, -1), np.array(node_context_embedding).reshape(1, -1)) for node_context_embedding in node_context_embeddings]
#             similarities = sorted([(e, i) for i, e in enumerate(similarities)], reverse=True)
#             percentile_threshold = np.percentile([s[0] for s in similarities], context_sim_threshold)
#             high_similarity_indices = [s[1] for s in similarities if s[0] > percentile_threshold and s[0] > context_sim_min_threshold]
#             if len(high_similarity_indices) > max_number_of_high_similarity_context_per_node:
#                 high_similarity_indices = high_similarity_indices[:max_number_of_high_similarity_context_per_node]
#             high_similarity_context = [node_context_list[index] for index in high_similarity_indices]
#             if edge_evidence:
#                 high_similarity_context = list(map(lambda x: x + '.', high_similarity_context))
#                 context_table = context_table[context_table.context.isin(high_similarity_context)]
#                 context_table.loc[:, "context"] = context_table.source + " " + context_table.predicate.str.lower() + " " + context_table.target + " and Provenance of this association is " + context_table.provenance + " and attributes associated with this association is in the following JSON format:\n " + context_table.evidence.astype('str') + "\n\n"                
#                 node_context_extracted += context_table.context.str.cat(sep=' ')
#             else:
#                 node_context_extracted += ". ".join(high_similarity_context)
#                 node_context_extracted += ". "
#         return node_context_extracted

def interactive(question, vectorstore, node_context_df, embedding_function_for_context_retrieval, llm_type, edge_evidence, system_prompt, api=True, llama_method="method-1"):
    print(" ")
    input("Press enter for Step 1 - Disease entity extraction using GPT-3.5-Turbo")
    print("Processing ...")
    entities = disease_entity_extractor_v2(question)
    
    #THU add
    #print(f"Entities: {entities}")

    max_number_of_high_similarity_context_per_node = int(config_data["CONTEXT_VOLUME"]/len(entities))
    print("Extracted entity from the prompt = '{}'".format(", ".join(entities)))
    print(" ")
    
    input("Press enter for Step 2 - Match extracted Disease entity to SPOKE nodes")
    print("Finding vector similarity ...")
    node_hits = []
    for entity in entities:
        node_search_result = vectorstore.similarity_search_with_score(entity, k=1)
        node_hits.append(node_search_result[0][0].page_content)
    print("Matched entities from SPOKE = '{}'".format(", ".join(node_hits)))
    print(" ")
    
    input("Press enter for Step 3 - Context extraction from SPOKE")
    node_context = []
    for node_name in node_hits:
        if not api:
            node_context.append(node_context_df[node_context_df.node_name == node_name].node_context.values[0])
        else:
            context, context_table = get_context_using_spoke_api(node_name)
            node_context.append(context)
    print("Extracted Context is : ")
    print(". ".join(node_context))
    print(" ")

    input("Press enter for Step 4 - Context pruning")
    question_embedding = embedding_function_for_context_retrieval.embed_query(question)
    node_context_extracted = ""
    for node_name in node_hits:
        if not api:
            node_context = node_context_df[node_context_df.node_name == node_name].node_context.values[0]
        else:
            node_context, context_table = get_context_using_spoke_api(node_name)                        
        node_context_list = node_context.split(". ")        
        node_context_embeddings = embedding_function_for_context_retrieval.embed_documents(node_context_list)
        similarities = [cosine_similarity(np.array(question_embedding).reshape(1, -1), np.array(node_context_embedding).reshape(1, -1)) for node_context_embedding in node_context_embeddings]
        similarities = sorted([(e, i) for i, e in enumerate(similarities)], reverse=True)
        percentile_threshold = np.percentile([s[0] for s in similarities], config_data["QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD"])
        high_similarity_indices = [s[1] for s in similarities if s[0] > percentile_threshold and s[0] > config_data["QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY"]]
        if len(high_similarity_indices) > max_number_of_high_similarity_context_per_node:
            high_similarity_indices = high_similarity_indices[:max_number_of_high_similarity_context_per_node]
        high_similarity_context = [node_context_list[index] for index in high_similarity_indices]               
        if edge_evidence:
            high_similarity_context = list(map(lambda x:x+'.', high_similarity_context)) 
            context_table = context_table[context_table.context.isin(high_similarity_context)]
            context_table.loc[:, "context"] =  context_table.source + " " + context_table.predicate.str.lower() + " " + context_table.target + " and Provenance of this association is " + context_table.provenance + " and attributes associated with this association is in the following JSON format:\n " + context_table.evidence.astype('str') + "\n\n"                
            node_context_extracted += context_table.context.str.cat(sep=' ')
        else:
            node_context_extracted += ". ".join(high_similarity_context)
            node_context_extracted += ". "
    print("Pruned Context is : ")
    print(node_context_extracted)
    print(" ")
    
    input("Press enter for Step 5 - LLM prompting")
    print("Prompting ", llm_type)
    if llm_type == "llama":
        from langchain import PromptTemplate, LLMChain
        template = get_prompt("Context:\n\n{context} \n\nQuestion: {question}", system_prompt)
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        llm = llama_model(config_data["LLAMA_MODEL_NAME"], config_data["LLAMA_MODEL_BRANCH"], config_data["LLM_CACHE_DIR"], stream=True, method=llama_method) 
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        output = llm_chain.run(context=node_context_extracted, question=question)
    elif "gpt" in llm_type:
        enriched_prompt = "Context: "+ node_context_extracted + "\n" + "Question: " + question
        output = get_GPT_response(enriched_prompt, system_prompt, llm_type, llm_type, temperature=config_data["LLM_TEMPERATURE"])
        stream_out(output)