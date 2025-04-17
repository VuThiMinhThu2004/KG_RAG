import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Memory
import torch

import ast
import requests

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

from config_loader import *


memory = Memory("cachegpt", verbose=0)


torch.cuda.empty_cache()
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def get_spoke_api_resp(base_uri, end_point, params=None):
    uri = base_uri + end_point
    if params:
        return requests.get(uri, params=params)
    else:
        return requests.get(uri)

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

# print(get_context_using_spoke_api_disease("Alagille syndrome"))
print(get_context_using_spoke_api_gene("UGT1A1"))