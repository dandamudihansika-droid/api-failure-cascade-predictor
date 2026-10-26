import pandas as pd
import networkx as nx
from src.data.graph import load_dependency_graph

def calculate_graph_features(df, graph):
    """
    Generates graph-based features from the dependency graph.
    
    Edge direction: caller -> callee (service depends on callee)
    """
    # Precompute graph metrics
    in_degree = dict(graph.in_degree())
    out_degree = dict(graph.out_degree())
    
    # Betweenness centrality
    betweenness = nx.betweenness_centrality(graph, normalized=True)
    
    # Transitive callers: nodes that can reach the current node
    # We can use all_pairs_shortest_path_length or simply count ancestors
    transitive_callers = {}
    for node in graph.nodes():
        # pred_set contains all nodes that can reach 'node'
        # In a DiGraph, this is the set of nodes in the reverse graph that 'node' can reach
        ancestors = nx.ancestors(graph, node)
        transitive_callers[node] = len(ancestors)

    # Map these to the dataframe
    df['graph_in_degree'] = df['service_name'].map(in_degree).fillna(0)
    df['graph_out_degree'] = df['service_name'].map(out_degree).fillna(0)
    df['graph_betweenness'] = df['service_name'].map(betweenness).fillna(0)
    df['graph_transitive_callers'] = df['service_name'].map(transitive_callers).fillna(0)
    
    # Critical path length (max distance from any root to this node)
    # Simplified as the longest path to this node in the DAG
    # If it's not a DAG, we use a heuristic
    path_lengths = {}
    try:
        # Only if it's a DAG
        if nx.is_directed_acyclic_graph(graph):
            for node in graph.nodes():
                # Longest path in DAG to node
                # We can iterate through all nodes and find max distance
                # This is inefficient but correct for small graphs
                dists = nx.single_source_shortest_path_length(graph, node) # This is from node TO others
                # We need TO node. Use reverse graph.
                rev_graph = graph.reverse()
                dists_to = nx.single_source_shortest_path_length(rev_graph, node)
                path_lengths[node] = max(dists_to.values()) if dists_to else 0
        else:
            # Non-DAG fallback: just use in-degree
            path_lengths = {node: in_degree[node] for node in graph.nodes()}
    except Exception:
        path_lengths = {node: 0 for node in graph.nodes()}
        
    df['graph_critical_path_len'] = df['service_name'].map(path_lengths).fillna(0)
    
    return df
