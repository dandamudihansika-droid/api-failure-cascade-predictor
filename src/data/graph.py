import networkx as nx
import pandas as pd

def load_dependency_graph(csv_path):
    """
    Loads dependency graph. Since the CSV contains only dependency COUNTS,
    we implement a documented, configurable dependency map to prevent misleading simulations.
    """
    df = pd.read_csv(csv_path)
    services = df['service_name'].unique()
    G = nx.DiGraph()
    G.add_nodes_from(services)
    
    # DOCUMENTED DEPENDENCY MAP (Consistent with Microservice Architecture)
    # Format: Caller -> Callee (Caller depends on Callee)
    dependency_map = {
        'PaymentService': ['AuthService', 'InventoryService'],
        'InventoryService': ['DatabaseService'],
        'AuthService': ['DatabaseService', 'LDAPService'],
        'SearchService': ['InventoryService', 'AnalyticsService'],
        'AnalyticsService': ['DatabaseService'],
        'GatewayService': ['AuthService', 'PaymentService', 'SearchService']
    }
    
    for caller, callees in dependency_map.items():
        if caller in G:
            for callee in callees:
                if callee in G:
                    G.add_edge(caller, callee)
    
    return G
