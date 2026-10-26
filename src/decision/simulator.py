import numpy as np
import networkx as nx
import joblib
import pandas as pd
from src.data.graph import load_dependency_graph

class CascadeSimulator:
    def __init__(self, model_path='models/cascade_predictor.joblib', 
                 encoder_path='models/label_encoder.joblib', 
                 graph_path='data/raw/api_failure_cascade_dataset.csv'):
        self.model = joblib.load(model_path)
        self.encoder = joblib.load(encoder_path)
        self.graph = load_dependency_graph(graph_path)
        
        # Mapping for severity probabilities (Monte Carlo seeds)
        # These are heuristic estimates for simulation, NOT learned causal probabilities.
        self.severity_probs = {
            'Low': 0.1,
            'Medium': 0.3,
            'High': 0.6,
            'Critical': 0.9
        }

    def whatif(self, service, severity):
        """
        Simulates the spread of a failure from a specific service.
        Returns potentially affected services and estimated impact.
        """
        if service not in self.graph:
            return {"error": f"Service {service} not found in graph."}
        
        if severity not in self.severity_probs:
            return {"error": f"Invalid severity. Choose from {list(self.severity_probs.keys())}"}

        # We traverse the graph in the direction of impact.
        # In the current graph: Service -> Dependency.
        # A failure in a dependency affects its CALLERS.
        # So we traverse the REVERSE graph to see who is impacted by 'service'.
        impact_graph = self.graph.reverse()
        
        # BFS to find all potentially affected services
        affected = nx.single_source_shortest_path_length(impact_graph, service)
        
        base_prob = self.severity_probs[severity]
        results = []
        
        for node, dist in affected.items():
            # Probability decays with distance (Heuristic)
            # P(impact) = base_prob * (0.7 ^ distance)
            est_prob = base_prob * (0.7 ** dist)
            
            results.append({
                'service': node,
                'distance': dist,
                'est_failure_prob': round(est_prob, 3),
                'source': 'Monte Carlo Heuristic'
            })
        
        # Sort by probability descending
        results = sorted(results, key=lambda x: x['est_failure_prob'], reverse=True)
        
        return {
            'root_service': service,
            'input_severity': severity,
            'cascade_depth': max(affected.values()) if affected else 0,
            'impacted_services': results
        }
