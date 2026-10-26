import pandas as pd

class MitigationRecommender:
    def __init__(self):
        # Criticality weights for services could be added here
        pass

    def recommend(self, service, current_metrics, risk_level):
        """
        Provide explainable mitigation rules based on service metrics and risk.
        """
        recommendations = []
        
        # Logic for Circuit Breakers
        # Trigger if error rate is high AND risk is Medium+
        error_rate = current_metrics.get('error_rate', 0)
        if error_rate > 0.15 and risk_level in ['Medium', 'High', 'Critical']:
            recommendations.append({
                'action': 'Enable Circuit Breaker',
                'reason': f"High error rate ({error_rate:.2%}) and {risk_level} risk. Prevent cascading failure to callers.",
                'priority': 'Immediate'
            })
            
        # Logic for Timeouts
        # If latency is high, we might want to LOWER timeouts to fail fast.
        # CRITICAL RULE: Never recommend increasing timeouts automatically during resource exhaustion.
        latency = current_metrics.get('latency_p95', 0)
        if latency > 500: # Example threshold
            recommendations.append({
                'action': 'Reduce Request Timeout',
                'reason': "High p95 latency detected. Lowering timeouts will prevent thread exhaustion in calling services.",
                'priority': 'High'
            })
        
        # Logic for Scaling
        cpu = current_metrics.get('cpu_usage', 0)
        if cpu > 80:
            recommendations.append({
                'action': 'Horizontal Scale-Out',
                'reason': f"CPU utilization at {cpu:.1f}%. Increase instance count to reduce per-node load.",
                'priority': 'Medium'
            })
            
        # Logic for Retries
        # If error rate is high, stop retries to avoid "retry storm"
        if error_rate > 0.3:
            recommendations.append({
                'action': 'Disable Automatic Retries',
                'reason': "Extreme error rate detected. Retries will likely exacerbate system load (Retry Storm).",
                'priority': 'Immediate'
            })

        if not recommendations:
            recommendations.append({
                'action': 'Monitor',
                'reason': "Metrics within acceptable bounds for current risk level.",
                'priority': 'Low'
            })
            
        return recommendations
