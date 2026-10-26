from collections import deque

class AlertEngine:
    def __init__(self, threshold=0.7, window_size=2):
        self.threshold = threshold
        self.window_size = window_size
        # Store history per service: {service_name: deque([risk_scores], maxlen=window_size)}
        self.history = {}

    def process_observation(self, service, risk_score):
        """
        Generate alert if predicted risk exceeds threshold for consecutive intervals.
        Expects a normalized score or probability in [0.0, 1.0].
        If an unbounded or 0-100 scale score is passed (score > 1.0),
        it is normalized to [0.0, 1.0] by dividing by 100.0 to avoid false positives.
        """
        # Normalize score scale if given on 0-100 scale
        if risk_score > 1.0:
            risk_score = min(float(risk_score) / 100.0, 1.0)
        else:
            risk_score = max(float(risk_score), 0.0)

        if service not in self.history:
            self.history[service] = deque(maxlen=self.window_size)
        
        self.history[service].append(risk_score)
        
        # Check for consecutive exceedance
        if len(self.history[service]) == self.window_size:
            if all(score >= self.threshold for score in self.history[service]):
                return {
                    'alert': True,
                    'service': service,
                    'severity': 'Warning' if risk_score < 0.9 else 'Critical',
                    'message': f"Risk score {risk_score:.2f} exceeded threshold {self.threshold} for {self.window_size} consecutive intervals."
                }
        
        return {'alert': False}
