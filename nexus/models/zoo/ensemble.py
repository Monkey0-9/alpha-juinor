class AIEnsembleBrain:
    """
    Coordinates multiple models to make a unified trading decision.
    """
    def __init__(self):
        self.models = []
        
    def add_model(self, model, weight=1.0):
        self.models.append({'model': model, 'weight': weight})
        
    def get_signal(self, features):
        """
        Aggregates signals from all models based on weight.
        Returns final signal: 1 (Buy), -1 (Sell), 0 (Hold)
        """
        if not self.models:
            return 0
            
        total_signal = 0
        total_weight = 0
        
        for item in self.models:
            model = item['model']
            weight = item['weight']
            signal = model.predict(features)
            
            total_signal += (signal * weight)
            total_weight += weight
            
        if total_weight == 0:
            return 0
            
        aggregated = total_signal / total_weight
        
        # Thresholding for final decision
        if aggregated > 0.3:
            return 1
        elif aggregated < -0.3:
            return -1
            
        return 0
