import numpy as np
from scipy import stats
from typing import Dict, List

class DriftDetector:
    def __init__(self, reference: np.ndarray, feature_names: List[str]):
        self.reference = reference
        self.feature_names = feature_names

    def detect(self, current: np.ndarray, alpha: float = 0.05) -> dict:
        per_feature: Dict[str, dict] = {}
        drifted: List[str] = []
        
        for i, name in enumerate(self.feature_names):
            ref_col = self.reference[:, i]
            cur_col = current[:, i]
            
            ks_stat, p_value = stats.ks_2samp(ref_col, cur_col)
            is_drift = bool(p_value < alpha) 
            
            per_feature[name] = {
                "statistic": float(ks_stat),
                "p_value": float(p_value),
                "drift_detected": is_drift,
            }
            if is_drift:
                drifted.append(name)
                
        return {
            "drift_detected": len(drifted) > 0,
            "n_drifted_features": len(drifted),
            "drifted_features": drifted,
            "per_feature": per_feature,
            "n_samples": int(current.shape[0]),
            "alpha": alpha,
        }