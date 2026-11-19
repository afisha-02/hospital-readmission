"""
Cost-sensitive decision analysis and business cost matrix for readmission risk.
Calculates expected financial/operational impact across decision thresholds.
"""

from typing import Dict, Any

class CostMatrix:
    """
    Configurable cost matrix for healthcare readmission decision economics.
    
    DISCLAIMER:
    Illustrative decision-cost framework for portfolio demonstration.
    Does NOT represent exact institutional hospital accounting or reimbursement contracts.
    """
    
    def __init__(
        self,
        cost_fn: float = 5000.0,  # Unplanned readmission, complications, penalty
        cost_fp: float = 500.0,   # Proactive post-discharge outreach, clinical review
        cost_tp: float = 750.0,   # Intervention cost + partial readmission mitigation
        cost_tn: float = 0.0      # Standard safe discharge without excess resource spend
    ):
        self.cost_fn = float(cost_fn)
        self.cost_fp = float(cost_fp)
        self.cost_tp = float(cost_tp)
        self.cost_tn = float(cost_tn)

    def calculate_cost(self, tn: int, fp: int, fn: int, tp: int) -> Dict[str, float]:
        """Calculates total expected cost and per-patient average cost."""
        total_cost = (
            fn * self.cost_fn +
            fp * self.cost_fp +
            tp * self.cost_tp +
            tn * self.cost_tn
        )
        total_patients = tn + fp + fn + tp
        per_patient_cost = total_cost / total_patients if total_patients > 0 else 0.0
        
        return {
            "total_cost": round(float(total_cost), 2),
            "per_patient_cost": round(float(per_patient_cost), 2),
            "fn_cost_burden": round(float(fn * self.cost_fn), 2),
            "fp_cost_burden": round(float(fp * self.cost_fp), 2),
            "tp_cost_burden": round(float(tp * self.cost_tp), 2),
            "tn_cost_burden": round(float(tn * self.cost_tn), 2)
        }
