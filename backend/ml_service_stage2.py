"""
MirAI Backend - Stage-2 ML Inference Service
Loads Stage-2 XGBoost model for genetic risk assessment
"""
import json
import pickle
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any
import xgboost as xgb

# Path to model artifacts
MODELS_DIR = Path(__file__).parent / "models"

# Stage-2 uses 2 features for genetic risk (Cascade Model)
STAGE2_FEATURES = [
    "Stage1_Prob",          # Probability from Stage 1 Clinical Model
    "APOE4_Count"           # 0, 1, or 2 copies of APOE ε4 allele
]

# Feature labels for explanation
STAGE2_FEATURE_LABELS = {
    "Stage1_Prob": "Clinical Risk Score (Stage 1)",
    "APOE4_Count": "APOE ε4 Allele Count"
}


class Stage2MLService:
    """Machine Learning inference service for Stage-2 genetic risk assessment"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.imputer = None
        self.is_loaded = False
    
    def load_artifacts(self):
        """Load Stage-2 XGBoost model, scaler, and imputer from disk"""
        try:
            # Load XGBoost model from JSON
            model_path = MODELS_DIR / "stage2_model.json"
            if not model_path.exists():
                print(f"⚠ Stage-2 model not found at {model_path}")
                return False
                
            self.model = xgb.Booster()
            self.model.load_model(str(model_path))
            
            # Load scaler (try pickle first, fallback to joblib)
            scaler_path = MODELS_DIR / "stage2_scaler.pkl"
            try:
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            except Exception:
                self.scaler = joblib.load(scaler_path)
            
            # Load imputer (try pickle first, fallback to joblib)
            imputer_path = MODELS_DIR / "stage2_imputer.pkl"
            try:
                with open(imputer_path, 'rb') as f:
                    self.imputer = pickle.load(f)
            except Exception:
                self.imputer = joblib.load(imputer_path)
            
            self.is_loaded = True
            print("✓ Stage-2 ML artifacts loaded successfully")
            return True
            
        except Exception as e:
            print(f"✗ Error loading Stage-2 ML artifacts: {e}")
            import traceback
            traceback.print_exc()
            self.is_loaded = False
            return False
    
    def prepare_features(self, inputs: Dict[str, Any]) -> np.ndarray:
        """
        Convert genetic inputs to feature vector
        
        Args:
            inputs: Dict with Stage1_Prob and APOE4_Count
            
        Returns:
            numpy array of features [Stage1_Prob, APOE4_Count]
        """
        features = []
        
        # 1. Stage1_Prob
        s1_prob = inputs.get("Stage1_Prob")
        if s1_prob is not None:
            features.append(float(s1_prob))
        else:
            features.append(np.nan)
            
        # 2. APOE4_Count
        apoe = inputs.get("APOE4_Count")
        if apoe is not None:
            features.append(float(apoe))
        else:
            features.append(np.nan)
        
        return np.array(features).reshape(1, -1)
    
    def predict_risk(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run Stage-2 genetic risk inference:
        1. Convert inputs to feature vector
        2. Apply imputer for missing values
        3. Apply scaler for normalization
        4. Run XGBoost inference
        
        Returns dict with probability, category, and feature analysis
        """
        if not self.is_loaded:
            raise RuntimeError("Stage-2 ML artifacts not loaded")
        
        # Step 1: Prepare feature vector
        features = self.prepare_features(inputs)
        
        # Step 2: Apply imputer (handle missing values)
        features_imputed = self.imputer.transform(features)
        
        # Step 3: Apply scaler (normalize)
        features_scaled = self.scaler.transform(features_imputed)
        
        # Step 4: XGBoost inference
        dmatrix = xgb.DMatrix(features_scaled)
        probability = float(self.model.predict(dmatrix)[0])
        
        # Ensure probability is in valid range
        probability = max(0.0, min(1.0, probability))
        
        # Determine risk category
        if probability < 0.33:
            risk_category = "low"
        elif probability < 0.66:
            risk_category = "moderate"
        else:
            risk_category = "high"
        
        # Feature contribution analysis
        # Feature contribution analysis
        feature_analysis = []
        apoe_count = inputs.get("APOE4_Count", 0)
        s1_prob = inputs.get("Stage1_Prob", 0.0)
        
        # APOE ε4 contribution
        apoe_risk_level = "low" if apoe_count == 0 else ("moderate" if apoe_count == 1 else "high")
        feature_analysis.append({
            "feature": "APOE4_Count",
            "label": "APOE ε4 Allele Count",
            "value": apoe_count,
            "description": f"{apoe_count} cop{'y' if apoe_count == 1 else 'ies'} of ε4 allele",
            "risk_level": apoe_risk_level
        })
        
        # Stage 1 Risk contribution
        s1_risk_level = "low" if s1_prob < 0.3 else ("moderate" if s1_prob < 0.6 else "high")
        feature_analysis.append({
            "feature": "Stage1_Prob",
            "label": "Stage 1 Clinical Risk",
            "value": round(s1_prob, 3),
            "description": f"Clinical Probability: {s1_prob:.3f}",
            "risk_level": s1_risk_level
        })
        
        return {
            "probability": round(probability, 4),
            "risk_score": round(probability * 100, 1),
            "risk_category": risk_category,
            "feature_analysis": feature_analysis
        }


# Singleton instance
stage2_ml_service = Stage2MLService()


def get_stage2_ml_service() -> Stage2MLService:
    """Get the Stage-2 ML service instance"""
    return stage2_ml_service
