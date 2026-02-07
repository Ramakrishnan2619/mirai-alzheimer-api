"""
MirAI Backend - ML Inference Service
Loads Stage-1 XGBoost model artifacts and performs risk prediction
"""
import json
import pickle
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import xgboost as xgb

# Path to model artifacts
MODELS_DIR = Path(__file__).parent / "models"

# Feature order expected by the XGBoost model (6 features based on model structure)
# These map to questionnaire responses that get aggregated into model features
FEATURE_ORDER = [
    "AGE",
    "PTGENDER",
    "PTEDUCAT",
    "FAQ",
    "EcogPtMem",
    "EcogPtTotal"
]

# Questionnaire questions that map to each model feature
QUESTION_TO_FEATURE_MAP = {
    "cognitive_score": [
        "judgment_problems", "reduced_interest", "repetition_issues",
        "learning_difficulty", "temporal_disorientation", "financial_difficulty",
        "appointment_memory", "daily_memory_problems"
    ],
    "functional_score": ["adl_assistance", "navigation_difficulty"],
    "age_normalized": ["age"],
    "family_risk_score": ["family_history_dementia", "family_relation_degree"],
    "cardiovascular_score": ["diabetes", "hypertension", "stroke_history", "high_cholesterol"],
    "lifestyle_score": ["physical_activity", "smoking_status", "sleep_hours", "depression", "head_injury"]
}

# Feature importance for explanations
FEATURE_IMPORTANCE = {
    "cognitive_score": 0.30,
    "functional_score": 0.15,
    "age_normalized": 0.20,
    "family_risk_score": 0.15,
    "cardiovascular_score": 0.12,
    "lifestyle_score": 0.08
}

# Human-readable feature names
FEATURE_LABELS = {
    "cognitive_score": "Cognitive Function Indicators",
    "functional_score": "Daily Living Function",
    "age_normalized": "Age Factor",
    "family_risk_score": "Family History Risk",
    "cardiovascular_score": "Cardiovascular Health",
    "lifestyle_score": "Lifestyle Factors"
}

# Question labels for explanation
QUESTION_LABELS = {
    "age": "Age",
    "sex": "Biological Sex",
    "education_years": "Education Level",
    "employment_status": "Employment Status",
    "family_history_dementia": "Family History of Dementia",
    "family_relation_degree": "Close Relative with Dementia",
    "judgment_problems": "Difficulty with Judgment/Decisions",
    "reduced_interest": "Reduced Interest in Activities",
    "repetition_issues": "Repeating Questions/Stories",
    "learning_difficulty": "Difficulty Learning New Things",
    "temporal_disorientation": "Forgetting Date/Time",
    "financial_difficulty": "Trouble with Finances",
    "appointment_memory": "Forgetting Appointments",
    "daily_memory_problems": "Daily Memory Problems",
    "adl_assistance": "Need Assistance with Daily Activities",
    "navigation_difficulty": "Getting Lost in Familiar Places",
    "diabetes": "Diabetes",
    "hypertension": "High Blood Pressure",
    "stroke_history": "History of Stroke",
    "physical_activity": "Physical Activity Level",
    "smoking_status": "Smoking History",
    "sleep_hours": "Sleep Duration",
    "depression": "Depression",
    "head_injury": "History of Head Injury",
    "high_cholesterol": "High Cholesterol"
}


class MLService:
    """Machine Learning inference service for Stage-1 risk assessment using XGBoost"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.imputer = None
        self.is_loaded = False
    
    def load_artifacts(self):
        """Load XGBoost model, scaler, and imputer from disk"""
        try:
            # Load XGBoost model from JSON
            model_path = MODELS_DIR / "stage1_model.json"
            self.model = xgb.Booster()
            self.model.load_model(str(model_path))
            
            # Load scaler
            scaler_path = MODELS_DIR / "stage1_scaler.pkl"
            try:
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            except Exception:
                # Fallback to joblib if pickle fails
                self.scaler = joblib.load(scaler_path)
            
            # Load imputer
            imputer_path = MODELS_DIR / "stage1_imputer.pkl"
            try:
                with open(imputer_path, 'rb') as f:
                    self.imputer = pickle.load(f)
            except Exception:
                # Fallback to joblib if pickle fails
                self.imputer = joblib.load(imputer_path)
            
            self.is_loaded = True
            print("✓ ML artifacts loaded successfully (XGBoost model)")
            return True
            
        except Exception as e:
            print(f"✗ Error loading ML artifacts: {e}")
            import traceback
            traceback.print_exc()
            self.is_loaded = False
            return False
    
    def _aggregate_to_model_features(self, responses: Dict[str, Any]) -> np.ndarray:
        """
        Aggregate questionnaire responses into the 6 features expected by the new XGBoost model:
        ['AGE', 'PTGENDER', 'PTEDUCAT', 'FAQ', 'EcogPtMem', 'EcogPtTotal']
        """
        features = []
        
        # 1. AGE (Numeric)
        try:
            age = float(responses.get("age", 65))
        except (ValueError, TypeError):
            age = 65.0
        features.append(age)
        
        # 2. PTGENDER (Male=1, Female=0)
        sex_str = str(responses.get("sex", "Female")).lower()
        sex_val = 1.0 if sex_str in ["male", "m", "1"] else 0.0
        features.append(sex_val)
        
        # 3. PTEDUCAT (Education Years)
        try:
            edu = float(responses.get("education_years", 12))
        except (ValueError, TypeError):
            edu = 12.0
        features.append(edu)
        
        # 4. FAQ (Functional Activities Questionnaire) - Sum of ADL/IADL difficulties
        # We approximate specific questions to FAQ items (usually 0-3 scale, here mapped from inputs)
        faq_score = 0.0
        
        # Map specific inputs to FAQ-like score (0=Normal, >0=Impaired)
        # Using 0-3 scale approximation: 0=None, 1=Some, 2=A lot, 3=Dependent
        if "adl_assistance" in responses: # 0, 1, 2 -> Map to 0, 5, 10 weight? 
            # adl_assistance is usually 0=No, 1=Yes/Sometimes, 2=Yes/Always
            val = float(responses["adl_assistance"])
            faq_score += val * 5.0 # Weight heavily as it implies dependency
            
        if "financial_difficulty" in responses:
            faq_score += float(responses["financial_difficulty"]) * 2.0
            
        if "navigation_difficulty" in responses:
            faq_score += float(responses["navigation_difficulty"]) * 2.0
            
        if "shopping_difficulty" in responses: # If present
            faq_score += float(responses["shopping_difficulty"]) * 2.0
            
        features.append(faq_score)
        
        # 5. EcogPtMem (Everyday Cognition - Memory) - Average of memory items (1-4 scale)
        # Questions: appointment_memory, daily_memory_problems, repetition_issues, temporal_disorientation
        mem_items = ["appointment_memory", "daily_memory_problems", "repetition_issues", "temporal_disorientation"]
        mem_values = []
        for q in mem_items:
            if q in responses:
                try:
                    # Inputs are usually 0=No, 1=Yes. We map to 1-4 scale?
                    # Or just use raw sums if model used raw sums?
                    # The prompt says EcogPtMem is "Score". Usually 1 (Better) to 4 (Worse).
                    # If input is 0/1 binary: 0->1(Normal), 1->3(Concern).
                    val = float(responses[q])
                    mem_values.append(1.0 + (val * 2.0)) # Map 0->1, 1->3
                except (ValueError, TypeError):
                    pass
        
        # Default to 1.0 (Normal) if no inputs
        ecog_mem = np.mean(mem_values) if mem_values else 1.0
        features.append(ecog_mem)
        
        # 6. EcogPtTotal (Everyday Cognition - Total) - Average of ALL cognitive items
        # Include Memory items + judgment, interest, learning
        other_items = ["judgment_problems", "reduced_interest", "learning_difficulty"]
        all_values = list(mem_values) # Start with memory values
        
        for q in other_items:
            if q in responses:
                try:
                    val = float(responses[q])
                    all_values.append(1.0 + (val * 2.0)) # Map 0->1, 1->3
                except (ValueError, TypeError):
                    pass
                    
        ecog_total = np.mean(all_values) if all_values else 1.0
        features.append(ecog_total)
        
        return np.array(features).reshape(1, -1)
    
    def predict_risk(self, responses: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run full inference pipeline:
        1. Aggregate questionnaire responses to model features
        2. Apply imputer for missing values
        3. Apply scaler for normalization
        4. Run XGBoost model inference
        
        Returns dict with risk_score, probability, category, contributing_factors
        """
        if not self.is_loaded:
            raise RuntimeError("ML artifacts not loaded")
        
        # Step 1: Aggregate to model features
        features = self._aggregate_to_model_features(responses)
        
        # Step 2: Apply imputer (handle missing values)
        features_imputed = self.imputer.transform(features)
        
        # Step 3: Apply scaler (normalize)
        features_scaled = self.scaler.transform(features_imputed)
        
        # Step 4: XGBoost inference
        dmatrix = xgb.DMatrix(features_scaled)
        probability = float(self.model.predict(dmatrix)[0])
        
        # Ensure probability is in valid range
        probability = max(0.0, min(1.0, probability))
        
        # Convert to risk score (0-100)
        risk_score = probability * 100
        
        # Determine risk category
        if probability < 0.3:
            risk_category = "low"
        elif probability < 0.6:
            risk_category = "moderate"
        else:
            risk_category = "elevated"
        
        # Identify contributing factors
        contributing_factors = self._identify_contributing_factors(
            responses, features_scaled[0]
        )
        
        return {
            "risk_score": round(risk_score, 1),
            "probability": round(probability, 4),
            "risk_category": risk_category,
            "contributing_factors": contributing_factors
        }
    
    def _identify_contributing_factors(
        self, 
        responses: Dict[str, Any],
        scaled_features: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Identify which factors contribute most to the risk score"""
        contributions = []
        
        for i, feature_name in enumerate(FEATURE_ORDER):
            # Calculate contribution based on scaled feature and importance
            feature_value = scaled_features[i]
            importance = FEATURE_IMPORTANCE.get(feature_name, 0.1)
            contribution = feature_value * importance
            
            # Only include significant contributions
            if contribution > 0.01:
                contributions.append({
                    "feature": feature_name,
                    "label": FEATURE_LABELS.get(feature_name, feature_name),
                    "value": round(float(feature_value), 2),
                    "contribution": round(float(contribution), 3),
                    "importance": importance
                })
        
        # Sort by contribution (highest first)
        contributions.sort(key=lambda x: x["contribution"], reverse=True)
        
        # Return top 5 contributing factors
        return contributions[:5]


# Singleton instance
ml_service = MLService()


def get_ml_service() -> MLService:
    """Get the ML service instance"""
    return ml_service
