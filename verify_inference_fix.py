import sys
import os
from pathlib import Path
import json

# Add backend directory to path
sys.path.append(os.path.join(os.getcwd(), "backend"))

from backend.ml_service import MLService

def test_inference():
    print("Initializing ML Service...")
    ml_service = MLService()
    if not ml_service.load_artifacts():
        print("Failed to load artifacts")
        return

    # Test Case 1: Low Risk (Healthy Control)
    # Young, Female, High Edu, No complaints
    input_low = {
        "age": 55,
        "sex": "Female",
        "education_years": 18,
        "adl_assistance": 0,
        "financial_difficulty": 0,
        "navigation_difficulty": 0,
        "appointment_memory": 0,
        "daily_memory_problems": 0,
        "repetition_issues": 0,
        "temporal_disorientation": 0,
        "judgment_problems": 0,
        "reduced_interest": 0,
        "learning_difficulty": 0
    }

    # Test Case 2: Broadly Symptomatic (High Risk)
    # Old, Male, Low Edu, Multiple complaints
    input_high = {
        "age": 82,
        "sex": "Male",
        "education_years": 10,
        "adl_assistance": 1, # Some help needed
        "financial_difficulty": 1,
        "navigation_difficulty": 1,
        "appointment_memory": 1,
        "daily_memory_problems": 1,
        "repetition_issues": 1,
        "temporal_disorientation": 1,
        "judgment_problems": 1,
        "reduced_interest": 1,
        "learning_difficulty": 1
    }
    
    # Test Case 3: Middle Ground
    input_mid = {
        "age": 72,
        "sex": "Female",
        "education_years": 14,
        "adl_assistance": 0,
        "daily_memory_problems": 1, # Mild memory issues
        "repetition_issues": 1
    }

    print("\n--- Running Inference Tests ---")
    
    # Run predictions
    res_low = ml_service.predict_risk(input_low)
    print(f"\nProfile 1 (Low Risk - Age 55, Healthy):")
    print(f"  Probability: {res_low['probability']:.4f}")
    print(f"  Score: {res_low['risk_score']}")
    print(f"  Factors: {json.dumps(res_low['contributing_factors'][:2], default=str)}")

    res_high = ml_service.predict_risk(input_high)
    print(f"\nProfile 2 (High Risk - Age 82, Symptomatic):")
    print(f"  Probability: {res_high['probability']:.4f}")
    print(f"  Score: {res_high['risk_score']}")
    
    res_mid = ml_service.predict_risk(input_mid)
    print(f"\nProfile 3 (Medium Risk - Age 72, Mild Symptoms):")
    print(f"  Probability: {res_mid['probability']:.4f}")
    print(f"  Score: {res_mid['risk_score']}")

    # Validation
    if res_high['probability'] > res_low['probability']:
        print("\n✅ SUCCESS: High risk profile has higher probability than low risk.")
    else:
        print("\n❌ FAILURE: High risk probability is NOT higher than low risk.")
        
    if res_low['probability'] == res_mid['probability'] == res_high['probability']:
        print("❌ FAILURE: All probabilities are identical (Constant Output Bug Persists).")
    else:
        print("✅ SUCCESS: Probabilities vary with input.")

if __name__ == "__main__":
    test_inference()
