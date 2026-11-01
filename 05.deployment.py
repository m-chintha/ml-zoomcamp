
import pickle
import pandas as pd

# Lead data
lead = {
    "lead_source": "paid_ads",
    "industry": "technology",
    "employment_status": "employed",
    "location": "north_america",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0,
    "interaction_count": 4,
    "lead_score": 0.41
}

# Load the trained pipeline
with open("pipeline.bin", "rb") as file:
    pipeline = pickle.load(file)

# Convert lead data into a DataFrame
#X = pd.DataFrame([lead])
X = [lead]


# Predict probabilities
probabilities = pipeline.predict_proba(X)

# Display results
print("Prediction probabilities:", probabilities)

# Display probability for each class
for class_label, probability in zip(
    pipeline.classes_,
    probabilities[0]
):
    print(f"{class_label}: {probability:.4f}")
