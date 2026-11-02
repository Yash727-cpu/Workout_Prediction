# Workout Recommendation System using KNN and K-Means

## What it does
- Takes fitness goal, fitness level, available time, and equipment as input.
- Uses preprocessing (one-hot encoding and scaling).
- Uses K-Means to group similar workout profiles.
- Uses K-Nearest Neighbors (KNN) to recommend the closest workout records.
- Displays recommendations in a Streamlit interface.

## Requirements
- Python 3.10 or newer (Python 3.11/3.12 is a safe choice for student projects)
- VS Code

## Run in VS Code on Windows
1. Extract this ZIP to a folder, for example `workout_recommendation_knn_kmeans`.
2. Open that folder in VS Code: **File > Open Folder**.
3. Open **Terminal > New Terminal**.
4. Create a virtual environment:
   ```powershell
   py -m venv .venv
   ```
5. Activate it in PowerShell:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   If activation is blocked, use the Command Prompt terminal and run:
   ```bat
   .venv\Scripts\activate.bat
   ```
6. Install packages:
   ```powershell
   python -m pip install -r requirements.txt
   ```
7. Run the app:
   ```powershell
   python -m streamlit run app.py
   ```
8. A browser tab should open. Choose preferences and click **Recommend workouts**.

## Project workflow (for your report)
1. Select the problem: users need help choosing suitable workouts.
2. Define objectives: personalize workout suggestions using KNN and K-Means.
3. Dataset: `workouts.csv` (sample dataset included).
4. Understand data: inspect goal, level, duration, equipment, workout, calories, and intensity.
5. Preprocess: encode categorical features and scale workout duration.
6. Select algorithms: K-Means clustering and KNN nearest-neighbor search.
7. Split data: for this simple nearest-neighbor recommendation demo, the model uses the workout catalogue as its reference set; it does not use a supervised train/test split because there is no labeled target column. Explain this honestly in the presentation.
8. Train/model fit: fit preprocessing, K-Means, and KNN on the workout catalogue.
9. Test: enter several different preference combinations and check whether results are sensible.
10. Evaluate: discuss relevance of top recommendations and test different inputs. Standard classification accuracy is not appropriate because this is a recommendation task without labeled correct answers.
11. Predict new data: the app accepts new user preferences.
12. Interface: Streamlit form and recommendation cards.

## Suggested evaluation for tomorrow
Test at least these cases:
- Weight Loss + Beginner + 20 minutes + No Equipment
- Muscle Gain + Intermediate + 30 minutes + Dumbbells
- Flexibility + Beginner + 15 minutes + Yoga Mat
Record whether the displayed workouts match the selected goal, level, time, and equipment. Since the data is a small sample catalogue, describe this as a prototype and do not claim clinical or real-world accuracy.

## Limitations
- The CSV is a small sample dataset created for demonstration.
- Estimated calories are approximate.
- Recommendations are educational and are not medical or personal training advice.
- A larger real dataset and user feedback would be needed for a rigorous evaluation.
