import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import gradio as gr

# Load the dataset
file_path = '/content/drive/MyDrive/3rdsem.xlsx'  # Update this path to the correct location of your file
data = pd.read_excel(file_path)

# Select features (internal marks) and target (GPA)
X = data.drop(columns=['S.NO', 'GPA'])
y = data['GPA']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Support Vector Regression (SVR) model
model = SVR(kernel='linear')  # You can try other kernels like 'rbf' or 'poly'
model.fit(X_train, y_train)

# Evaluate the model (optional, for reference)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")

# Function to predict the fourth-semester GPA
def predict_gpa(subject1, subject2, subject3, subject4, subject5, subject6, subject7, subject8, subject9, subject10, subject11):
    """
    Predicts the fourth-semester GPA based on user-defined internal marks for 11 subjects.
    """
    subject_marks = [subject1, subject2, subject3, subject4, subject5, subject6, subject7, subject8, subject9, subject10, subject11]
    predicted_gpa = model.predict([subject_marks])[0]
    return predicted_gpa

# Gradio interface using gr.Number
inputs = [
    gr.Number(value=50, label=f"Subject {i+1} Marks") for i in range(11)
]
output = gr.Textbox(label="Predicted 4th Semester GPA")

gr.Interface(fn=predict_gpa, inputs=inputs, outputs=output, title="4th Semester GPA Predictor", description="Enter your internal marks for the 4th semester subjects to predict your GPA.").launch()
