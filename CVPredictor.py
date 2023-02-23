from flask import Flask, request, render_template , jsonify
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder 
import PyPDF2
import pandas as pd

app = Flask(__name__)

# Load the pre-trained machine learning model
model = joblib.load('model.joblib')

# Load the TfidfVectorizer object used to extract features from the resumes
vectorizer = joblib.load('vectorizer.joblib')

category_names= {
    0:'Advocate',
    1:'Arts',
    2:'Automation Testing',
    3:'Blockchain',
    4:'Business Analyst',
    5:'Civil Engineer',
    6:'Data Science',
    7:'Database',
    8:'DevOps Engineer',
    9:'DotNet Developer',
    10:'ETL Developer',
    11:'Electrical Engineering',
    12:'HR',
    13:'Hadoop',
    14:'Health and fitness',
    15:'Java Developer',
    16:'Mechanical Engineer',
    17:'Network Security Engineer',
    18:'Operations Manager',
    19:'PMO',
    20:'Python Developer',
    21:'SAP Developer',
    22:'Sales',
    23:'Testing',
    24:'Web Designing'
}

@app.route('/', methods=['GET','POST'])
def index():
    category = None
    if request.method == 'POST':
        # Get the resume file from the request
        resume_file = request.files['resume']

        # Extract text from the resume file
        resume_text = extract_text_from_pdf(resume_file)

        # Extract features from the resume
        features = vectorizer.transform([resume_text])

        # Load the dataset into a pandas DataFrame
        df = pd.read_csv('UpdatedResumeDataSet.csv')
        
        # Extract the category labels from the 'category' column
        category_labels = df['Category'].unique().tolist()

        # Create a LabelEncoder object and fit it to your category labels
        label_encoder = LabelEncoder()
        label_encoder.fit(category_labels)

        # Make a prediction using the machine learning model
        prediction = model.predict(features)

        # Decode the numerical category predictions
        category_predictions = label_encoder.inverse_transform(prediction)
        # Set the predicted category
        category = category_predictions[0]

    # Render the index.html template with the predicted category
    return render_template('index.html', category=category)

    # Return a JSON response with the predicted category label
    #return jsonify({'category': category})

def extract_text_from_pdf(resume_file):
    # Read the PDF file using PyPDF2
    pdf_reader = PyPDF2.PdfReader(resume_file)
    num_pages = len(pdf_reader.pages)
    # Extract text from all pages of the PDF file
    text = ''
    for page_num in range(num_pages):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()

    return text

if __name__ == '__main__':
    app.run(debug=True)