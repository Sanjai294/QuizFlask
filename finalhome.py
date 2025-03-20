from flask import Flask, request, jsonify
import google.generativeai as genai
import json
import re
from flask_cors import CORS
import os
import io
import firebase_admin
from firebase_admin import credentials, storage

app = Flask(__name__)
CORS(app)

# Firebase Configuration
SERVICE_ACCOUNT_KEY = 'serviceAccountKey.json'
BUCKET_NAME = 'quip-664c9.firebasestorage.app'  # Replace with your actual bucket name

# Initialize Firebase
cred = credentials.Certificate(SERVICE_ACCOUNT_KEY)
firebase_admin.initialize_app(cred, {
    'storageBucket': BUCKET_NAME
})
bucket = storage.bucket()

# Log the actual bucket name for confirmation
print(f"Connected to Firebase Storage bucket: {bucket.name}")

# Gemini API Configuration
API_KEY = "AIzaSyAx-Ezi4I5ltiHT_gyWPJvTYvZqzZwMKOc"
genai.configure(api_key=API_KEY)

generation_config = {
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 8192,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
    safety_settings=safety_settings
)

def get_files_from_firebase_path(path):
    """
    Fetches all text files from the specified path in Firebase Storage
    
    Args:
        path: String path in Firebase Storage
        
    Returns:
        Concatenated text content of all files in the path
    """
    print(f"Fetching files from Firebase Storage path: {path}")
    
    # List all blobs with the specified prefix
    blobs = bucket.list_blobs(prefix=path)
    
    text_data = ""
    file_count = 0
    
    for blob in blobs:
        # Skip if it's a directory (ends with /)
        if blob.name.endswith('/'):
            continue
            
        # Skip if it's not a text file (simple check for .txt extension)
        if not blob.name.endswith('.txt'):
            continue
            
        print(f"Processing file: {blob.name}")
        
        try:
            # Download the content
            content = blob.download_as_string().decode('utf-8')
            file_count += 1
            
            # Add content to our text data with file name as header
            filename = blob.name.split('/')[-1]
            text_data += f"--- {filename} ---\n{content}\n\n"
            print(f"Successfully downloaded {len(content)} characters from {filename}")
        except Exception as e:
            print(f"Error downloading file {blob.name}: {str(e)}")
    
    print(f"Retrieved {file_count} files with total {len(text_data)} characters")
    return text_data

def fetch_text_files_from_firebase(college, department, semester, subject, unit):
    """
    Builds the Firebase Storage path and fetches text content
    
    Args:
        college, department, semester, subject, unit: Path components
        
    Returns:
        Concatenated text content of all files in the target directory
    """
    # Ensure semester is correctly formatted
    if semester and not semester.startswith("Semester "):
        semester = f"Semester {semester}"
    
    # Build the path for Firebase Storage
    path_components = [
        'data',
        college,
        department,
        semester,
        subject,
        unit
    ]
    
    # Clean path components (remove any empty strings)
    path_components = [p for p in path_components if p]
    
    # Create path string (Firebase uses forward slashes)
    path = '/'.join(path_components)
    if not path.endswith('/'):
        path += '/'
        
    print(f"Constructed Firebase path: {path}")
    
    # Get files from this path
    return get_files_from_firebase_path(path)

def format_prompt(subject: str, unit: str, context: str) -> str:
    return f"""Using the following context about {subject} - {unit}, generate 20 quiz questions.

Context:
{context}

Guidelines:
1. The questions should be relevant to {unit} in {subject}.
2. Include exactly 4 options per question.
3. Only one option should be correct.
4. Provide a detailed explanation for the correct answer.
5. Mark difficulty as 'Easy', 'Medium', or 'Hard'.
6. Include a short hint that helps guide towards the answer without giving it away.

Return response in this exact JSON format:
{{
    "questions": [
        {{
            "question": "Question text here",
            "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
            "correct_answer": "Option X",
            "explanation": "Explanation here",
            "difficulty": "Easy/Medium/Hard",
            "hint": "A helpful hint here"
        }}
    ]
}}

Rules:
1. Generate exactly 20 questions
2. Make questions progressively harder
3. Base questions on the provided context
4. Ensure hints don't directly give away the answer
5. Make questions appropriate for university-level learning

Return only valid JSON without any additional text or markdown."""

def clean_response(response_text: str) -> dict:
    """Cleans and formats the AI response."""
    try:
        # Remove markdown JSON code blocks if present
        cleaned_text = re.sub(r"```json\n(.*)\n```", r"\1", response_text, flags=re.DOTALL)
        cleaned_text = re.sub(r"```\n(.*)\n```", r"\1", cleaned_text, flags=re.DOTALL)
        
        # Find a JSON object in the response
        json_match = re.search(r'\{[\s\S]*\}', cleaned_text)
        if not json_match:
            raise ValueError("No JSON object found in response")
        
        json_str = json_match.group(0)
        response_data = json.loads(json_str)
        
        # Validate the questions data
        if 'questions' not in response_data:
            raise ValueError("No questions array found in response")
            
        questions = response_data['questions']
        
        if not questions or not isinstance(questions, list):
            raise ValueError(f"Questions must be a non-empty array, got: {type(questions)}")
        
        # Validate each question has required fields
        required_fields = {'question', 'options', 'correct_answer', 'explanation', 'difficulty', 'hint'}
        for i, q in enumerate(questions):
            if not isinstance(q, dict):
                raise ValueError(f"Question {i+1} is not a valid object")
                
            missing_fields = required_fields - set(q.keys())
            if missing_fields:
                raise ValueError(f"Question {i+1} missing required fields: {missing_fields}")
            
            # Validate options is a list with 4 items
            if not isinstance(q['options'], list) or len(q['options']) != 4:
                raise ValueError(f"Question {i+1} options must be a list with 4 items")
                
            # Validate correct_answer is one of the options
            if q['correct_answer'] not in q['options']:
                raise ValueError(f"Question {i+1} correct_answer must be one of the options")
        
        return response_data
    except Exception as e:
        print(f"Error cleaning response: {e}")
        raise

@app.route('/quiz', methods=['POST'])
def generate_questions():
    try:
        data = request.get_json()
        print("\nReceived request from Dart:", data)
        
        required_fields = ['subject', 'unit']
        for field in required_fields:
            if not data.get(field):
                print(f"Missing field: {field}")
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Extract path components
        college = data.get('college', '')
        department = data.get('department', '')
        semester = data.get('semester', '')
        subject = data.get('subject', '')
        unit = data.get('unit', '')
        
        # Use Firebase Storage instead of Google Drive
        text_data = fetch_text_files_from_firebase(
            college, department, semester, subject, unit
        )
        
        if not text_data:
            print("No text data found in Firebase Storage.")
            return jsonify({'error': 'No text data found in Firebase Storage'}), 500
        
        print(f"Retrieved {len(text_data)} characters of text data")
        
        # If text data is very large, print only the beginning for debugging
        if len(text_data) > 500:
            print(f"First 500 characters: {text_data[:500]}...")
        else:
            print(f"Text data: {text_data}")

        prompt = format_prompt(data['subject'], data['unit'], text_data)
        print("Generated prompt:", prompt)

        response = model.generate_content(prompt)
        
        if not response or not response.text:
            print("No response from AI model.")
            return jsonify({'error': 'No response from AI model'}), 500
        
        print("AI response received, length:", len(response.text))
        
        formatted_response = clean_response(response.text)
        
        print("Formatted response successfully generated.")
        
        # Format response to match the expected structure in quiz_service.dart
        # The client expects a string that can be parsed to JSON with a "questions" array
        response_for_client = {
            "files": [
                {
                    "name": f"{subject}_{unit}_questions.json",
                    "content": json.dumps({"questions": formatted_response["questions"]})
                }
            ]
        }
        
        return jsonify(response_for_client)
        
    except Exception as e:
        print(f"Error generating questions: {str(e)}")
        return jsonify({'error': f'Error generating questions: {str(e)}'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    print("Starting Quiz Generator Backend using Firebase Storage...")
    app.run(host='0.0.0.0', port=5000, debug=True)