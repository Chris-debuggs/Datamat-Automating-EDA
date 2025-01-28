import os
import PyPDF2
import json
import traceback

def read_file(file):
    if file.name.endswith("pdf"):
        try:
            pdf_reader = PyPDF2.PdfFileReader(file)
            text = ''
            for page in pdf_reader.pages:  # Corrected the loop to access pages
                text += page.extractText()
            return text
        
        except Exception as e:
            raise Exception(f"Error reading PDF file: {e}")
    
    elif file.name.endswith("txt"):
        return file.read().decode("utf-8")
    
    else:
        raise Exception("Unsupported file type")
    
    def get_table_data(quiz_dict): 
    try:
        if isinstance(quiz_dict, str):
            quiz_dict = json.loads(quiz_dict) 

        # Ensure quiz_dict is a dictionary
        if not isinstance(quiz_dict, dict):
            raise ValueError("Input should be a dictionary or a valid JSON string representing a dictionary.")

        quiz_table_data = []

        for key, value in quiz_dict.items():  # Expecting a dictionary directly
            mcq = value["mcq"]
            options = "  ||  ".join(
                [
                    f"{option} -> {option_value}" for option, option_value in value["options"].items()
                ]
            )
            correct = value["correct"]
            quiz_table_data.append({"MCQ": mcq, "Choices": options, "Correct": correct})

        return quiz_table_data

    except Exception as e:
        traceback.print_exception(type(e), e, e.__traceback__)
        return []