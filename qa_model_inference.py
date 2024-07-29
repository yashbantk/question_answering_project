import os
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# Set your model path here
MODEL_PATH = "models/bert-base-uncased-20percent"

def save_tokenizer(model_path):
    print("Tokenizer not found. Saving pre-trained tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.save_pretrained(model_path)
    print("Tokenizer saved successfully.")
    return tokenizer

def download_and_save_model(model_path):
    print("Downloading and saving pre-trained model...")
    model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")
    model.save_pretrained(model_path)
    print("Model saved successfully.")
    return model

def load_model_and_tokenizer(model_path):
    try:
        print(f"Attempting to load model and tokenizer from: {model_path}")
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        required_files = ['config.json', 'pytorch_model.bin']
        if not all(file in os.listdir(model_path) for file in required_files):
            print("Required model files are missing, downloading the model...")
            model = download_and_save_model(model_path)
        else:
            print("Loading local model...")
            model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        
        tokenizer_files = ['tokenizer_config.json', 'vocab.txt', 'tokenizer.json']
        if not all(file in os.listdir(model_path) for file in tokenizer_files):
            tokenizer = save_tokenizer(model_path)
        else:
            print("Loading local tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        print("Model and tokenizer loaded successfully.")
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return None, None

def answer_question(question, context, tokenizer, model):
    inputs = tokenizer(question, context, return_tensors="pt")
    print(f"Inputs: {inputs}")
    
    try:
        with torch.no_grad():
            outputs = model(**inputs)
        print(f"Outputs: {outputs}")

        if 'start_logits' in outputs and 'end_logits' in outputs:
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            print("Start logits:", start_logits)
            print("End logits:", end_logits)

            answer_start = torch.argmax(start_logits)
            answer_end = torch.argmax(end_logits) + 1
            
            print(f"Answer start index: {answer_start}")
            print(f"Answer end index: {answer_end}")

            # Ensure the answer positions are within the valid range
            if answer_start >= len(inputs.input_ids[0]) or answer_end > len(inputs.input_ids[0]) or answer_start > answer_end:
                print("Invalid start or end index detected.")
                answer = "Unable to extract a valid answer."
            else:
                answer_tokens = inputs.input_ids[0][answer_start:answer_end]
                answer_tokens = [token for token in answer_tokens if token != tokenizer.sep_token_id and token != tokenizer.pad_token_id]
                answer = tokenizer.decode(answer_tokens)
                print(f"Answer tokens: {answer_tokens}")
                print(f"Answer: {answer}")
        else:
            print("Unexpected model output format. Please check model architecture.")
            answer = "Unable to extract answer due to unexpected model output format."
        
        return answer
    except Exception as e:
        print(f"Error during inference: {e}")
        return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    print("Welcome to the QA Model Inference Tool")
    print("--------------------------------------")

    tokenizer, model = load_model_and_tokenizer(MODEL_PATH)
    
    if tokenizer is None or model is None:
        print("Failed to load the model or tokenizer. Please check the MODEL_PATH and ensure all required files are present.")
        exit()

    print("\nModel and tokenizer loaded successfully. You can now start asking questions.")
    while True:
        question = input("\nEnter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
        
        context = input("Enter the context: ")
        
        answer = answer_question(question, context, tokenizer, model)
        print(f"\nAnswer: {answer}")

    print("\nThank you for using the QA model!")
