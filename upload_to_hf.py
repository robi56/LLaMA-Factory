import argparse
from huggingface_hub import HfApi

# --- Argument Parsing ---
# This script is designed to be run from the command line.
parser = argparse.ArgumentParser(description="Upload a local model to a private Hugging Face repository.")
parser.add_argument("--model_path", type=str, required=True, help="Path to the directory containing the model files.")
parser.add_argument("--repo_id", type=str, required=True, help="The ID of the Hugging Face repository (e.g., 'your-username/my-private-model').")

args = parser.parse_args()

def upload_model_to_private_repo(model_path: str, repo_id: str):
    """
    Uploads a model from a local directory to a new, private Hugging Face repository.

    Args:
        model_path (str): The path to the local directory with the model files.
        repo_id (str): The ID of the repository to create on the Hub.
                       This must be in the format 'username/repo-name'.
    """
    api = HfApi()

    try:
        # Create a new private repository on the Hugging Face Hub.
        # If it already exists, `exist_ok=True` prevents an error.
        print(f"Creating or checking for repository: {repo_id}")
        api.create_repo(repo_id=repo_id, private=True, exist_ok=True)
        print("Repository created successfully.")

        # Upload the entire folder containing the model and tokenizer files.
        print(f"Uploading files from {model_path} to {repo_id}")
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            commit_message="Initial model upload"
        )
        print(f"Model successfully uploaded to: https://huggingface.co/{repo_id}")

    except Exception as e:
        print(f"An error occurred during the upload process: {e}")
        print("Please ensure you are logged in via 'huggingface-cli login' and have the correct permissions.")

if __name__ == "__main__":
    # Example Usage:
    # python upload_model.py --model_path "./path/to/my/model" --repo_id "my-username/my-cool-private-model"
    upload_model_to_private_repo(args.model_path, args.repo_id)