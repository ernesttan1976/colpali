import os
import spaces
import base64
from io import BytesIO

import gradio as gr
import torch

from pdf2image import convert_from_path, convert_from_bytes
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from colpali_engine.models import ColQwen2, ColQwen2Processor

import sys

from db import DocumentEmbeddingDatabase

# Create the directory for the embeddings database if it doesn't exist
os.makedirs("./data/embeddings_db", exist_ok=True)

# Initialize the database - this connects to the local file-based database
db = DocumentEmbeddingDatabase(db_path="./data/embeddings_db")

print(f"Python path: {sys.executable}")
print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch path: {torch.__file__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# Define model paths
MODEL_DIR = "./models/colqwen2"
MODEL_PATH = os.path.join(MODEL_DIR, "model")
PROCESSOR_PATH = os.path.join(MODEL_DIR, "processor")

PAGE_LIMIT=150

@spaces.GPU
def install_fa2():
    print("Install FA2")
    os.system("pip install flash-attn --no-build-isolation")
# install_fa2()


# Replace the load_model function in app.py with this corrected version
def load_model():
    """Load model from disk if available, otherwise download and save it."""
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        print(f"Model directory: {MODEL_DIR}")
        print(f"Model directory exists: {os.path.exists(MODEL_DIR)}")
        
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        
        # Check more thoroughly if model exists
        model_files_exist = os.path.exists(os.path.join(MODEL_PATH, "config.json"))
        processor_files_exist = os.path.exists(os.path.join(PROCESSOR_PATH, "config.json"))
        print(f"Model config exists: {model_files_exist}")
        print(f"Processor config exists: {processor_files_exist}")
        
        # List files in model directory to debug
        if os.path.exists(MODEL_PATH):
            print(f"Files in model directory: {os.listdir(MODEL_PATH)}")
        if os.path.exists(PROCESSOR_PATH):
            print(f"Files in processor directory: {os.listdir(PROCESSOR_PATH)}")
        
        # Only attempt to load if critical files exist
        if model_files_exist and processor_files_exist:
            print("Loading model from disk - step 1...")
            try:
                # Use absolute paths to avoid any reference issues
                abs_model_path = os.path.abspath(MODEL_PATH)
                abs_processor_path = os.path.abspath(PROCESSOR_PATH)
                print(f"Using absolute model path: {abs_model_path}")
                
                # Load model with trust_remote_code
                print("Loading model from disk - step 2...")
                model = ColQwen2.from_pretrained(
                    abs_model_path,
                    torch_dtype=torch.bfloat16,
                    device_map=device,
                    local_files_only=False,
                    trust_remote_code=True,
                    revision=None  # Important: don't try to fetch remote info
                )
                print("Model loaded successfully!")
                
                print("Loading processor...")
                processor = ColQwen2Processor.from_pretrained(
                    abs_processor_path,
                    local_files_only=True,
                    trust_remote_code=True,
                    revision=None
                )
                print("Processor loaded successfully!")
                
                print("Putting model in evaluation mode...")
                model = model.eval()
                print("Model ready!")
                
                return model, processor
            except Exception as e:
                print(f"Error loading model from disk: {e}")
                print(f"Error type: {type(e)}")
                print("Forcing download of a new model...")
                # Force download new model
                return download_model(device)
        else:
            print("Model files not found or incomplete on disk, downloading...")
            return download_model(device)
    except Exception as e:
        print(f"Exception in load_model: {e}")
        print(f"Exception type: {type(e)}")
        raise


def download_model(device):
    print("Downloading model (first run only)...")
    try:
        print("Download step 1 - initializing...")
        model = ColQwen2.from_pretrained(
            "vidore/colqwen2-v1.0",
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True
        )
        print("Download step 2 - model downloaded successfully!")
        
        print("Setting model to eval mode...")
        model = model.eval()
        
        print("Downloading processor...")
        processor = ColQwen2Processor.from_pretrained(
            "vidore/colqwen2-v1.0",
            trust_remote_code=True
        )
        print("Processor downloaded successfully!")
        
        # Save model and processor to disk with absolute paths
        print("Saving model to disk for future use...")
        try:
            abs_model_path = os.path.abspath(MODEL_PATH)
            abs_processor_path = os.path.abspath(PROCESSOR_PATH)
            
            print(f"Saving model to {abs_model_path}...")
            model.save_pretrained(abs_model_path)
            print(f"Model saved successfully!")
            
            print(f"Saving processor to {abs_processor_path}...")
            processor.save_pretrained(abs_processor_path)
            print(f"Processor saved successfully!")
        except Exception as e:
            print(f"Error saving model to disk: {e}")
            print(f"Error type: {type(e)}")
    except Exception as e:
        print(f"Error downloading model: {e}")
        print(f"Error type: {type(e)}")
        raise
    
    return model, processor

# Load model and processor
model, processor = load_model()


def encode_image_to_base64(image):
    """Encodes a PIL image to a base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
    

def query_gpt4o_mini(query, images, api_key):
    """Calls OpenAI's GPT-4o-mini with the query and image data."""

    if api_key and api_key.startswith("sk"):
        try:
            from openai import OpenAI
        
            base64_images = [encode_image_to_base64(image[0]) for image in images]
            client = OpenAI(api_key=api_key.strip())
            PROMPT = """
            You are a smart assistant designed to answer questions about a PDF document.
            You are given relevant information in the form of PDF pages. Use them to construct a short response to the question, and cite your sources (page numbers, etc).
            If it is not possible to answer using the provided pages, do not attempt to provide an answer and simply say the answer is not present within the documents.
            Give detailed and extensive answers, only containing info in the pages you are given.
            You can answer using information contained in plots and figures if necessary.
            Answer in the same language as the query.
            
            Query: {query}
            PDF pages:
            """
        
            response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                  "role": "user",
                  "content": [
                    {
                      "type": "text",
                      "text": PROMPT.format(query=query)
                    }] + [{
                      "type": "image_url",
                      "image_url": {
                        "url": f"data:image/jpeg;base64,{im}"
                        },
                    } for im in base64_images]
                }
              ],
              max_tokens=500,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"OpenAI API connection failure. Error: {e}"
        
    return "Enter your OpenAI API key to get a custom response"


@spaces.GPU
def search(query: str, ds, images, k, api_key):
    try:
        k = min(k, len(ds))
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if device != model.device:
            model.to(device)
            
        qs = []
        with torch.no_grad():
            batch_query = processor.process_queries([query]).to(model.device)
            embeddings_query = model(**batch_query)
            qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))

        scores = processor.score(qs, ds, device=device)

        top_k_indices = scores[0].topk(k).indices.tolist()

        results = []
        for idx in top_k_indices:
            results.append((images[idx], f"Page {idx}"))

        # Generate response from GPT-4o-mini
        ai_response = query_gpt4o_mini(query, results, api_key)

        return results, ai_response
    except Exception as e:
        return [], f"Error in search function: {str(e)}"



def convert_files(files):
    """Convert uploaded files to images, handling different file types from Gradio."""
    images = []
    
    for f in files:
        try:
            # Handle the file based on its type
            if hasattr(f, 'name'):
                # This is likely a file object with a name attribute
                file_path = f.name
                print(f"Processing file with path: {file_path}")
                images.extend(convert_from_path(file_path, thread_count=4))
            elif isinstance(f, tuple) and len(f) == 2:
                # If it's a tuple of (name, file-like object) as returned by some Gradio versions
                temp_name, temp_file = f
                print(f"Processing tuple with name: {temp_name}")
                # If it's a file-like object, read it and convert from bytes
                if hasattr(temp_file, 'read'):
                    file_content = temp_file.read()
                    images.extend(convert_from_bytes(file_content, thread_count=4))
                else:
                    # If it's a path
                    images.extend(convert_from_path(temp_file, thread_count=4))
            elif isinstance(f, str):
                # If it's directly a file path
                print(f"Processing file path: {f}")
                images.extend(convert_from_path(f, thread_count=4))
            else:
                # Try to get the file path from the object
                print(f"Unknown file type: {type(f)}, trying to handle generically")
                if hasattr(f, 'file'):
                    # Some Gradio versions provide a file attribute
                    file_content = f.file.read()
                    images.extend(convert_from_bytes(file_content, thread_count=4))
                elif hasattr(f, 'read'):
                    # If it's a file-like object
                    file_content = f.read()
                    images.extend(convert_from_bytes(file_content, thread_count=4))
                else:
                    raise TypeError(f"Unsupported file type: {type(f)}. Please provide a valid PDF file.")
        except Exception as e:
            print(f"Error processing file {f}: {e}")
            import traceback
            traceback.print_exc()
            # Continue with other files rather than failing completely
            continue

    if len(images) > PAGE_LIMIT:
        raise gr.Error("The number of images in the dataset should be less than 150.")
    
    if not images:
        raise ValueError("No valid PDF files were processed. Please check your uploads.")
        
    return images


# Replace the existing index function with this version
def index(files, ds):
    try:
        print("Converting files")
        print(f"File types: {[type(f) for f in files]}")
        
        # Reset the embeddings list and images list
        ds = []
        all_images = []
        
        for f in files:
            # Get the file path
            if hasattr(f, 'name'):
                file_path = f.name
            elif isinstance(f, tuple) and len(f) == 2:
                file_path = f[0]  # Use the name from the tuple
            elif isinstance(f, str):
                file_path = f
            else:
                # Try other approaches to get the path
                if hasattr(f, 'file'):
                    file_path = str(f.file)
                else:
                    # Generate a temporary unique identifier if we can't get the path
                    import hashlib
                    file_path = f"unknown_file_{hashlib.md5(str(f).encode()).hexdigest()}"
            
            filename = os.path.basename(file_path)
            print(f"Processing file: {filename} (path: {file_path})")
            
            # Check if embeddings exist for this file
            if db.embeddings_exist(file_path):
                print(f"Loading existing embeddings for {filename}")
                
                # Convert file to images for display only
                try:
                    images = convert_files([f])
                    if not images:
                        print(f"Could not convert {filename} to images")
                        continue
                except Exception as e:
                    print(f"Error converting file to images: {e}")
                    continue
                
                # Load embeddings from database
                file_embeddings = db.load_embeddings(file_path)
                
                if file_embeddings and len(file_embeddings) > 0:
                    # Add to our lists
                    ds.extend(file_embeddings)
                    all_images.extend(images)
                    
                    print(f"Loaded {len(file_embeddings)} existing embeddings")
                else:
                    print(f"Failed to load embeddings, will regenerate")
                    # Fall back to generating new embeddings
                    process_new_file(f, file_path, ds, all_images)
            else:
                print(f"No existing embeddings for {filename}, generating new ones")
                # Process the file normally
                process_new_file(f, file_path, ds, all_images)
                
        return f"Processed {len(files)} files with {len(ds)} total embeddings", ds, all_images
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        return f"Error in indexing: {str(e)}\n{traceback_str}", ds, []


# Add this new function
def process_new_file(f, file_path, ds, all_images):
    """Process a file by generating new embeddings and saving them"""
    try:
        # Convert file to images
        images = convert_files([f])
        
        if not images:
            print(f"Could not convert {os.path.basename(file_path)} to images")
            return
            
        # Get embeddings for this file only
        file_ds = []
        status, file_ds, _ = index_gpu(images, file_ds)
        
        # Save the new embeddings if successful
        if file_ds and len(file_ds) > 0:
            saved = db.save_embeddings(file_path, file_ds, len(images))
            if saved:
                print(f"Saved {len(file_ds)} new embeddings for {os.path.basename(file_path)}")
            else:
                print(f"Failed to save embeddings for {os.path.basename(file_path)}")
            
            # Add to our complete lists
            ds.extend(file_ds)
            all_images.extend(images)
        else:
            print(f"Failed to generate embeddings for {os.path.basename(file_path)}")
    except Exception as e:
        print(f"Error processing new file {os.path.basename(file_path)}: {e}")

# Modified index_gpu function (keeping the core functionality the same)
@spaces.GPU
def index_gpu(images, ds):
    """Example script to run inference with ColPali (ColQwen2)"""
    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if device != model.device:
            model.to(device)
            
        # run inference - docs
        dataloader = DataLoader(
            images,
            batch_size=1,
            # num_workers=2,
            shuffle=False,
            collate_fn=lambda x: processor.process_images(x).to(model.device),
        )

        for batch_doc in tqdm(dataloader):
            with torch.no_grad():
                batch_doc = {k: v.to(device) for k, v in batch_doc.items()}
                embeddings_doc = model(**batch_doc)
            ds.extend(list(torch.unbind(embeddings_doc.to("cpu"))))
        return f"Uploaded and converted {len(images)} pages", ds, images
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        return f"Error in processing: {str(e)}\n{traceback_str}", ds, []

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ColPali: Efficient Document Retrieval with Vision Language Models (ColQwen2) 📚")
    gr.Markdown("""Demo to test ColQwen2 (ColPali) on PDF documents. 
    ColPali is model implemented from the [ColPali paper](https://arxiv.org/abs/2407.01449).

    This demo allows you to upload PDF files and search for the most relevant pages based on your query.
    Refresh the page if you change documents !

    ⚠️ This demo uses a model trained exclusively on A4 PDFs in portrait mode, containing english text. Performance is expected to drop for other page formats and languages.
    Other models will be released with better robustness towards different languages and document formats !
    """)
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("## 1️⃣ Upload PDFs")
            file = gr.File(file_types=["pdf"], file_count="multiple", label="Upload PDFs")

            convert_button = gr.Button("🔄 Index documents")
            message = gr.Textbox("Files not yet uploaded", label="Status")
            api_key = gr.Textbox(placeholder="Enter your OpenAI KEY here (optional)", label="API key", value=os.getenv('OPENAI_API_KEY', ''), type="password")
            embeds = gr.State(value=[])
            imgs = gr.State(value=[])

        with gr.Column(scale=3):
            gr.Markdown("## 2️⃣ Search")
            query = gr.Textbox(placeholder="Enter your query here", label="Query")
            k = gr.Slider(minimum=1, maximum=10, step=1, label="Number of results", value=5)


    # Define the actions
    search_button = gr.Button("🔍 Search", variant="primary")
    output_gallery = gr.Gallery(label="Retrieved Documents", height=600, show_label=True)
    output_text = gr.Textbox(label="AI Response", placeholder="Generated response based on retrieved documents")

    convert_button.click(index, inputs=[file, embeds], outputs=[message, embeds, imgs])
    search_button.click(search, inputs=[query, embeds, imgs, k, api_key], outputs=[output_gallery, output_text])

if __name__ == "__main__":
    # Use a simpler launch method to avoid compatibility issues
    import atexit
    import shutil
    import tempfile
    
    # Create a temporary directory for file uploads if it doesn't exist
    temp_dir = os.path.join(tempfile.gettempdir(), "colpali_uploads")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Clean up function to remove temp files
    def cleanup():
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Register the cleanup function
    atexit.register(cleanup)
    
    # Launch Gradio with simplified server settings
    demo.queue(max_size=10).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )