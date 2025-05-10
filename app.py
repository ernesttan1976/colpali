import os
import spaces
import base64
from io import BytesIO

import gradio as gr
import torch

from pdf2image import convert_from_path
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from colpali_engine.models import ColQwen2, ColQwen2Processor

import sys
print(f"Python path: {sys.executable}")
print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch path: {torch.__file__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# Define model paths
MODEL_DIR = "./models/colqwen2"
MODEL_PATH = os.path.join(MODEL_DIR, "model")
PROCESSOR_PATH = os.path.join(MODEL_DIR, "processor")


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
            return "OpenAI API connection failure. Verify the provided key is correct (sk-***)."
        
    return "Enter your OpenAI API key to get a custom response"


@spaces.GPU
def search(query: str, ds, images, k, api_key):
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


def index(files, ds):
    print("Converting files")
    images = convert_files(files)
    print(f"Files converted with {len(images)} images.")
    return index_gpu(images, ds)
    


def convert_files(files):
    images = []
    for f in files:
        images.extend(convert_from_path(f, thread_count=4))

    if len(images) >= 150:
        raise gr.Error("The number of images in the dataset should be less than 150.")
    return images


@spaces.GPU
def index_gpu(images, ds):
    """Example script to run inference with ColPali (ColQwen2)"""

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device != model.device:
        model.to(device)
        
    # run inference - docs
    dataloader = DataLoader(
        images,
        batch_size=4,
        # num_workers=4,
        shuffle=False,
        collate_fn=lambda x: processor.process_images(x).to(model.device),
    )

    for batch_doc in tqdm(dataloader):
        with torch.no_grad():
            batch_doc = {k: v.to(device) for k, v in batch_doc.items()}
            embeddings_doc = model(**batch_doc)
        ds.extend(list(torch.unbind(embeddings_doc.to("cpu"))))
    return f"Uploaded and converted {len(images)} pages", ds, images



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
    demo.queue(max_size=10).launch(debug=True)