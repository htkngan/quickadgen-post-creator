from typing import Union
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import logging
from openai import OpenAI
import asyncio
import time
import tiktoken  # Add this import for token counting
import os

from Image_processing import ImageContext, AdGenerator
import re
from google.genai import types

# Load environment variables
load_dotenv()

client_gemini = genai.Client()
client_gpt = OpenAI()

#gpt_model = os.getenv("gpt_model")
gemni_model = "gemini-2.0-flash"

app = FastAPI()

# Update CORS middleware to be more permissive during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins during development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic model for input
class Item(BaseModel):
    itemCode: str
    itemName: str 
    serviceHours: Union[int, None] = None
    description: Union[str, None] = None
    price: Union[int, None] = None

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Ad Content Generator API"}

# Add health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}

#Set up two LLM clients
def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string."""
    try:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return len(encoding.encode(text))
    except Exception as e:
        logger.error(f"Error counting tokens: {str(e)}")
        return 0

async def run_gemini(item: Item):
    start_time = time.time()
    try:
        prompt = f"""
        Bạn là một chuyên gia viết quảng cáo, nhiệm vụ của bạn là viết một quảng cáo hoàn chỉnh bằng tiếng Việt cho sản phẩm {item.itemName} để đăng tải ngay lên nền tảng Facebook. Không cần giải thích nội dung.
        Với mô tả sản phẩm như sau: {item.description if item.description else "Không có mô tả, sử dụng kiến thức chung về sản phẩm."}. Cùng một số thông tin kèm theo như số giờ dịch vụ {item.serviceHours} và giá {item.price}.
        Giọng điệu phải thuyết phục và tập trung vào việc thu hút sự chú ý nhanh chóng và truyền tải giá trị chỉ trong vài giây, sử dụng ngôn ngữ nói trực tiếp đến mong muốn và điểm khó khăn của đối tượng. 
        Mục tiêu là tạo ra một kết nối có ý nghĩa và thúc đẩy hành động ngay lập tức, cho dù đó là mua hàng, đăng ký hay tìm hiểu thêm về sản phẩm.
        """
        
        logger.info(f"Attempting to generate content for Gemini with product: {item.itemName}")
        response = await asyncio.to_thread(
            client_gemini.models.generate_content,
            model=gemni_model,
            contents=[prompt]
        )
        
        if not response or not hasattr(response, 'text'):
            logger.error(f"Invalid response from Gemini: {response}")
            return {"model": gemni_model, "status": "error", "ad_content": None, "time": 0, "error": "Invalid response format"}
        
        formatted_text = response.text.strip()
        token_count = count_tokens(formatted_text)
        end_time = time.time()
        logger.info(f"Successfully generated Gemini content of length: {len(formatted_text)}")
        logger.info(f"Gemini response text: {formatted_text}")
        
        return {
            "model": gemni_model,
            "status": "success",
            "ad_content": formatted_text,
            "total_tokens": token_count,
            "time": end_time - start_time
        }
    
    except Exception as e:
        end_time = time.time()
        logger.error(f"Gemini error: {str(e)}")
        return {
            "model": gemni_model,
            "status": "error",
            "ad_content": None,
            "time": end_time - start_time,
            "error": str(e)
        }

async def run_gpt(item: Item, gpt_model: str):
    start_time = time.time()
    try:
        prompt = f"""
        Bạn là một chuyên gia viết quảng cáo, nhiệm vụ của bạn là viết một quảng cáo hoàn chỉnh bằng tiếng Việt cho sản phẩm {item.itemName} để đăng tải ngay lên nền tảng Facebook. 
        Với mô tả sản phẩm như sau: {item.description if item.description else "Không có mô tả, sử dụng kiến thức chung về sản phẩm."}. Cùng một số thông tin kèm theo như số giờ dịch vụ {item.serviceHours} và giá {item.price}.
        Giọng điệu phải thuyết phục và tập trung vào việc thu hút sự chú ý nhanh chóng và truyền tải giá trị chỉ trong vài giây, sử dụng ngôn ngữ nói trực tiếp đến mong muốn và điểm khó khăn của đối tượng. 
        Mục tiêu là tạo ra một kết nối có ý nghĩa và thúc đẩy hành động ngay lập tức, cho dù đó là mua hàng, đăng ký hay tìm hiểu thêm về sản phẩm.
        """
        
        logger.info(f"Attempting to generate content for GPT with product: {item.itemName}")
        response = await asyncio.to_thread(
            client_gpt.chat.completions.create,
            model=gpt_model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        if not response or not response.choices or not response.choices[0].message.content:
            logger.error(f"Invalid response from GPT: {response}")
            return {"model": gpt_model, "status": "error", "ad_content": None, "time": 0, "error": "Invalid response format"}
        
        formatted_text = response.choices[0].message.content.strip()
        token_count = count_tokens(formatted_text)
        end_time = time.time()
        logger.info(f"Successfully generated GPT content of length: {len(formatted_text)}")
        logger.info(f"GPT response text: {formatted_text}")
        
        return {
            "model": gpt_model,
            "status": "success",
            "ad_content": formatted_text,
            "total_tokens": token_count,
            "time": end_time - start_time
        }
    
    except Exception as e:
        end_time = time.time()
        logger.error(f"GPT error: {str(e)}")
        return {
            "model": gpt_model,
            "status": "error",
            "ad_content": None,
            "time": end_time - start_time,
            "error": str(e)
        }
        
def clean_text(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & pictographs
        "\U0001F680-\U0001F6FF"  # Transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # Flags
        "\U00002500-\U00002BEF"  # Chinese characters
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"
        "\u3030"
        "]+", flags=re.UNICODE)

    punctuation_pattern = re.compile(r"[.,!?;:“”\"\'‘’…—–\-–()\[\]{}<>•*~@#$%^&+=/\\|]")

    # Xoá emoji
    text = emoji_pattern.sub('', text)
    # Xoá dấu câu
    text = punctuation_pattern.sub('', text)

    return text.strip()
        
async def run_gemini_image(item: Item, prompt: str, image_context: ImageContext):
    start_time = time.time()
    try:
        text_input = f"""
                generate a poster for a advertise {clean_text(prompt.text)} with the style like {image_context} but more creative,
                the poster should have the creative pattern, imagvivid images and should be a high quality, ratio 16:9,
                style photographic, modern, impressive, creative
        """
        logger.info(f"Attempting to generate image for Gemini with product: {item.itemName}")
        response = await asyncio.to_thread(
            client_gemini.models.generate_image,
            model="gemini-2.0-flash-exp-image-generation",
            contents=[text_input],
            config=types.GenerateContentConfig(
            response_modalities=['TEXT','IMAGE'])
        )
        
        if not response or not hasattr(response, 'image'):
            logger.error(f"Invalid response from Gemini: {response}")
            return {"model": gemni_model, "status": "error", "ad_content": None, "time": 0, "error": "Invalid response format"}
        
        formatted_text = response.image.strip()
        token_count = count_tokens(formatted_text)
        end_time = time.time()
        logger.info(f"Successfully generated Gemini image of length: {len(formatted_text)}")
        
        return {
            "model": gemni_model,
            "status": "success",
            "ad_content": formatted_text,
            "total_tokens": token_count,
            "time": end_time - start_time
        }
    
    except Exception as e:
        end_time = time.time()
        logger.error(f"Gemini error: {str(e)}")
        return {
            "model": gemni_model,
            "status": "error",
            "ad_content": None,
            "time": end_time - start_time,
            "error": str(e)
        }
    

@app.post("/generate-ad")
async def generate_ad(item: Item):
    logger.info(f"Received request for product: {item.itemName}")
    
    # Chạy hai model song song
    results = await asyncio.gather(
        run_gemini(item),
        run_gpt(item, "gpt-4.1-mini"),
        run_gpt(item, "gpt-4.1-nano")
    )
    
    # Ghi log kết quả tổng thể
    for result in results:
        if result["status"] == "success":
            logger.info(f"{result['model']} completed in {result['time']:.3f} seconds")
        else:
            logger.error(f"{result['model']} failed: {result.get('error', 'Unknown error')}")
    
    # Trả về kết quả từ cả hai model
    return JSONResponse(
        content={
            "status": "success",
            "results": results
        },
        status_code=200
    )

# Example endpoint to retrieve item (for testing)
@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

# Example endpoint to update item (for testing)
@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.itemName, "item_id": item_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)