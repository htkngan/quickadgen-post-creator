# Image_processing.py - Utility functions and classes for image processing in ad generation

# Standard and third-party libraries
import io
import os
import re
import PIL
from PIL import Image
from google import genai
from google.genai import types
import cv2
import numpy as np
import torch
from dotenv import load_dotenv

# OCR and object removal libraries
from paddleocr import PaddleOCR
from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config, HDStrategy

# Load environment variables from .env file
load_dotenv()

class ImageContext:
    """
    Helper class to manage image data and conversions (from file, buffer, bytes, base64)
    """
    def __init__(self, image: Image.Image = None, path: str = None):
        self.image = image 
        self.buffer = None
        self.path = path
    
    def load_from_path(self, path: str):
        """Load image from file path into ImageContext"""
        self.path = path
        self.image = Image.open(path).convert("RGBA")
        self.buffer = None
        return self

    def save_to_path(self, path: str):
        """Save current image to file"""
        if self.image is not None:
            self.image.save(path, format="PNG")
            self.path = path
        return self

    def to_buffer(self):
        """Convert image to BytesIO buffer"""
        if self.image is not None:
            buffer = io.BytesIO()
            self.image.save(buffer, format="PNG")
            buffer.seek(0)
            self.buffer = buffer
        return self

    def from_buffer(self, buffer: io.BytesIO):
        """Load image from BytesIO buffer"""
        self.buffer = buffer
        self.image = Image.open(buffer)
        return self

    def to_bytes(self, format="PNG"):
        """Convert image to bytes in specified format"""
        if self.image is not None:
            img_byte_arr = io.BytesIO()
            self.image.save(img_byte_arr, format=format)
            return img_byte_arr.getvalue()
        return None
        
    def from_bytes(self, image_bytes: bytes):
        """Load image from bytes data"""
        if image_bytes:
            self.buffer = io.BytesIO(image_bytes)
            self.image = Image.open(self.buffer)
        return self
    
    def get_base64(self, format="PNG"):
        """Convert image to base64 string"""
        import base64
        image_bytes = self.to_bytes(format)
        if image_bytes:
            return base64.b64encode(image_bytes).decode('utf-8')
        return None
        
    def from_base64(self, base64_string: str):
        """Load image from base64 string"""
        import base64
        if base64_string:
            image_bytes = base64.b64decode(base64_string)
            return self.from_bytes(image_bytes)
        return self

class PromptManager:
    """
    Class to manage and generate prompts for ad generation.
    """

    @staticmethod
    def generate_product_prompt(item: dict) -> str:
        prompt = (
            f"Tạo một poster quảng cáo vô cùng cao cấp và sáng tạo để đăng tải lên mạng xã hội facebook hình vuông cho {item.get('itemName', '')} "
            "tuân thủ quy tắc không hiển thị hình ảnh sản phẩm trong poster quảng cáo vì lí do bảo mật. "
        )
        if item.get('description'):
            prompt += f"Ảnh quảng cáo nên có chủ đề biểu diễn cho {item['description']}."
        if item.get('pattern'):
            prompt += f" Ảnh quảng cáo cần có các chi tiết như {item['pattern']}. Nếu {item['pattern']} rỗng hãy cho nó phong cách thật high end, elegant, luxury, minimalist"
        return prompt

    @staticmethod
    def generate_service_prompt(itemName: str, ads_text: str) -> str:
        return (
            f'Create a background to advertise a product in facebook: {itemName} with the content is {ads_text}. '
            'The poster should have a creative pattern, vivid images, and be high quality, 16:9 ratio, '
            'style photographic, modern, impressive, creative.'
        )
    
    @staticmethod    
    def generate_service_prompt_with_template(itemName: str, ads_text: str, image: PIL) -> str:
        return (
            f'Create a background to advertise a product in facebook: {itemName} with the content is {ads_text} with the style like {image} but more creative. '
                'The poster should have a creative pattern, vivid images, and be high quality, 16:9 ratio, '
                'style photographic, modern, impressive, creative.'
        )
        
class AdGenerator:
    """
    Main class for ad image generation, text detection/removal, and product overlay
    """
    def __init__(self):
         # Initialize PaddleOCR for Vietnamese language
        paddle_ocr_instance = PaddleOCR(use_angle_cls=True, lang='vi', show_log=False)

        # Initialize PaddleOCR for Vietnamese language
        lama_model_instance = ModelManager(
            name="lama",
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        self.ocr = paddle_ocr_instance
        self.lama_model = lama_model_instance 
        # Initialize Google GenAI client
        self.genai_client = genai.Client()
    
    @staticmethod
    def clean_text(text):
        """Remove emoji, special characters, and punctuation from text"""
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

        text = emoji_pattern.sub('', text)
        text = punctuation_pattern.sub('', text)

        return text.strip()

    def generate_background_service(self, itemName: str, ads_text: str, image_bytes: bytes | None, model: str = "gemini-2.0-flash-exp-image-generation") -> ImageContext:
        """
        Generate a background image for advertising using GenAI
        """
        if image_bytes:
            image = Image.open(io.BytesIO(image_bytes))
            prompt = PromptManager.generate_service_prompt_with_template(itemName, self.clean_text(ads_text), image)
            response_image = self.genai_client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                    response_modalities=['TEXT', 'IMAGE']
                )
            )
              
            for part in response_image.candidates[0].content.parts:
                if part.inline_data is not None:
                    image = Image.open(io.BytesIO(part.inline_data.data))
                    return ImageContext(image=image)
            raise ValueError("No image returned from GenAI")
        else:
            prompt = PromptManager.generate_service_prompt(itemName, self.clean_text(ads_text))
            
            response_image = self.genai_client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE']
                )
            )
               
            for part in response_image.candidates[0].content.parts:
                if part.inline_data is not None:
                    image = Image.open(io.BytesIO(part.inline_data.data))
                    return ImageContext(image=image)
            raise ValueError("No image returned from GenAI")

    def generate_background_product(self, item: dict, model: str = "gemini-2.0-flash-exp-image-generation") -> ImageContext:
        """
        Generate a background image for a product using GenAI
        """
        prompt = PromptManager.generate_product_prompt(item)
        response_image = self.genai_client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE']
            )
        )
   
        for part in response_image.candidates[0].content.parts:
            if part.inline_data is not None:
                image = Image.open(io.BytesIO(part.inline_data.data))
                return ImageContext(image=image)
        raise ValueError("No image returned from GenAI")
    
    def detect_text_in_image(self, image_input) -> tuple[np.ndarray, bool]:
        """
        Detect text in an image (accepts PIL.Image or ImageContext) and return a mask and flag.
        """
        if hasattr(image_input, "image"):
            image = image_input.image
        else:
            image = image_input

        img_np = np.array(image)
        if len(img_np.shape) == 3 and img_np.shape[2] == 4:  # RGBA
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        
        result = self.ocr.ocr(img_np, cls=True)
        if result is None:
            return np.zeros(img_np.shape[:2], dtype=np.uint8), False

        mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
        text_detected = False
        margin = 10

        for line in result:
            if line is None:
                continue
            for word_info in line:
                text_detected = True
                points = np.array(word_info[0], dtype=np.int32)
                
                x_min = max(0, np.min(points[:, 0]) - margin)
                y_min = max(0, np.min(points[:, 1]) - margin)
                x_max = min(img_np.shape[1], np.max(points[:, 0]) + margin)
                y_max = min(img_np.shape[0], np.max(points[:, 1]) + margin)
                
                expanded_rect = np.array([
                    [x_min, y_min],
                    [x_max, y_min],
                    [x_max, y_max],
                    [x_min, y_max]
                ], dtype=np.int32).reshape((-1, 1, 2))
                
                cv2.fillPoly(mask, [expanded_rect], color=255)
                original_points = points.reshape((-1, 1, 2))
                cv2.fillPoly(mask, [original_points], color=255)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        return mask, text_detected

    def remove_text_from_image(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        """
        Remove text from a PIL Image using a mask
        """
        image_np = np.array(image)
        if len(image_np.shape) == 3 and image_np.shape[2] == 4:  # RGBA
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

        if mask.shape[:2] != image_np.shape[:2]:
            mask = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]))

        config = Config(
            ldm_steps=20,
            hd_strategy=HDStrategy.ORIGINAL,
            hd_strategy_crop_margin=32,
            hd_strategy_crop_trigger_size=2048,
            hd_strategy_resize_limit=2048,
        )

        result = self.lama_model(image_np, mask, config)

        if result.dtype != np.uint8:
            if np.max(result) <= 1.0:
                result = (result * 255).astype(np.uint8)
            else:
                result = result.astype(np.uint8)

        return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    
    def generate_clean_background(self, image_context: ImageContext):
        """
        Generate a clean background by removing detected text
        """
        if not isinstance(image_context, ImageContext):
            raise TypeError("Input must be an ImageContext")
        mask, has_text = self.detect_text_in_image(image_context)
        if not has_text:
            return image_context
        cleaned_image = self.remove_text_from_image(image_context.image, mask)
        return ImageContext(image=cleaned_image)

    def add_products_to_image(self, background: Image.Image, products: list[Image.Image], positions: list[str]) -> Image.Image:
        """
        Add multiple products to a background image
        """
        bg_image = background.copy()
        
        for i, product_image in enumerate(products):
            position = positions[i] if i < len(positions) else 'center'
            
            original_ratio = product_image.width / product_image.height
            target_height = int(bg_image.height * 0.5)
            target_width = int(target_height * original_ratio)
            
            if target_width > bg_image.width:
                target_width = int(bg_image.width * 0.8)
                target_height = int(target_width / original_ratio)
            
            product_image = product_image.resize(
                (target_width, target_height),
                Image.Resampling.LANCZOS
            )
            
            if position == 'left':
                pos_x = int(bg_image.width * 0.1)
                pos_y = (bg_image.height - product_image.height) // 2
            elif position == 'right':
                pos_x = int(bg_image.width * 0.9) - product_image.width
                pos_y = (bg_image.height - product_image.height) // 2
            elif position == 'center':
                pos_x = (bg_image.width - product_image.width) // 2
                pos_y = (bg_image.height - product_image.height) // 2
            elif position == 'top':
                pos_x = (bg_image.width - product_image.width) // 2
                pos_y = int(bg_image.height * 0.1)
            elif position == 'bottom':
                pos_x = (bg_image.width - product_image.width) // 2
                pos_y = int(bg_image.height * 0.9) - product_image.height
            elif position == 'top-left':
                pos_x = int(bg_image.width * 0.1)
                pos_y = int(bg_image.height * 0.1)
            elif position == 'top-right':
                pos_x = int(bg_image.width * 0.9) - product_image.width
                pos_y = int(bg_image.height * 0.1)
            elif position == 'bottom-left':
                pos_x = int(bg_image.width * 0.1)
                pos_y = int(bg_image.height * 0.9) - product_image.height
            elif position == 'bottom-right':
                pos_x = int(bg_image.width * 0.9) - product_image.width
                pos_y = int(bg_image.height * 0.9) - product_image.height
            else:
                raise ValueError("Invalid position")
            
            bg_image.paste(product_image, (pos_x, pos_y), product_image)
        
        return bg_image