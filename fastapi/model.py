from pydantic import BaseModel
from typing import Union

# Pydantic model for input item
class Item(BaseModel):
    itemCode: str
    itemName: str 
    serviceHours: Union[int, None] = None
    description: Union[str, None] = None
    price: Union[int, None] = None


# Data models for ad generation
class AdResponse(BaseModel):
    id: str
    created_at: str
    image_data: str  # Base64 encoded image