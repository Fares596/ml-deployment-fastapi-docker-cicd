from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
async def home():
    return {"data": "Hello World"}

@app.get("/items/")
def get_items():
    return {"items": ["apple", "banana", "cherry"]}


class Item(BaseModel):
    name: str
    price: float

@app.post("/items/")
def create_item(item: Item):
    return {"message": "Item created successfully", "item": item}


@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_id": item_id, "updated_item": item}


@app.delete("/items/{item_id}")
def delete_item(item_id: int):
    return {"message": f"Item with ID {item_id} deleted successfully"}

