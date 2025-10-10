from pydantic import BaseModel
from fastapi import FastAPI

fastapi = FastAPI()

class Item(BaseModel):
    name: str
    price: float

@fastapi.get("/{name}")
async def get_name(name: str):
    return { "name": name }

@fastapi.get("/")
async def get_data(name: str, age: int = 0, height: int = 10):
    return { "name": name, "age": age, "height": height }


@fastapi.post("/items")
async def create_item (item: Item):
    return {"item": item}