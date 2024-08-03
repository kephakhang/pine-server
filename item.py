from pydantic import BaseModel

class Item(BaseModel):
    question: str = ''
    uid: str = None
