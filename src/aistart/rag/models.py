from pydantic import BaseModel


class Chunk(BaseModel):
    id: str
    source: str
    index: int
    text: str
