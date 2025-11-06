from pydantic import BaseModel, Field
from typing import Dict, Any


class MappingOutput(BaseModel):
    query_key: str 
    params: Dict[str,Any]