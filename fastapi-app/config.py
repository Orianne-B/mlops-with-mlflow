from dotenv import find_dotenv, load_dotenv
import os
from pydantic import BaseModel

load_dotenv(find_dotenv())


class Configuration(BaseModel):
    model_uri: str = os.getenv("MODEL_URI")
