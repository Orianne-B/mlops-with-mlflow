from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI
import uvicorn

from core import config
from routes import router

load_dotenv(find_dotenv())


# Initialize FastAPI app
app = FastAPI(title=config.API_TITLE)

# Include the router
app.include_router(router)


if __name__ == "__main__":
    uvicorn.run("app:app", host=config.API_HOST, port=config.API_PORT, reload=True)
