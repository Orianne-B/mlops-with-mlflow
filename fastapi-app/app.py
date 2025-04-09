from fastapi import FastAPI
import uvicorn

from routes import router

# Initialize FastAPI app
app = FastAPI(title="ML Flow Service")

# Include the router
app.include_router(router)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
