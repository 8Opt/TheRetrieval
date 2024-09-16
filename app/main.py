import uvicorn
from setup_app import setup_app, lifespan

app = setup_app(lifespan)

@app.get("/")
async def root():
    return {"message": "Welcome to OmniLens"}


if __name__ == "__main__": 
    uvicorn.run("main:app", 
                host="0.0.0.0", 
                port=8000, 
                log_level="info", 
                reload=True
                )