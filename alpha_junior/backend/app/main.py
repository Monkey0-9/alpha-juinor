from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Alpha Junior Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Alpha Junior API is live"}

@app.get("/api/alpha")
async def get_alpha():
    # Placeholder for alpha generation logic
    return {"status": "success", "alpha_score": 0.85, "strategy": "momentum"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
