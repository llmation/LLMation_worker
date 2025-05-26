import uvicorn
from app import create_app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "app:create_app",
        factory=True,
        host="0.0.0.0",
        port=12000,
        reload=True,
        access_log=True
    )
