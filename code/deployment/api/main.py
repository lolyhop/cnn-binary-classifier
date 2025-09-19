from code.deployment.api.router import router

import uvicorn
from fastapi import FastAPI

app: FastAPI = FastAPI(title="Cats vs Dogs Classifier API")

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
