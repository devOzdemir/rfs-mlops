from fastapi import FastAPI
app = FastAPI()
@app.get("/")
def read_root():
    return {"Hello": "RFS Pricing API"}
