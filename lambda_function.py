from mangum import Mangum
from main import app  # import FastAPI app

handler = Mangum(app)
