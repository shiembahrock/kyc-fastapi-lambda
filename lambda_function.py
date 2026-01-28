from mangum import Mangum
from main import app  # import FastAPI app

#handler = Mangum(app)
handler = Mangum(app, api_gateway_base_path="/dev")
