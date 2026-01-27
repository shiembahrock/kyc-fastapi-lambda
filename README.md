# KYC Backend

This is a Python-based backend service for KYC (Know Your Customer) operations, built with [FastAPI](https://fastapi.tiangolo.com/). It is containerized with Docker, making it ready for deployment on AWS (App Runner, ECS, or EKS).

## Project Structure

- `main.py`: The entry point of the FastAPI application.
- `requirements.txt`: Python dependencies.
- `Dockerfile`: Configuration for building the Docker image.
- `kyc-frontend/`: React-based frontend application.

## Local Development

### Prerequisites

- Python 3.11+
- Node.js & npm
- Docker (optional, for container testing)

### Backend Setup

1.  **Create a virtual environment:**

    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**

    ```bash
    uvicorn main:app --reload
    ```

    The API will be available at `http://127.0.0.1:8000`.
    Documentation is available at `http://127.0.0.1:8000/docs`.

### Frontend Setup

1.  **Navigate to the frontend directory:**
    ```bash
    cd kyc-frontend
    ```

2.  **Install dependencies:**
    ```bash
    npm install
    ```

3.  **Run the development server:**
    ```bash
    npm run dev
    ```
    The UI will be available at `http://localhost:5173`.

## Docker Usage

1.  **Build the image:**

    ```bash
    docker build -t kyc-backend .
    ```

2.  **Run the container:**

    ```bash
    docker run -p 8000:8000 kyc-backend
    ```

## AWS Deployment

### Option 1: AWS App Runner (Recommended for Simplicity)

1.  Push your code to a GitHub repository.
2.  Go to the AWS App Runner console.
3.  Create a service linked to your GitHub repository.
4.  Configure the build settings:
    -   **Runtime:** Python 3
    -   **Build command:** `pip install -r requirements.txt`
    -   **Start command:** `uvicorn main:app --host 0.0.0.0 --port 8080`
    -   **Port:** 8080
    *(Note: You may need to adjust the port in `main.py` or the start command to match)*

    **OR** (using Docker):

1.  Push your Docker image to Amazon ECR (Elastic Container Registry).
2.  Create an App Runner service pointing to your ECR image.

### Option 2: AWS ECS (Elastic Container Service)

1.  Push your Docker image to Amazon ECR.
2.  Create an ECS Task Definition using the image.
3.  Create an ECS Service to run the task (Fargate is recommended for serverless management).
