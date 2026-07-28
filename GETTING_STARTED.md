# Getting Started

Follow these steps to launch the AeroShield Predictive Maintenance Gateway & Dashboard locally on your machine.

## 1. Install Dependencies

First, make sure you are in the project root directory and install all required Python packages using pip:

```bash
pip install -r requirements.txt
```

## 2. Start the Backend (FastAPI Gateway)

The backend must be run from within the `backend` directory so that Python can properly resolve the module paths. Open a terminal, navigate to the `backend` folder, and start the `uvicorn` server:

```bash
cd backend
python -m uvicorn app.main:app --port 8000 --reload
```
*The API will be available at [http://localhost:8000](http://localhost:8000) and documentation at [http://localhost:8000/docs](http://localhost:8000/docs).*

### Optional: Changing Datasets
The pipeline dynamically supports multiple datasets (CMAPSS and CruiseBench). You can switch between them by setting the `DATASET_TYPE` environment variable before starting the backend:
```bash
# Windows PowerShell
$env:DATASET_TYPE="cruisebench"
# Linux/macOS
export DATASET_TYPE="cruisebench"
```

## 3. Start the Frontend (Streamlit Dashboard)

Open a new, separate terminal (leave the backend running), ensure you are in the project root directory, and launch the Streamlit frontend:

```bash
python -m streamlit run frontend/dashboard.py
```
*The Streamlit dashboard will automatically open in your default browser at [http://localhost:8501](http://localhost:8501).*

> [!NOTE]
> On the first run, Streamlit may ask you for an email address to receive updates. You can simply leave the input blank and press `Enter` to bypass this prompt.
