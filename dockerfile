# ✅ Use official lightweight Python image
FROM python:3.10-slim

# ✅ Set working directory
WORKDIR /app

# ✅ Copy requirements first
COPY requirements.txt .

# ✅ Upgrade pip and install dependencies (including CPU-only PyTorch)
RUN pip install --upgrade pip \
 && pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# ✅ Copy all app files
COPY . .

# ✅ Expose Streamlit port
EXPOSE 8501

# ✅ Environment variables
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_HEADLESS=true

# ✅ Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
