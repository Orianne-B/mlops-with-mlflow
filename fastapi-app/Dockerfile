# Get python image
FROM python:3.10

# Set working directory
WORKDIR /app

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

EXPOSE 5089

CMD ["uvicorn","app:app","--host","0.0.0.0","--port","5089"]
