FROM ultralytics/ultralytics:latest-jetson-jetpack6
COPY requirements.txt /app/
RUN python -m pip install --no-cache-dir -r /app/requirements.txt
EXPOSE 5010