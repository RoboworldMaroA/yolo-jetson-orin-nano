FROM ultralytics/ultralytics:latest-jetson-jetpack6
RUN python -m pip install --no-cache-dir flask
EXPOSE 5001