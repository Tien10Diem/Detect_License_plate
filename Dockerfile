FROM ultralytics/ultralytics:8.4.16

WORKDIR /lp

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY src src
COPY scripts scripts
COPY weights weights

CMD ["python", "src/pipeline/inference.py"]
