FROM python:3.10

WORKDIR /app

COPY . /app

RUN pip install mlflow scikit-learn numpy

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "./mlruns"]