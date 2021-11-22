FROM python:3.6

WORKDIR /APP

COPY ./ ./
COPY ../../metrics.json ./
COPY ../../temp/scaler.joblib ./
COPY ../../temp/model.joblib ./

RUN pip install -r requirements.txt

ENTRYPOINT [ "python" ]
CMD [ "app.py" ]