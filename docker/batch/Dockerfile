FROM python:3.6
EXPOSE 5001
WORKDIR /APP

COPY ./ ./

RUN pip install -r requirements.txt

ENTRYPOINT [ "python" ]
CMD [ "app.py" ]