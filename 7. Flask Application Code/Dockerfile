# Instructions copied from - https://hub.docker.com/_/python/
FROM python:3.6-onbuild

#Installing dependencies
RUN pip install pip
RUN pip install pandas
RUN pip install flask
RUN pip install scikit-learn

#Copying the Application into the Image's File System
COPY static /src/static
COPY templates /src/templates
COPY app.py /src/app.py
COPY requirements.txt /src/requirements.txt
COPY RF_Reg.pkl /src/RF_Reg.pkl

EXPOSE 5000

ENTRYPOINT ["python"]

CMD ["app.py"]