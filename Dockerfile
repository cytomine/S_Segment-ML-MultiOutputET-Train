FROM cytomine/software-python3-base:v2.2.2

# --------------------------------------------------------------------------------------------
# Instal Pyxit
RUN pip install pyxit==1.1.5

# --------------------------------------------------------------------------------------------
ADD descriptor.json /app/descriptor.json
ADD run.py /app/run.py

ENTRYPOINT ["python", "/app/run.py"]
