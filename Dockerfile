
#aws python Lambda System
FROM public.ecr.aws/lambda/python:3.11

# Install the function's dependencies using file requirements.txt
# from your project folder.
COPY requirements.txt  .

RUN  pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

#Urllib3 is upgraded due to compatibility issues with BOTO3 and LANGCHAIN
#RUN pip install --upgrade urllib3 --target "${LAMBDA_TASK_ROOT}"

ENV PINECONE_API_KEY="your pinecone api key"
ENV OPENAI_API_KEY="your openai api key"
ENV PINECONE_ENV="your pinecone environment name"

# Copy function code
COPY app.py ${LAMBDA_TASK_ROOT}

# Expose the port on which your API will run (assuming it is using port 5000)
#EXPOSE 5000

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "app.readfile_and_createembeddings" ] 