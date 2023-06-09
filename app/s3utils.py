import boto3
import uuid
import os

from s3config import *

client_s3 = boto3.client(
    service_name = 's3',
    region_name = "ap-northeast-2",
    aws_access_key_id = AWS_ACCESS_KEY,
    aws_secret_access_key = AWS_SECRET_ACCESS_KEY
)

def upload_file(file_path, file_name):
    bucket_name = 'repost-bucket'
    uplaod_name = generate_upload_name(file_name)
    try:
        with open(file_path, 'rb') as f:
            client_s3.upload_fileobj(f, bucket_name, uplaod_name)
        return generate_url(uplaod_name)
    except Exception as e:
        print(f"error => {e}")
        return

def generate_url(upload_name):
    s = 'https://s3.ap-northeast-2.amazonaws.com'+'/repost-bucket/'+ upload_name
    return s

def generate_upload_name(file_name):
    s = 'image/'+ str(uuid.uuid4()) + get_file_extension(file_name)
    return s

def get_file_extension(filename):
    _, extension = os.path.splitext(filename)
    return extension.lower()