import boto3
import os

lambda_client = boto3.client('lambda', region_name=os.getenv('AWS_REGION', 'us-east-1'))
