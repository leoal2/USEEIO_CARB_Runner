import boto3
from botocore import UNSIGNED
from botocore.client import Config
import re

bucket_name = 'dmap-data-commons-ord'

def list_parquet_files(bucket_name=bucket_name, prefix='flowsa/FlowBySector/', state=False):
    # Create an S3 client with unsigned requests
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED), verify=False)
    if state:
        pattern = re.compile("^(.*)(GHG_state_[0-9]{4}_m1_.*parquet)$")
    else:
        pattern = re.compile("^(.*)(GHG_national_[0-9]{4}_m2_.*parquet)$")
    # List objects within the specified bucket and prefix
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    valid_files = []

    # Filter and print Parquet files
    if 'Contents' in response:
        parquet_files = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.parquet')]
        for file in parquet_files:
            if pattern.match(file):
                valid_files.append(pattern.match(file).groups()[1])
    else:
        print("No files found.")
    
    return valid_files

def list_state_files(bucket_name=bucket_name, prefix="stateio", return_tuple=False):
    # Create an S3 client with unsigned requests
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED), verify=False)
    pattern = re.compile("^(.*)(TwoRegion_Summary_Make_[0-9]{4}_.*)\\.rds$")
    # List objects within the specified bucket and prefix
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    valid_files = []
    # Filter and print Parquet files
    if 'Contents' in response:
        relevant_files = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.rds')]
        for file in relevant_files:
            if pattern.match(file):
                valid_files.append(pattern.match(file).groups()[1])
    else:
        print("No files found.")
    
    if return_tuple:
        year_ver = []
        pattern = re.compile("^.*_Make_([0-9]{4})_(.*)$")
        for file in valid_files:
            year_ver.append(pattern.match(file).groups())
        
        return year_ver
            
    else:
        return valid_files
# Specify your bucket name and prefix
