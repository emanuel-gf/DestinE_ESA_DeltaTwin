import os
import boto3
import requests
from urllib.parse import parse_qs, urlparse
from lxml import html


class S3Connector:
    """A clean connector for S3-compatible storage services."""

    def __init__(
        self,
        endpoint_url: str,
        access_key_id: str,
        secret_access_key: str,
        region_name: str = 'default'
    ) -> None:
        """Initialize the S3Connector with connection parameters."""
        self.endpoint_url = endpoint_url
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.region_name = region_name

        # Create session
        self.session = boto3.session.Session()

        # Initialize S3 resource and client
        self.s3 = self._create_s3_resource()
        self.s3_client = self._create_s3_client()

    def _create_s3_resource(self):
        return self.session.resource(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            region_name=self.region_name
        )

    def _create_s3_client(self):
        return self.session.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            region_name=self.region_name
        )

    def get_s3_client(self):
        """Get the boto3 S3 client."""
        return self.s3_client

    def get_s3_resource(self):
        """Get the boto3 S3 resource."""
        return self.s3

    def get_bucket(self, bucket_name: str):
        """Get a specific bucket by name."""
        return self.s3.Bucket(bucket_name)

    def list_buckets(self) -> list:
        """List all available buckets."""
        response = self.s3_client.list_buckets()
        return [bucket['Name'] for bucket in response.get('Buckets', [])]
