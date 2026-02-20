"""
Azure Blob Storage utilities for saving training outputs.

This module provides functions to upload models, visualizations, logs,
and other training outputs to Azure Blob Storage.
"""

import os
from pathlib import Path
from typing import Optional, List, Union
import logging

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env file from project root (go up from src/ to project root)
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        logging.info(f"Loaded environment variables from {env_path}")
    else:
        # Also try current directory
        load_dotenv()
except ImportError:
    # python-dotenv not installed, skip .env loading
    pass

try:
    from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
    from azure.core.exceptions import AzureError
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    logging.warning("azure-storage-blob not installed. Azure storage features will be disabled.")


logger = logging.getLogger(__name__)


class AzureStorageManager:
    """
    Manager for uploading files to Azure Blob Storage.

    This class handles connection to Azure Blob Storage and provides
    methods to upload training outputs (models, visualizations, logs).
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
        account_name: Optional[str] = None,
        account_key: Optional[str] = None,
        container_name: str = "training-outputs",
        create_container_if_not_exists: bool = True
    ):
        """
        Initialize Azure Storage Manager.

        Args:
            connection_string: Azure Storage connection string
            account_name: Azure Storage account name (if using account_key)
            account_key: Azure Storage account key (if not using connection_string)
            container_name: Name of the blob container to use
            create_container_if_not_exists: If True, try to create container if it doesn't exist.
                                            If False, only use existing container.

        Note:
            Either connection_string OR (account_name + account_key) must be provided.
            If neither is provided, will try to get from environment variables:
            - AZURE_STORAGE_CONNECTION_STRING
            - AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_ACCOUNT_KEY
        """
        if not AZURE_AVAILABLE:
            raise ImportError(
                "azure-storage-blob is not installed. "
                "Install it with: pip install azure-storage-blob"
            )

        # Get credentials from parameters or environment variables
        if connection_string:
            self.connection_string = connection_string
        else:
            self.connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")

        if not self.connection_string:
            # Try to construct from account name and key
            account_name = account_name or os.environ.get("AZURE_STORAGE_ACCOUNT_NAME")
            account_key = account_key or os.environ.get("AZURE_STORAGE_ACCOUNT_KEY")

            if account_name and account_key:
                self.connection_string = (
                    f"DefaultEndpointsProtocol=https;"
                    f"AccountName={account_name};"
                    f"AccountKey={account_key};"
                    f"EndpointSuffix=core.windows.net"
                )
            else:
                raise ValueError(
                    "Azure Storage credentials not provided. "
                    "Set AZURE_STORAGE_CONNECTION_STRING or "
                    "AZURE_STORAGE_ACCOUNT_NAME + AZURE_STORAGE_ACCOUNT_KEY"
                )

        self.container_name = container_name
        self.create_container_if_not_exists = create_container_if_not_exists

        # Initialize blob service client
        try:
            self.blob_service_client = BlobServiceClient.from_connection_string(
                self.connection_string
            )
            # Create container if it doesn't exist (if enabled)
            if create_container_if_not_exists:
                self._ensure_container_exists()
            else:
                # Just verify container exists
                self._verify_container_exists()
            logger.info(f"Azure Storage Manager initialized. Container: {container_name}")
        except AzureError as e:
            error_code = getattr(e, 'error_code', None)
            if error_code == 'AuthorizationFailure':
                logger.error(
                    f"Authorization failed. Possible causes:\n"
                    f"  1. Invalid credentials in .env file\n"
                    f"  2. Credentials don't have permission to create containers\n"
                    f"  3. Container '{container_name}' doesn't exist and creation is disabled\n"
                    f"   Solution: Create the container manually via Azure Portal or CLI:\n"
                    f"   az storage container create --name {container_name} --account-name <your-account>"
                )
            else:
                logger.error(f"Failed to initialize Azure Storage: {e}")
            raise

    def _ensure_container_exists(self):
        """Create container if it doesn't exist."""
        try:
            container_client = self.blob_service_client.get_container_client(
                self.container_name
            )
            if not container_client.exists():
                container_client.create_container()
                logger.info(f"Created container: {self.container_name}")
            else:
                logger.info(f"Container {self.container_name} already exists")
        except AzureError as e:
            error_code = getattr(e, 'error_code', None)
            if error_code == 'AuthorizationFailure':
                # Container might already exist, try to verify
                try:
                    container_client = self.blob_service_client.get_container_client(
                        self.container_name
                    )
                    if container_client.exists():
                        logger.warning(
                            f"Cannot create container (authorization failed), "
                            f"but container '{self.container_name}' exists. Continuing..."
                        )
                        return
                except:
                    pass

                logger.error(
                    f"Authorization failed when creating container. "
                    f"Your credentials may not have permission to create containers.\n"
                    f"  1. Create the container manually via Azure Portal or CLI:\n"
                    f"     az storage container create --name {self.container_name} --account-name <your-account>\n"
                    f"  2. Or set create_container_if_not_exists=False when initializing AzureStorageManager"
                )
            raise

    def _verify_container_exists(self):
        """Verify that container exists."""
        try:
            container_client = self.blob_service_client.get_container_client(
                self.container_name
            )
            if not container_client.exists():
                raise ValueError(
                    f"Container '{self.container_name}' does not exist. "
                    f"Create it manually via Azure Portal or CLI:\n"
                    f"  az storage container create --name {self.container_name} --account-name <your-account>"
                )
        except AzureError as e:
            error_code = getattr(e, 'error_code', None)
            if error_code == 'AuthorizationFailure':
                logger.warning(
                    f"Cannot verify container existence (authorization failed). "
                    f"Assuming container exists and continuing..."
                )
            else:
                raise

    def upload_file(
        self,
        local_path: str,
        blob_name: Optional[str] = None,
        overwrite: bool = True
    ) -> str:
        """
        Upload a single file to Azure Blob Storage.

        Args:
            local_path: Path to the local file to upload
            blob_name: Name of the blob in Azure (default: same as local filename)
            overwrite: Whether to overwrite if blob already exists

        Returns:
            URL of the uploaded blob
        """
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"File not found: {local_path}")

        if blob_name is None:
            blob_name = local_path.name

        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )

            with open(local_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=overwrite)

            blob_url = blob_client.url
            logger.info(f"Uploaded {local_path} to {blob_name}")
            return blob_url
        except AzureError as e:
            logger.error(f"Failed to upload {local_path}: {e}")
            raise

    def upload_from_memory(
        self,
        data: bytes,
        blob_name: str,
        overwrite: bool = True
    ) -> str:
        """
        Upload data directly from memory to Azure Blob Storage.

        Args:
            data: Bytes data to upload
            blob_name: Name of the blob in Azure
            overwrite: Whether to overwrite if blob already exists

        Returns:
            URL of the uploaded blob
        """
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )

            blob_client.upload_blob(data, overwrite=overwrite)
            blob_url = blob_client.url
            logger.info(f"Uploaded data to {blob_name} ({len(data)} bytes)")
            return blob_url
        except AzureError as e:
            logger.error(f"Failed to upload data to {blob_name}: {e}")
            raise

    def download_blob(
        self,
        blob_name: str,
        local_path: Optional[str] = None
    ) -> Union[bytes, Path]:
        """
        Download a blob from Azure Blob Storage.

        Args:
            blob_name: Name of the blob in Azure
            local_path: If provided, save to this file path and return the Path.
                        If None, return the raw bytes (useful for loading in memory).

        Returns:
            If local_path is provided: Path to the downloaded file.
            Otherwise: bytes content of the blob.
        """
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )

            download_stream = blob_client.download_blob()

            if local_path is not None:
                local_path = Path(local_path)
                local_path.parent.mkdir(parents=True, exist_ok=True)
                with open(local_path, "wb") as f:
                    f.write(download_stream.readall())
                logger.info(f"Downloaded {blob_name} to {local_path}")
                return local_path
            else:
                data = download_stream.readall()
                logger.info(f"Downloaded {blob_name} ({len(data)} bytes to memory)")
                return data
        except AzureError as e:
            logger.error(f"Failed to download {blob_name}: {e}")
            raise

    def upload_directory(
        self,
        local_dir: str,
        blob_prefix: Optional[str] = None,
        pattern: str = "*",
        recursive: bool = True
    ) -> List[str]:
        """
        Upload all files from a directory to Azure Blob Storage.

        Args:
            local_dir: Path to the local directory to upload
            blob_prefix: Prefix to add to blob names (default: directory name)
            pattern: Glob pattern to match files (default: "*")
            recursive: Whether to upload files recursively

        Returns:
            List of URLs of uploaded blobs
        """
        local_dir = Path(local_dir)
        if not local_dir.exists() or not local_dir.is_dir():
            raise ValueError(f"Directory not found: {local_dir}")

        if blob_prefix is None:
            blob_prefix = local_dir.name

        uploaded_urls = []

        # Find all files matching the pattern
        if recursive:
            files = list(local_dir.rglob(pattern))
        else:
            files = list(local_dir.glob(pattern))

        # Filter to only files (not directories)
        files = [f for f in files if f.is_file()]

        for file_path in files:
            # Calculate relative path from local_dir
            relative_path = file_path.relative_to(local_dir)
            blob_name = f"{blob_prefix}/{relative_path}".replace("\\", "/")

            try:
                url = self.upload_file(str(file_path), blob_name)
                uploaded_urls.append(url)
            except Exception as e:
                logger.warning(f"Failed to upload {file_path}: {e}")

        logger.info(f"Uploaded {len(uploaded_urls)} files from {local_dir}")
        return uploaded_urls

    def upload_training_outputs(
        self,
        model_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        logs_dir: Optional[str] = None,
        run_name: Optional[str] = None
    ) -> dict:
        """
        Upload all training outputs to Azure Blob Storage.

        Args:
            model_path: Path to the saved model file
            output_dir: Directory containing visualizations and outputs
            logs_dir: Directory containing TensorBoard logs
            run_name: Name for this training run (used as prefix)

        Returns:
            Dictionary with upload results and URLs
        """
        results = {
            "model_url": None,
            "outputs_urls": [],
            "logs_urls": [],
            "run_name": run_name
        }

        try:
            # Upload model
            if model_path and Path(model_path).exists():
                model_blob_name = f"{run_name}/model/{Path(model_path).name}" if run_name else f"model/{Path(model_path).name}"
                results["model_url"] = self.upload_file(model_path, model_blob_name)
                logger.info(f"Model uploaded: {results['model_url']}")

            # Upload outputs directory
            if output_dir and Path(output_dir).exists():
                output_prefix = f"{run_name}/outputs" if run_name else "outputs"
                results["outputs_urls"] = self.upload_directory(
                    output_dir,
                    blob_prefix=output_prefix
                )
                logger.info(f"Uploaded {len(results['outputs_urls'])} output files")

            # Upload logs directory
            if logs_dir and Path(logs_dir).exists():
                logs_prefix = f"{run_name}/logs" if run_name else "logs"
                results["logs_urls"] = self.upload_directory(
                    logs_dir,
                    blob_prefix=logs_prefix
                )
                logger.info(f"Uploaded {len(results['logs_urls'])} log files")

            return results
        except Exception as e:
            logger.error(f"Failed to upload training outputs: {e}")
            raise


def upload_to_azure(
    model_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    logs_dir: Optional[str] = None,
    run_name: Optional[str] = None,
    container_name: str = "training-outputs",
    connection_string: Optional[str] = None,
    create_container_if_not_exists: bool = True
) -> dict:
    """
    Convenience function to upload training outputs to Azure.

    Args:
        model_path: Path to the saved model file
        output_dir: Directory containing visualizations and outputs
        logs_dir: Directory containing TensorBoard logs
        run_name: Name for this training run
        container_name: Azure container name
        connection_string: Azure connection string (optional, uses env var if not provided)
        create_container_if_not_exists: If True, try to create container if it doesn't exist

    Returns:
        Dictionary with upload results
    """
    manager = AzureStorageManager(
        connection_string=connection_string,
        container_name=container_name,
        create_container_if_not_exists=create_container_if_not_exists
    )

    return manager.upload_training_outputs(
        model_path=model_path,
        output_dir=output_dir,
        logs_dir=logs_dir,
        run_name=run_name
    )
