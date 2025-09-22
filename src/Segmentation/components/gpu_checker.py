import os
import torch
from packaging import version
from src.Segmentation import logger
from src.Segmentation.entity.config_entity import GPUCheckerConfig


# ======================================================
# Class to check GPU availability and PyTorch/CUDA versions
# ======================================================
class GPUChecker:
    def __init__(self, config: GPUCheckerConfig):
        """
        Initialize GPUChecker with a configuration object.

        Args:
            config (GPUCheckerConfig): Configuration including:
                - fail_on_no_gpu: whether to raise an error if GPU is not available
                - required_min_cuda_version: minimum CUDA version required
                - required_min_pytorch_version: minimum PyTorch version required
        """
        self.config = config
        # Ensure CUDA errors are immediately reported (for debugging)
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    def check_torch_version(self):
        """
        Check the installed PyTorch version against the required minimum.

        Raises:
            RuntimeError: If installed PyTorch version is lower than required.

        Returns:
            str: Installed PyTorch version.
        """
        try:
            installed_version = torch.__version__
            logger.info(f"Installed PyTorch version: {installed_version}")

            if self.config.required_min_pytorch_version:
                if version.parse(installed_version) < version.parse(self.config.required_min_pytorch_version):
                    message = (f"❌ Installed PyTorch version {installed_version} "
                               f"is LOWER than required minimum {self.config.required_min_pytorch_version}")
                    logger.error(message)
                    raise RuntimeError(message)
                logger.info(f"✅ PyTorch version check passed (>= {self.config.required_min_pytorch_version})")

            return installed_version
        except Exception as e:
            logger.exception("❌ Failed to check PyTorch version")
            raise

    def check_gpu_availability(self):
        """
        Check if GPU is available and meets the required CUDA version.

        Logs information about each detected GPU.

        Raises:
            RuntimeError: If GPU is not available and fail_on_no_gpu is True,
                          or if CUDA version is lower than required.

        Returns:
            bool: True if GPU is available and valid, False otherwise.
        """
        try:
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                logger.info("✅ GPU is available!")
                logger.info(f"Number of GPUs: {num_gpus}")

                # Log details for each GPU
                for i in range(num_gpus):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_capability = torch.cuda.get_device_capability(i)
                    logger.info(f"  GPU {i}: {gpu_name} (Compute Capability: {gpu_capability})")

                # Check CUDA version against minimum requirement
                if self.config.required_min_cuda_version:
                    cuda_version = torch.version.cuda
                    if version.parse(cuda_version) < version.parse(self.config.required_min_cuda_version):
                        message = (f"❌ Installed CUDA version {cuda_version} "
                                   f"is LOWER than required minimum {self.config.required_min_cuda_version}")
                        logger.error(message)
                        raise RuntimeError(message)
                    logger.info(f"✅ CUDA version check passed (>= {self.config.required_min_cuda_version})")

                return True
            else:
                # GPU not available
                message = "❌ GPU is NOT available."
                logger.error(message)
                if self.config.fail_on_no_gpu:
                    raise RuntimeError(message)
                return False
        except Exception as e:
            logger.exception("❌ Unexpected error while checking GPU")
            raise
