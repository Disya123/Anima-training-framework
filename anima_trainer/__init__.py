"""Architecture-aware training tools for the native latent-space Anima model."""

from .config import TrainerConfig, load_config
from .concepts import ConceptMode

__all__ = ["ConceptMode", "TrainerConfig", "load_config"]
__version__ = "0.1.0"

