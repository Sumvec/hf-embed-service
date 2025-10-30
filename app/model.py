import logging
from sentence_transformers import SentenceTransformer
from .settings import settings

log = logging.getLogger(__name__)

MODEL_NAMES = [
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/all-MiniLM-L6-v2"
]
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class EmbedModel:
    def __init__(self):
        self.models = {}

    def load(self):
        for name in MODEL_NAMES:
            log.info(f"Loading model: {name}")
            self.models[name] = SentenceTransformer(name)

    def encode(self, texts, batch_size: int | None = None, model_name: str | None = None):
        name = model_name or DEFAULT_MODEL_NAME
        model = self.models.get(name)
        if model is None:
            raise RuntimeError(f"Model '{name}' not loaded")
        bs = batch_size or settings.BATCH_SIZE
        return model.encode(texts, batch_size=bs, show_progress_bar=False, convert_to_numpy=True)

embed_model = EmbedModel()