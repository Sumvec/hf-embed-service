import logging
from sentence_transformers import SentenceTransformer
from .settings import settings

log = logging.getLogger(__name__)

class EmbedModel:
    def __init__(self):
        self.model = None

    def load(self):
        log.info(f"Loading model: {settings.MODEL_NAME}")
        self.model = SentenceTransformer(settings.MODEL_NAME)

    def encode(self, texts, batch_size: int | None = None):
        if self.model is None:
            raise RuntimeError("Model not loaded")
        bs = batch_size or settings.BATCH_SIZE
        return self.model.encode(texts, batch_size=bs, show_progress_bar=False, convert_to_numpy=True)

embed_model = EmbedModel()