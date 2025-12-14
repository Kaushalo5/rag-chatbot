from app.core.config import settings
print('Gemini key exists:', bool(settings.GEMINI_API_KEY))
print('Gemini model:', settings.GEMINI_MODEL)
