from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

previous_messages = []

def is_duplicate(message, threshold=0.8):
    # Lazy load model to avoid startup delays
    model = SentenceTransformer("all-MiniLM-L6-v2")

    global previous_messages

    emb = model.encode([message])

    for prev in previous_messages:
        sim = cosine_similarity(emb, prev)[0][0]

        if sim > threshold:
            return True

    previous_messages.append(emb)

    return False