"""
model_utils.py
--------------

Runtime intent engine for HealthBuddy.

Preferred runtime model:
    model.joblib

Fallback:
    model_semantic.joblib

The TF-IDF model is preferred because it uses significantly
less memory than Sentence Transformers on limited deployment services.
"""

import json
import os
import random

import joblib
import numpy as np


# =========================================================
# SETTINGS
# =========================================================

# If the classifier is less confident than this,
# HealthBuddy will ask the user to rephrase instead
# of forcing an incorrect intent.
CONFIDENCE_THRESHOLD = 0.30


# Very short messages can produce unreliable TF-IDF predictions.
# These are allowed through because they are common conversational
# messages with clear meanings.
SHORT_MESSAGE_INTENTS = {
    "hi": "greeting",
    "hey": "greeting",
    "hello": "greeting",
    "yo": "greeting",
    "sup": "greeting",
    "hiya": "greeting",

    "bye": "goodbye",
    "goodbye": "goodbye",
    "thanks": "thanks",
    "thank you": "thanks",
}


FALLBACK_MESSAGE = (
    "I'm sorry, I didn't quite understand that. "
    "I can help with general health information, wellness, "
    "sleep, nutrition, exercise, appointments, or clinic hours. "
    "Could you rephrase?"
)


# =========================================================
# EMERGENCY SAFETY KEYWORDS
# =========================================================

EMERGENCY_KEYWORDS = [
    "can't breathe",
    "cant breathe",
    "cannot breathe",
    "difficulty breathing",
    "trouble breathing",
    "hard to breathe",
    "not breathing",
    "chest pain",
    "severe chest pain",
    "unconscious",
    "severe bleeding",
    "heavy bleeding",
    "heart attack",
    "stroke",
    "collapsed",
    "cardiac arrest",
    "choking",
    "severe allergic reaction",
    "not responding",
    "life threatening",
]


class IntentPredictor:

    def __init__(self, base_dir="."):
        self.base_dir = base_dir
        self.intents = self._load_intents()
        self.model_type, self.model = self._load_model()
        self._embedder = None

    # =====================================================
    # FILE HELPERS
    # =====================================================

    def _path(self, name):
        return os.path.join(self.base_dir, name)

    def _load_intents(self):
        with open(
            self._path("intents.json"),
            "r",
            encoding="utf-8"
        ) as f:
            return json.load(f)

    # =====================================================
    # MODEL LOADING
    # =====================================================

    def _load_model(self):

        semantic_path = self._path("model_semantic.joblib")
        tfidf_path = self._path("model.joblib")

        # Always prefer lightweight TF-IDF model
        if os.path.exists(tfidf_path):
            pipeline = joblib.load(tfidf_path)
            return "tfidf", pipeline

        # Semantic model remains available as fallback
        if os.path.exists(semantic_path):
            data = joblib.load(semantic_path)
            return "semantic", data

        raise FileNotFoundError(
            "No HealthBuddy model was found."
        )

    # =====================================================
    # SEMANTIC MODEL
    # =====================================================

    def _get_embedder(self):

        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            embedder_name = self.model.get(
                "embedder_name",
                "all-MiniLM-L6-v2"
            )

            self._embedder = SentenceTransformer(
                embedder_name,
                device="cpu"
            )

        return self._embedder

    # =====================================================
    # EMERGENCY CHECK
    # =====================================================

    def _is_emergency(self, text):

        lowered = text.lower()

        return any(
            keyword in lowered
            for keyword in EMERGENCY_KEYWORDS
        )

    # =====================================================
    # SIMPLE CONVERSATIONAL CHECK
    # =====================================================

    def _simple_message(self, text):

        normalized = " ".join(
            text.lower().strip().split()
        )

        return SHORT_MESSAGE_INTENTS.get(normalized)

    # =====================================================
    # TF-IDF PREDICTION
    # =====================================================

    def _tfidf_predict(self, text):

        probabilities = self.model.predict_proba([text])[0]
        classes = self.model.classes_

        best_index = np.argmax(probabilities)

        intent = classes[best_index]
        confidence = float(probabilities[best_index])

        scores = dict(
            zip(classes, probabilities)
        )

        # -------------------------------------------------
        # IMPORTANT:
        # Do not force a prediction when confidence is low.
        # -------------------------------------------------

        if confidence < CONFIDENCE_THRESHOLD:

            return {
                "intent": "fallback",
                "confidence": confidence,
                "scores": scores,
            }

        return {
            "intent": intent,
            "confidence": confidence,
            "scores": scores,
        }

    # =====================================================
    # SEMANTIC PREDICTION
    # =====================================================

    def _semantic_predict(self, text):

        embedder = self._get_embedder()

        query_embedding = embedder.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        references = self.model["intent_references"]

        scores = {}

        for intent_name, data in references.items():

            reference_embeddings = np.asarray(
                data["embeddings"],
                dtype=np.float32
            )

            similarities = np.dot(
                reference_embeddings,
                query_embedding
            )

            scores[intent_name] = float(
                np.max(similarities)
            )

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        best_intent = ranked[0][0]
        best_similarity = ranked[0][1]

        threshold = float(
            self.model.get(
                "similarity_threshold",
                0.43
            )
        )

        if best_similarity < threshold:

            return {
                "intent": "fallback",
                "confidence": best_similarity,
                "scores": scores,
            }

        return {
            "intent": best_intent,
            "confidence": best_similarity,
            "scores": scores,
        }

    # =====================================================
    # MAIN PREDICTION
    # =====================================================

    def predict(self, user_text: str):

        text = (user_text or "").strip()

        # -------------------------------------------------
        # Empty message
        # -------------------------------------------------

        if not text:

            return {
                "response": FALLBACK_MESSAGE,
                "intent": "fallback",
                "confidence": 0.0,
                "source": "fallback",
                "model_type": self.model_type,
            }

        # -------------------------------------------------
        # Emergency safety override
        # -------------------------------------------------

        if self._is_emergency(text):

            response = random.choice(
                self.intents[
                    "emergency_redirect"
                ]["responses"]
            )

            return {
                "response": response,
                "intent": "emergency_redirect",
                "confidence": 1.0,
                "source": "emergency_safety",
                "model_type": self.model_type,
            }

        # -------------------------------------------------
        # Known short conversational messages
        # -------------------------------------------------

        simple_intent = self._simple_message(text)

        if simple_intent:

            response = random.choice(
                self.intents[
                    simple_intent
                ]["responses"]
            )

            return {
                "response": response,
                "intent": simple_intent,
                "confidence": 1.0,
                "source": "conversation_rule",
                "model_type": self.model_type,
            }

        # -------------------------------------------------
        # Model prediction
        # -------------------------------------------------

        if self.model_type == "semantic":

            result = self._semantic_predict(text)

        else:

            result = self._tfidf_predict(text)

        intent = result["intent"]
        confidence = float(
            result["confidence"]
        )

        # -------------------------------------------------
        # Model fallback
        # -------------------------------------------------

        if intent == "fallback":

            return {
                "response": FALLBACK_MESSAGE,
                "intent": "fallback",
                "confidence": confidence,
                "source": "confidence_fallback",
                "model_type": self.model_type,
            }

        # -------------------------------------------------
        # Invalid intent protection
        # -------------------------------------------------

        if intent not in self.intents:

            return {
                "response": FALLBACK_MESSAGE,
                "intent": "fallback",
                "confidence": confidence,
                "source": "invalid_intent_fallback",
                "model_type": self.model_type,
            }

        # -------------------------------------------------
        # Normal response
        # -------------------------------------------------

        response = random.choice(
            self.intents[intent]["responses"]
        )

        return {
            "response": response,
            "intent": intent,
            "confidence": confidence,
            "source": "tfidf_model"
            if self.model_type == "tfidf"
            else "semantic_model",
            "model_type": self.model_type,
        }
