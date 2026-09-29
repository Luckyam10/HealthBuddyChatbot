"""
model_utils.py
--------------

Runtime semantic engine for HealthBuddy.

Preferred model:
    model_semantic.joblib

Fallback:
    model.joblib

The semantic model uses sentence embeddings and cosine similarity
against semantic intent references.

This is NOT an if/else classifier for individual user phrases.

The only explicit rule is the emergency safety override.
"""

import json
import os
import random

import joblib
import numpy as np


DEFAULT_SIMILARITY_THRESHOLD = 0.43


FALLBACK_MESSAGE = (
    "I'm sorry, I didn't quite understand that. "
    "I can help with general health information, wellness, "
    "sleep, nutrition, exercise, appointments, or clinic hours. "
    "Could you rephrase?"
)


# ---------------------------------------------------------------------
# Emergency safety layer
# ---------------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # Files
    # -----------------------------------------------------------------

    def _path(self, name):

        return os.path.join(
            self.base_dir,
            name,
        )

    # -----------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------

    def _load_intents(self):

        with open(
            self._path("intents.json"),
            "r",
            encoding="utf-8",
        ) as f:

            return json.load(f)

    # -----------------------------------------------------------------
    # Model loading
    # -----------------------------------------------------------------

    def _load_model(self):

        semantic_path = self._path(
            "model_semantic.joblib"
        )

        tfidf_path = self._path(
            "model.joblib"
        )

        if os.path.exists(semantic_path):

            data = joblib.load(
                semantic_path
            )

            return (
                "semantic",
                data,
            )

        if os.path.exists(tfidf_path):

            pipeline = joblib.load(
                tfidf_path
            )

            return (
                "tfidf",
                pipeline,
            )

        raise FileNotFoundError(
            "No HealthBuddy model was found.\n\n"
            "Run:\n"
            "python train_model_semantic.py\n\n"
            "or train the TF-IDF fallback model."
        )

    # -----------------------------------------------------------------
    # Sentence Transformer
    # -----------------------------------------------------------------

    def _get_embedder(self):

        if self._embedder is None:

            from sentence_transformers import SentenceTransformer

            embedder_name = self.model.get(
                "embedder_name",
                "all-MiniLM-L6-v2",
            )

            self._embedder = SentenceTransformer(
                embedder_name,
                device="cpu",
            )

        return self._embedder

    # -----------------------------------------------------------------
    # Emergency detection
    # -----------------------------------------------------------------

    def _is_emergency(self, text):

        lowered = text.lower()

        return any(
            keyword in lowered
            for keyword in EMERGENCY_KEYWORDS
        )

    # -----------------------------------------------------------------
    # Semantic prediction
    # -----------------------------------------------------------------

    def _semantic_predict(self, text):

        embedder = self._get_embedder()

        query_embedding = embedder.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        references = self.model[
            "intent_references"
        ]

        scores = {}

        for intent_name, data in references.items():

            reference_embeddings = np.asarray(
                data["embeddings"],
                dtype=np.float32,
            )

            similarities = np.dot(
                reference_embeddings,
                query_embedding,
            )

            scores[intent_name] = float(
                np.max(similarities)
            )

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        best_intent = ranked[0][0]
        best_similarity = ranked[0][1]

        threshold = float(
            self.model.get(
                "similarity_threshold",
                DEFAULT_SIMILARITY_THRESHOLD,
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

    # -----------------------------------------------------------------
    # TF-IDF fallback
    # -----------------------------------------------------------------

    def _tfidf_predict(self, text):

        probabilities = self.model.predict_proba(
            [text]
        )[0]

        classes = self.model.classes_

        best_index = np.argmax(
            probabilities
        )

        intent = classes[best_index]

        confidence = float(
            probabilities[best_index]
        )

        return {
            "intent": intent,
            "confidence": confidence,
            "scores": dict(
                zip(
                    classes,
                    probabilities,
                )
            ),
        }

    # -----------------------------------------------------------------
    # Public prediction API
    # -----------------------------------------------------------------

    def predict(self, user_text: str):

        text = (
            user_text or ""
        ).strip()

        if not text:

            return {
                "response": FALLBACK_MESSAGE,
                "intent": "fallback",
                "confidence": 0.0,
                "source": "fallback",
                "model_type": self.model_type,
            }

        # -------------------------------------------------------------
        # Safety layer
        # -------------------------------------------------------------

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

        # -------------------------------------------------------------
        # Semantic understanding
        # -------------------------------------------------------------

        if self.model_type == "semantic":

            result = self._semantic_predict(
                text
            )

        else:

            result = self._tfidf_predict(
                text
            )

        intent = result["intent"]

        confidence = float(
            result["confidence"]
        )

        # -------------------------------------------------------------
        # Unknown meaning
        # -------------------------------------------------------------

        if intent == "fallback":

            return {
                "response": FALLBACK_MESSAGE,
                "intent": "fallback",
                "confidence": confidence,
                "source": "semantic_fallback",
                "model_type": self.model_type,
            }

        # -------------------------------------------------------------
        # Known semantic meaning
        # -------------------------------------------------------------

        if intent not in self.intents:

            return {
                "response": FALLBACK_MESSAGE,
                "intent": "fallback",
                "confidence": confidence,
                "source": "invalid_intent_fallback",
                "model_type": self.model_type,
            }

        response = random.choice(
            self.intents[
                intent
            ]["responses"]
        )

        return {
            "response": response,
            "intent": intent,
            "confidence": confidence,
            "source": "semantic_model",
            "model_type": self.model_type,
        }

