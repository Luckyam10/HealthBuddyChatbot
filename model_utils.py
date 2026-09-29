"""
model_utils.py
--------------

Runtime intent engine for HealthBuddy.

Preferred model:
    model.joblib

Fallback:
    model_semantic.joblib

The TF-IDF model is preferred because it is lightweight and
does not require loading Sentence Transformers during normal
runtime operation.

HealthBuddy provides general health information only.
It does not diagnose medical conditions.
"""

import json
import os
import random
import re

import joblib
import numpy as np


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.30

DEFAULT_SIMILARITY_THRESHOLD = 0.43


FALLBACK_MESSAGE = (
    "I'm sorry, I didn't quite understand that. "
    "I can help with general health information, common symptoms, "
    "wellness, sleep, nutrition, exercise, appointments, or clinic hours. "
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
    # Dataset / responses
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

        tfidf_path = self._path(
            "model.joblib"
        )

        semantic_path = self._path(
            "model_semantic.joblib"
        )

        # -------------------------------------------------------------
        # Prefer TF-IDF
        # -------------------------------------------------------------

        if os.path.exists(tfidf_path):

            pipeline = joblib.load(
                tfidf_path
            )

            return (
                "tfidf",
                pipeline,
            )

        # -------------------------------------------------------------
        # Semantic fallback
        # -------------------------------------------------------------

        if os.path.exists(semantic_path):

            data = joblib.load(
                semantic_path
            )

            return (
                "semantic",
                data,
            )

        raise FileNotFoundError(
            "No HealthBuddy model was found.\n\n"
            "Expected either:\n"
            "  model.joblib\n"
            "or:\n"
            "  model_semantic.joblib"
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
    # TF-IDF prediction
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

        scores = dict(
            zip(
                classes,
                probabilities,
            )
        )

        # -------------------------------------------------------------
        # Confidence threshold
        # -------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # Response generation
    # -----------------------------------------------------------------

    def _get_response(self, intent):

        # -------------------------------------------------------------
        # Normal intent response
        # -------------------------------------------------------------

        if intent in self.intents:

            data = self.intents[intent]

            responses = data.get(
                "responses",
                [],
            )

            if responses:

                return random.choice(
                    responses
                )

        # -------------------------------------------------------------
        # Temporary response for newly trained intents
        # -------------------------------------------------------------

        return self._generic_health_response(
            intent
        )

    # -----------------------------------------------------------------
    # Generic responses for expanded symptom intents
    # -----------------------------------------------------------------

    def _generic_health_response(self, intent):

        responses = {

            "headache": (
                "Headaches can have many common causes, such as "
                "stress, lack of sleep, dehydration, or illness. "
                "If the headache is severe, unusual, persistent, "
                "or comes with concerning symptoms, consider seeking "
                "medical advice."
            ),

            "fever": (
                "Fever can occur with infections and other illnesses. "
                "Rest, fluids, and monitoring your symptoms can be helpful. "
                "Seek medical advice if the fever is severe, persistent, "
                "or accompanied by concerning symptoms."
            ),

            "cough": (
                "Coughing can happen with colds, flu, allergies, "
                "or other respiratory conditions. General supportive "
                "care and monitoring may help. If it is severe, persistent, "
                "or accompanied by breathing difficulty, seek medical advice."
            ),

            "sore_throat": (
                "A sore throat can occur with colds, flu, allergies, "
                "or other infections. Rest and staying hydrated may help. "
                "If symptoms are severe or persistent, consider consulting "
                "a healthcare professional."
            ),

            "runny_nose": (
                "A runny nose can occur with colds, allergies, and other "
                "respiratory conditions. Rest, fluids, and monitoring "
                "your symptoms may help."
            ),

            "congestion": (
                "Nasal congestion can occur with colds, allergies, "
                "or other respiratory conditions. Staying hydrated and "
                "resting may help. Persistent or severe symptoms should "
                "be discussed with a healthcare professional."
            ),

            "dizziness": (
                "Dizziness can have many possible causes, including "
                "dehydration, illness, or changes in blood pressure. "
                "If it is severe, recurring, or accompanied by concerning "
                "symptoms, seek medical advice."
            ),

            "fatigue": (
                "Feeling tired can be related to sleep, stress, illness, "
                "nutrition, or other factors. If fatigue continues or "
                "interferes with daily activities, consider discussing "
                "it with a healthcare professional."
            ),

            "weakness": (
                "Weakness can have many possible causes. Rest and hydration "
                "may be helpful, but persistent or significant weakness "
                "should be discussed with a healthcare professional."
            ),

            "body_pain": (
                "Body aches can occur with infections, physical activity, "
                "stress, or other conditions. Rest and hydration may help. "
                "Persistent or severe pain should be evaluated by a "
                "healthcare professional."
            ),

            "back_pain": (
                "Back pain can have many causes, including strain or posture. "
                "Gentle activity and avoiding movements that worsen the pain "
                "may help. Persistent or severe pain should be evaluated."
            ),

            "joint_pain": (
                "Joint pain can have several possible causes, including "
                "strain, injury, or inflammation. If the pain persists, "
                "worsens, or limits movement, consider medical advice."
            ),

            "stomach_pain": (
                "Stomach pain can have many causes. Pay attention to when "
                "it happens and whether other symptoms occur. Severe, "
                "persistent, or worsening abdominal pain should be "
                "evaluated by a healthcare professional."
            ),

            "nausea": (
                "Nausea can occur with illness, food-related problems, "
                "motion sickness, or other causes. Staying hydrated and "
                "eating lightly may help some people. Persistent or severe "
                "nausea should be discussed with a healthcare professional."
            ),

            "vomiting": (
                "Vomiting can happen for many reasons, including illness "
                "or food-related problems. Staying hydrated is important. "
                "If vomiting is persistent, severe, or you cannot keep "
                "fluids down, seek medical advice."
            ),

            "diarrhea": (
                "Diarrhea can have several causes, including infections "
                "and food-related problems. Staying hydrated is important. "
                "Persistent or severe diarrhea should be discussed with "
                "a healthcare professional."
            ),

            "constipation": (
                "Constipation can sometimes be related to diet, hydration, "
                "activity, or changes in routine. Adequate fluids and "
                "fiber may help. Persistent constipation should be "
                "discussed with a healthcare professional."
            ),

            "acne": (
                "Acne is common and can be influenced by hormones, skin "
                "oil, and other factors. Gentle skin care can help. "
                "Persistent or severe acne may benefit from advice from "
                "a healthcare professional."
            ),

            "rash": (
                "Rashes can have many possible causes, including irritation, "
                "allergies, or infections. Avoiding known irritants may help. "
                "A persistent, worsening, or concerning rash should be "
                "evaluated by a healthcare professional."
            ),

            "itching": (
                "Itching can result from dry skin, irritation, allergies, "
                "or other causes. Avoiding irritants and keeping the skin "
                "moisturized may help. Persistent or severe itching should "
                "be evaluated."
            ),

            "allergy": (
                "Allergy symptoms can vary and may include sneezing, itching, "
                "or a runny nose. Identifying and avoiding known triggers "
                "may help. Severe allergic reactions require urgent medical "
                "attention."
            ),

            "ear_pain": (
                "Ear pain can have several possible causes, including "
                "irritation or infection. Avoid putting objects into the "
                "ear. Persistent or severe ear pain should be evaluated "
                "by a healthcare professional."
            ),

            "eye_symptoms": (
                "Eye symptoms can have many causes, including irritation, "
                "allergies, or infection. Avoid rubbing the eyes and monitor "
                "the symptoms. Persistent or concerning eye symptoms should "
                "be evaluated."
            ),

            "toothache": (
                "Tooth pain can be caused by dental problems such as decay "
                "or irritation. Maintaining oral hygiene is important, but "
                "persistent tooth pain should be evaluated by a dentist."
            ),

            "heartburn": (
                "Heartburn is often associated with stomach acid moving "
                "upward into the esophagus. Some people find that avoiding "
                "large meals and known trigger foods helps. Frequent or "
                "severe symptoms should be discussed with a healthcare "
                "professional."
            ),

            "indigestion": (
                "Indigestion can cause discomfort, fullness, or bloating "
                "after eating. Eating slowly and avoiding foods that "
                "trigger symptoms may help. Persistent symptoms should "
                "be discussed with a healthcare professional."
            ),

            "hair_loss": (
                "Hair loss can have many possible causes, including stress, "
                "hormonal changes, nutrition, or other conditions. "
                "Persistent or significant hair loss can be discussed "
                "with a healthcare professional."
            ),

            "motion_sickness": (
                "Motion sickness can cause nausea, dizziness, or discomfort "
                "during travel. Looking toward the horizon and getting "
                "fresh air may help some people."
            ),

            "urinary_symptoms": (
                "Urinary symptoms can have different causes. If you have "
                "painful urination, persistent symptoms, blood in the urine, "
                "or other concerning symptoms, consider seeking medical advice."
            ),

            "menstrual_health": (
                "Menstrual symptoms can vary from person to person. "
                "Tracking symptoms and maintaining regular rest, hydration, "
                "and nutrition may help. Severe or unusually persistent "
                "symptoms should be discussed with a healthcare professional."
            ),

            "sunburn": (
                "Sunburn can cause red, irritated, and painful skin. "
                "Staying out of direct sunlight and keeping the affected "
                "area comfortable may help. Severe sunburn should be "
                "evaluated by a healthcare professional."
            ),

            "skin_care": (
                "Basic skin care usually includes gentle cleansing, "
                "moisturizing, and protecting the skin from excessive "
                "sun exposure. If you have persistent skin concerns, "
                "consider consulting a healthcare professional."
            ),

            "breathing_information": (
                "Breathing symptoms can have many possible causes. "
                "If you are having severe difficulty breathing or cannot "
                "breathe normally, seek urgent medical attention."
            ),
        }

        return responses.get(
            intent,
            FALLBACK_MESSAGE,
        )

    # -----------------------------------------------------------------
    # Public prediction API
    # -----------------------------------------------------------------

    def predict(self, user_text: str):

        text = (
            user_text or ""
        ).strip()

        # -------------------------------------------------------------
        # Empty message
        # -------------------------------------------------------------

        if not text:

            return {
                "response": FALLBACK_MESSAGE,
                "intent": "fallback",
                "confidence": 0.0,
                "source": "fallback",
                "model_type": self.model_type,
            }

        # -------------------------------------------------------------
        # Normalize conversational input
        #
        # This allows inputs such as:
        #   "How are you?"
        #   "How are you!!!"
        #   "Can you help?"
        #
        # to match the same rule.
        # -------------------------------------------------------------

        text_clean = re.sub(
            r"\s+",
            " ",
            text.lower().strip()
        )

        text_clean = re.sub(
            r"[.!?,]+$",
            "",
            text_clean
        )

        # -------------------------------------------------------------
        # Common conversational phrases
        # -------------------------------------------------------------

        if text_clean in {
            "how are you",
            "how are u",
            "how are you doing",
            "how are u doing",
            "how have you been",
            "how is it going",
            "how's it going",
            "how are things",
            "how are things going",
            "you doing okay",
            "are you doing okay",
        }:

            return {
                "response": self._get_response("greeting"),
                "intent": "greeting",
                "confidence": 1.0,
                "source": "rule",
                "model_type": self.model_type,
            }

        # -------------------------------------------------------------
        # Generic help requests
        # -------------------------------------------------------------

        if text_clean in {
            "help me",
            "i need help",
            "can you help",
            "can someone help me",
            "please help me",
            "i need some help",
            "i need assistance",
            "can you assist me",
        }:

            return {
                "response": self._get_response("general_help"),
                "intent": "general_help",
                "confidence": 1.0,
                "source": "rule",
                "model_type": self.model_type,
            }

        # -------------------------------------------------------------
        # Emergency safety layer
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
        # Model prediction
        # -------------------------------------------------------------

        if self.model_type == "tfidf":

            result = self._tfidf_predict(
                text
            )

        else:

            result = self._semantic_predict(
                text
            )

        intent = result["intent"]

        confidence = float(
            result["confidence"]
        )

        # -------------------------------------------------------------
        # Fallback
        # -------------------------------------------------------------

        if intent == "fallback":

            source = (
                "tfidf_fallback"
                if self.model_type == "tfidf"
                else "semantic_fallback"
            )

            return {
                "response": FALLBACK_MESSAGE,
                "intent": "fallback",
                "confidence": confidence,
                "source": source,
                "model_type": self.model_type,
            }

        # -------------------------------------------------------------
        # Generate response
        # -------------------------------------------------------------

        response = self._get_response(
            intent
        )

        return {
            "response": response,
            "intent": intent,
            "confidence": confidence,
            "source": (
                "tfidf_model"
                if self.model_type == "tfidf"
                else "semantic_model"
            ),
            "model_type": self.model_type,
        }
