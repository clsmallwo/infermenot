from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from mcq_data import ALL_LETTERS
from passage_inference import DEFAULT_PASSAGE_MODEL, LongformerRaceMCQ
from training_lookup import TrainingLookup
from transformer_mcq import UnifiedQAMCQPiper


app = FastAPI(title="MCQ Inference UI")
training_lookup = TrainingLookup()


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    kind: str
    model_name: str
    estimated_size: str
    description: str


MODEL_SPECS = {
    "longformer-race": ModelSpec(
        id="longformer-race",
        label="Longformer RACE",
        kind="longformer",
        model_name=DEFAULT_PASSAGE_MODEL,
        estimated_size="3.2 GB",
        description="Passage-aware multiple-choice model.",
    ),
    "unifiedqa-t5-large": ModelSpec(
        id="unifiedqa-t5-large",
        label="UnifiedQA T5 Large",
        kind="unifiedqa",
        model_name="allenai/unifiedqa-t5-large",
        estimated_size="3.0 GB",
        description="Large QA model scored by answer likelihood.",
    ),
}
DEFAULT_MODEL_ID = "longformer-race"
MODEL_CACHE = {}


class PredictRequest(BaseModel):
    question: str = Field(min_length=1)
    choices: Dict[str, str]
    model_id: str = DEFAULT_MODEL_ID


class MagicRequest(BaseModel):
    question: str = Field(min_length=1)
    model_id: str = DEFAULT_MODEL_ID


MAGIC_EIGHT_BALL_CHOICES = {
    "A": "It is certain.",
    "B": "Outlook good.",
    "C": "Signs point to yes.",
    "D": "Ask again later.",
    "E": "Cannot predict now.",
    "F": "Reply hazy.",
    "G": "My sources say no.",
    "H": "Very doubtful.",
}


QUESTION_WORDS = {
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "when",
    "where",
    "why",
    "how",
    "is",
    "are",
    "was",
    "were",
    "do",
    "does",
    "did",
    "can",
    "could",
    "should",
    "would",
    "will",
}


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    seen_inodes = set()
    total = 0
    for file in path.rglob("*"):
        if not file.is_file():
            continue
        stat = file.stat()
        inode = (stat.st_dev, stat.st_ino)
        if inode in seen_inodes:
            continue
        seen_inodes.add(inode)
        total += stat.st_size
    return total


def format_size(bytes_count: int) -> str:
    if not bytes_count:
        return ""
    size = float(bytes_count)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} TB"


def local_cache_path(model_name: str) -> Path:
    return Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_name.replace('/', '--')}"


def model_size(spec: ModelSpec) -> str:
    return spec.estimated_size


def model_options() -> list[dict]:
    return [
        {
            "id": spec.id,
            "label": spec.label,
            "kind": spec.kind,
            "model_name": spec.model_name,
            "size": model_size(spec),
            "description": spec.description,
            "default": spec.id == DEFAULT_MODEL_ID,
        }
        for spec in MODEL_SPECS.values()
    ]


def get_model(model_id: str):
    spec = MODEL_SPECS.get(model_id)
    if not spec:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_id}")
    if model_id not in MODEL_CACHE:
        if spec.kind == "longformer":
            MODEL_CACHE[model_id] = LongformerRaceMCQ(spec.model_name)
        elif spec.kind == "unifiedqa":
            MODEL_CACHE[model_id] = UnifiedQAMCQPiper(spec.model_name)
        else:
            raise HTTPException(status_code=500, detail=f"Unsupported model kind: {spec.kind}")
    return spec, MODEL_CACHE[model_id]


def softmax(scores: Dict[str, float]) -> Dict[str, float]:
    maximum = max(scores.values())
    exp_scores = {letter: math.exp(score - maximum) for letter, score in scores.items()}
    total = sum(exp_scores.values())
    return {letter: value / total for letter, value in exp_scores.items()}


def predict_with_model(model_id: str, question: str, choices: Dict[str, str]):
    spec, model = get_model(model_id)
    result = model.predict(question, choices)
    scores = getattr(result, "scores", None)
    probabilities = getattr(result, "probabilities", None)
    if probabilities is None:
        probabilities = softmax(scores)
    if scores is None:
        scores = probabilities
    return spec, result.answer, probabilities, scores, str(model.device)


def confidence_summary(probabilities: Dict[str, float]) -> dict:
    ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    if not ordered:
        return {"level": "none", "margin": 0.0, "top_probability": 0.0}
    top_probability = ordered[0][1]
    runner_up = ordered[1][1] if len(ordered) > 1 else 0.0
    margin = top_probability - runner_up
    if margin >= 0.35 and top_probability >= 0.6:
        level = "high"
    elif margin >= 0.15 and top_probability >= 0.45:
        level = "medium"
    else:
        level = "low"
    return {"level": level, "margin": margin, "top_probability": top_probability}


def input_quality(question: str, choices: Dict[str, str], probabilities: Dict[str, float]) -> dict:
    words = [word.strip(".,!?;:()[]{}\"'").lower() for word in question.split()]
    first_word = words[0] if words else ""
    issues = []
    if len(words) < 3:
        issues.append("The prompt is very short; the model may be ranking associations rather than answering a question.")
    if "?" not in question and first_word not in QUESTION_WORDS:
        issues.append("The prompt does not look like a question.")
    if len(choices) < 3:
        issues.append("Only two choices were provided; add a neutral or more specific option when the question is subjective.")
    if any(len(choice.split()) <= 1 for choice in choices.values()) and len(words) < 5:
        issues.append("Short labels with a short prompt produce weak MCQ signals.")
    confidence = confidence_summary(probabilities)
    if confidence["level"] == "low":
        issues.append("The top choices are close together, so treat this as a weak preference.")
    status = "needs_clarification" if issues else "ok"
    return {"status": status, "issues": issues, "confidence": confidence}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return Path("web/index.html").read_text(encoding="utf-8")


@app.get("/api/status")
def status():
    spec = MODEL_SPECS[DEFAULT_MODEL_ID]
    cached_model = MODEL_CACHE.get(DEFAULT_MODEL_ID)
    device = str(cached_model.device) if cached_model else "not loaded"
    return {
        "model": spec.model_name,
        "model_id": spec.id,
        "model_label": spec.label,
        "model_size": model_size(spec),
        "device": device,
        "uses_metal": device == "mps",
        "score_mode": spec.kind,
        "models": model_options(),
    }


@app.get("/api/models")
def models():
    return {"default_model_id": DEFAULT_MODEL_ID, "models": model_options()}


@app.post("/api/predict")
def predict(payload: PredictRequest):
    choices = {
        letter: payload.choices.get(letter, "").strip()
        for letter in ALL_LETTERS
        if payload.choices.get(letter, "").strip()
    }
    if len(choices) < 2:
        raise HTTPException(status_code=400, detail="Enter at least two answer choices.")
    try:
        spec, answer, probabilities, scores, device = predict_with_model(
            payload.model_id,
            payload.question.strip(),
            choices,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        MODEL_CACHE.pop(payload.model_id, None)
        raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}") from exc
    return {
        "answer": answer,
        "probabilities": probabilities,
        "scores": scores,
        "model_id": spec.id,
        "model_label": spec.label,
        "model_size": model_size(spec),
        "device": device,
        "uses_metal": device == "mps",
        "score_mode": spec.kind,
        "quality": input_quality(payload.question.strip(), choices, probabilities),
        "training_match": training_lookup.find(payload.question.strip()),
    }


@app.post("/api/magic-eight-ball")
def magic_eight_ball(payload: MagicRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Enter a question.")
    try:
        spec, answer, probabilities, scores, device = predict_with_model(
            payload.model_id,
            question,
            MAGIC_EIGHT_BALL_CHOICES,
        )
    except Exception as exc:
        MODEL_CACHE.pop(payload.model_id, None)
        raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}") from exc
    return {
        "answer": answer,
        "message": MAGIC_EIGHT_BALL_CHOICES[answer],
        "probabilities": probabilities,
        "scores": scores,
        "choices": MAGIC_EIGHT_BALL_CHOICES,
        "model_id": spec.id,
        "model_label": spec.label,
        "model_size": model_size(spec),
        "device": device,
        "uses_metal": device == "mps",
        "score_mode": f"magic-eight-ball-{spec.kind}",
        "quality": {
            "status": "ok",
            "issues": [],
            "confidence": confidence_summary(probabilities),
        },
        "training_match": training_lookup.find(question),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("webui:app", host="127.0.0.1", port=8000, reload=False)
