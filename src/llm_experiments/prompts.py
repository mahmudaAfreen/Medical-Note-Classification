"""Prompt construction and response parsing for LLM label experiments."""

from __future__ import annotations

import json
import re
from typing import Any

from src.llm_experiments.data import LABEL_ALIASES, canonical_label, normalize_whitespace


LABEL_DESCRIPTIONS = {
    "Acute Assessment": (
        "A conclusive diagnosis or primary assessment for the current complaint."
    ),
    "Acute Symptoms": (
        "Current symptoms, chief complaint, or symptom-oriented presentation."
    ),
    "Diagnostic Testing": (
        "Ordering tests such as labs, imaging, measurements, or other diagnostics."
    ),
    "Discussion": (
        "Counseling, shared decision making, patient agreement, or questions."
    ),
    "Drug History": "Alcohol, tobacco, cannabis, caffeine, or other substance use.",
    "Family History": "Medical history or symptoms in family members.",
    "Follow-up": "Recommended follow-up timing, monitoring, or appointments.",
    "Lab Examination": "Review or interpretation of laboratory values or panels.",
    "Medication": "Medication prescriptions, changes, continuations, or dosing.",
    "Other Socials": "Social context such as job, family, living situation, or support.",
    "Other Treatments": (
        "Non-medication treatments such as therapy, surgery, devices, diet, or rest."
    ),
    "Personal History": "Past medical history, chronic conditions, or prior events.",
    "Personal Information": "Patient demographics or identifying background facts.",
    "Physical Examination": "Vital signs or clinician physical exam findings.",
    "Radiology Examination": "Review or interpretation of imaging results.",
    "Reassessment": "Assessment of an existing or chronic condition.",
    "Referral": "Referral to a specialist, service, or therapy provider.",
    "Therapeutic History": "Past or current medication/treatment use by the patient.",
    "Vegetative History": "Body functions or review of systems such as sleep or appetite.",
}


SYSTEM_PROMPT = (
    "You are a medical NLP classifier.\n"
    "Choose exactly one label from the provided allowed labels.\n"
    "The label value must exactly match one allowed label string.\n"
    "Do not invent new labels.\n"
    "Do not copy words from the sentence as the label.\n"
    "Do not output SOAP section names such as Subjective, Objective, Assessment, or Plan.\n"
    "If none of the labels is perfect, choose the closest allowed label.\n"
    "Return ONLY a JSON object.\n"
    'Format: {"label":"LABEL_NAME"}\n'
    "Do not explain your answer.\n"
    "Do not repeat the sentence.\n"
    "Do not output markdown.\n"
    "Do not output code fences.\n"
)


def description_for_label(label: str) -> str:
    canonical = LABEL_ALIASES.get(label, label)
    return LABEL_DESCRIPTIONS.get(canonical, "Medical note intent label.")


def labels_block(labels: list[str]) -> str:
    return "\n".join(
        f"- {label}: {description_for_label(label)}" for label in labels
    )


def build_user_prompt(example: dict[str, Any], labels: list[str]) -> str:
    section = normalize_whitespace(example.get("section")) or "UNKNOWN"
    text = normalize_whitespace(example.get("text"))
    return (
        "Classify this medical-note sentence.\n\n"
        f"SOAP section: {section}\n"
        f"Sentence: {text}\n\n"
        "Allowed labels:\n"
        f"{labels_block(labels)}\n\n"
        "The JSON label must be copied exactly from the allowed labels above.\n"
        'Return JSON only in this exact shape: {"label": "<one allowed label>"}'
    )


def build_messages(
    example: dict[str, Any],
    labels: list[str],
    few_shot_examples: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for shot in few_shot_examples or []:
        messages.append({"role": "user", "content": build_user_prompt(shot, labels)})
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps({"label": shot["label"]}, ensure_ascii=False),
            }
        )
    messages.append({"role": "user", "content": build_user_prompt(example, labels)})
    return messages


def fallback_chat_template(messages: list[dict[str, str]]) -> str:
    chunks = []
    for message in messages:
        role = message["role"].upper()
        chunks.append(f"{role}:\n{message['content']}")
    chunks.append("ASSISTANT:\n")
    return "\n\n".join(chunks)


def messages_to_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template) and getattr(tokenizer, "chat_template", None):
        return apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return fallback_chat_template(messages)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    while start != -1:
        depth = 0
        for index in range(start, len(text)):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : index + 1]
                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError:
                        break
                    if isinstance(parsed, dict):
                        return parsed
        start = text.find("{", start + 1)
    return None


def _normalize_for_matching(value: str) -> str:
    value = canonical_label(value, canonicalize=False)
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return normalize_whitespace(value)


def _allowed_label_variants(labels: list[str]) -> dict[str, str]:
    variants: dict[str, str] = {}
    for allowed_label in labels:
        raw_variants = {allowed_label}
        if allowed_label in LABEL_ALIASES:
            raw_variants.add(LABEL_ALIASES[allowed_label])
        for raw_label, canonical in LABEL_ALIASES.items():
            if allowed_label == canonical:
                raw_variants.add(raw_label)
        for variant in raw_variants:
            variants[_normalize_for_matching(variant)] = allowed_label
    return variants


def parse_label_from_response(
    response: str,
    labels: list[str],
) -> tuple[str | None, str]:
    """Return ``(label, parse_status)`` from a raw model response."""
    allowed_by_norm = _allowed_label_variants(labels)

    parsed = _extract_json_object(response)
    if parsed is not None:
        raw_label = parsed.get("label")
        if isinstance(raw_label, str):
            normalized = _normalize_for_matching(raw_label)
            if normalized in allowed_by_norm:
                return allowed_by_norm[normalized], "json"

    normalized_response = _normalize_for_matching(response)
    matches = []
    for normalized_label, label in allowed_by_norm.items():
        if re.search(rf"\b{re.escape(normalized_label)}\b", normalized_response):
            matches.append(label)
    unique_matches = sorted(set(matches), key=len, reverse=True)
    if len(unique_matches) == 1:
        return unique_matches[0], "text_match"
    if len(unique_matches) > 1:
        return unique_matches[0], "ambiguous_text_match"
    return None, "invalid"
