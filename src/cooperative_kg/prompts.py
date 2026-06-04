"""LLM prompt templates for cooperative KG agents."""

from __future__ import annotations

import json

from .constants import HUMAN_STYLE_GUIDE


def build_agent1_prompt(transcript: str) -> str:
    return f"""
You are Agent 1: a high-recall clinical entity extractor.

Extract all candidate clinical entities from the transcript. Be inclusive, but
do not create absent/negated findings here; Agent 3 handles those.

{HUMAN_STYLE_GUIDE}

Node types:
- SYMPTOM
- DIAGNOSIS
- TREATMENT
- PROCEDURE
- LOCATION
- MEDICAL_HISTORY
- LAB_RESULT

Transcript:
{transcript}

Output ONLY a valid JSON array. Each item:
{{"id":"C_001","text":"short canonical phrase","type":"SYMPTOM","evidence":"tight quote","turn_id":"P-1"}}
""".strip()


def build_agent2_prompt(candidates: list[dict], transcript: str) -> str:
    return f"""
You are Agent 2: a clinical precision filter.

Keep candidates that a human curator would put into the KG. Remove:
- entities that are only doctor screening questions with no patient affirmation
- filler, vague duration-only items, and duplicate paraphrases
- over-specific phrases when a shorter clinical entity is already present
- unsupported entities not grounded by the evidence

Keep:
- active symptoms, suspected/active diagnoses, chronic disease, medications,
  tests/procedures, clinically relevant exposures, social/family history
- patient-concern diagnoses explicitly discussed, such as covid-19

{HUMAN_STYLE_GUIDE}

Transcript:
{transcript}

Candidates:
{json.dumps(candidates, indent=2, ensure_ascii=False)}

Output ONLY JSON:
{{"keep_ids":["C_001"],"drop_ids":["C_002"]}}
""".strip()


def build_agent3_prompt(transcript: str) -> str:
    return f"""
You are Agent 3: negation / absent finding extractor.

Extract clinically salient denied findings from this transcript. The human KG
does NOT include every normal ROS denial. Only emit absent findings that matter
for the chief complaint or differential diagnosis.

Use:
- SYMPTOM text: "absent X" (absent fever, absent chest pain)
- MEDICAL_HISTORY text: "no X" or "non-smoker" when relevant

Skip generic denials that are unrelated to the visit.

{HUMAN_STYLE_GUIDE}

Transcript:
{transcript}

Output ONLY a valid JSON array:
[{{"id":"NEG_001","text":"absent fever","type":"SYMPTOM","evidence":"No.","turn_id":"P-7"}}]
""".strip()


def build_agent4_prompt(nodes: list[dict], transcript: str) -> str:
    inventory = "\n".join(f'{n["id"]}: [{n["type"]}] "{n["text"]}"' for n in nodes)
    return f"""
You are Agent 4: clinical relation extractor.

Build edges using ONLY the node IDs in the inventory.

Allowed relations:
- CAUSES: risk factor/exposure/history causes or contributes to diagnosis/symptom
- INDICATES: symptom/procedure/history suggests diagnosis
- LOCATED_AT: symptom/diagnosis/procedure at anatomical location
- RULES_OUT: test/procedure or absent finding rules out condition
- TAKEN_FOR: treatment for diagnosis, symptom, or chronic medical history
- CONFIRMS: result/procedure confirms finding or diagnosis

Human relation style:
- Most INDICATES edges are SYMPTOM -> DIAGNOSIS.
- Most TAKEN_FOR edges are TREATMENT -> MEDICAL_HISTORY/DIAGNOSIS/SYMPTOM.
- Most LOCATED_AT edges are SYMPTOM -> LOCATION.
- A covid swab usually RULES_OUT covid-19 when ordered for possible covid.
- Self-isolation, tylenol, inhalers, insulin, steroids, antibiotics, etc. should
  connect to the condition or symptom they are used for.

Inventory:
{inventory}

Transcript:
{transcript}

Output ONLY a valid JSON array:
[{{"source_id":"N_001","target_id":"N_005","type":"INDICATES","evidence":"tight quote","turn_id":"D-39"}}]
""".strip()


def build_agent5_prompt(kg: dict) -> str:
    nodes = kg.get("nodes", [])
    payload = [{"id": n["id"], "text": n["text"], "type": n["type"]} for n in nodes]
    return f"""
You are Agent 5: clinical canonicalization agent.

Rewrite each node text to the human-curated style. Preserve id and type exactly.
Do not add or remove items. Use lowercase short clinical phrases except where
the source already contains a meaningful product/proper form.

{HUMAN_STYLE_GUIDE}

Rules:
- "stuffy nose" -> "nasal congestion"
- "covid" -> "covid-19"
- "type one diabetes" -> "type 1 diabetes"
- "short of breath" -> "shortness of breath"
- Preserve "absent X", "no X", and "non-smoker" prefixes.
- Keep specific phrases such as dry cough, productive cough, tylenol cold,
  covid swab, chest x-ray, family history of asthma.

Input:
{json.dumps(payload, indent=2, ensure_ascii=False)}

Output ONLY a valid JSON array, same ids:
[{{"id":"N_001","text":"canonical text"}}]
""".strip()
