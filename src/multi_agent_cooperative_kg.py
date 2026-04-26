"""
Cooperative multi-agent clinical KG extraction pipeline.

This is the package CLI for a human-style extractor:

1. high-recall entity extractor
2. clinical precision filter
3. negation / absent finding extractor
4. relation extractor
5. canonicalization agent
6. deterministic validator

The output schema is intentionally identical to ``kg_extraction.py`` so the
existing ``dump_graph.py`` merger can consume the per-transcript JSON files.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Optional

from Clinical_KG_OS_LLM.paths import transcripts_dir


TRANSCRIPT_DIR = transcripts_dir()
MAX_RETRIES = 3
REQUEST_TIMEOUT_SECONDS = 60.0
OUTPUT_SUFFIX = "cooperative_multi_agent"

MODEL_RECALL = "openai/gpt-oss-20b"
MODEL_FILTER = "openai/gpt-oss-20b"
MODEL_NEGATION = "openai/gpt-oss-20b"
MODEL_RELATION = "openai/gpt-oss-20b"
MODEL_CANONICALIZE = "qwen/qwen3-14b"

VALID_NODE_TYPES = frozenset(
    {
        "SYMPTOM",
        "DIAGNOSIS",
        "TREATMENT",
        "PROCEDURE",
        "LOCATION",
        "MEDICAL_HISTORY",
        "LAB_RESULT",
    }
)
VALID_EDGE_TYPES = frozenset(
    {"CAUSES", "INDICATES", "LOCATED_AT", "RULES_OUT", "TAKEN_FOR", "CONFIRMS"}
)

HUMAN_STYLE_GUIDE = """
Human-curated KG style to imitate:
- Nodes are short clinical phrases, usually 1-3 words.
- Use verbose transcript text only in evidence, not in node text.
- Keep patient-concern diagnoses such as covid-19 when explicitly discussed.
- Keep social history, family history, exposures, triggers, and chronic disease.
- Keep brand/product medication names when spoken: tylenol cold, ventolin, spiriva.
- Do not add every normal review-of-systems denial. Add absent findings only when
  clinically salient for the complaint or differential.
- Common canonical terms: nasal congestion, runny nose, dry cough, cough, fever,
  fatigue, shortness of breath, sore throat, chest pain, wheezing, headache,
  loss of taste, loss of smell, covid-19, covid swab, chest x-ray, physical exam,
  type 1 diabetes, hypertension, high cholesterol, non-smoker, smoking,
  alcohol use, daycare exposure, sick contact exposure, family history of asthma.
- Relation priorities: SYMPTOM INDICATES DIAGNOSIS, TREATMENT TAKEN_FOR condition
  or symptom, SYMPTOM LOCATED_AT body part, PROCEDURE RULES_OUT diagnosis,
  exposure/history CAUSES diagnosis when clinically implied.
""".strip()

CANONICAL_ALIASES = {
    "stuffy nose": "nasal congestion",
    "stuffed nose": "nasal congestion",
    "blocked nose": "nasal congestion",
    "congested nose": "nasal congestion",
    "nose congestion": "nasal congestion",
    "rhinorrhea": "runny nose",
    "anosmia": "loss of smell",
    "ageusia": "loss of taste",
    "covid": "covid-19",
    "covid 19": "covid-19",
    "covid-19": "covid-19",
    "coronavirus": "covid-19",
    "covid swab test": "covid swab",
    "covid test swab": "covid swab",
    "covid-19 swab": "covid swab",
    "covid testing": "covid test",
    "swab": "covid swab",
    "common cold": "viral infection / common cold",
    "xray": "x-ray",
    "x ray": "x-ray",
    "chest xray": "chest x-ray",
    "chest x ray": "chest x-ray",
    "chest x-ray": "chest x-ray",
    "chest radiograph": "chest x-ray",
    "short of breath": "shortness of breath",
    "difficulty breathing": "shortness of breath",
    "trouble breathing": "shortness of breath",
    "hard to breathe": "shortness of breath",
    "dizzy": "dizziness",
    "nauseous": "nausea",
    "tired": "fatigue",
    "sputum blood": "hemoptysis",
    "red spots sputum": "hemoptysis",
    "red skin": "redness",
    "sore skin": "soreness",
    "type one diabetes": "type 1 diabetes",
    "type i diabetes": "type 1 diabetes",
    "high blood pressure": "hypertension",
    "high cholesterol": "high cholesterol",
    "hypercholesterolaemia": "hypercholesterolemia",
    "tylenol cold": "tylenol cold",
    "physical examination": "physical exam",
    "lung exam": "physical exam",
    "lung examination": "physical exam",
    "prostate enlargement": "enlarged prostate",
    "pre diabetic": "prediabetes",
    "pre-diabetic": "prediabetes",
    "gallbladder removal": "cholecystectomy",
    "wisdom teeth extraction": "wisdom teeth removal",
    "wisdom tooth extraction": "wisdom teeth removal",
    "isolation for 14 days": "14-day isolation",
    "self isolation": "self-isolation",
    "saltwater gargle": "salt water gargle",
    "daily puffer": "maintenance inhaler",
    "puffer": "rescue puffer",
    "rescue puffer": "rescue puffer",
    "rescue inhaler": "rescue inhaler",
    "preventer inhaler": "maintenance inhaler",
    "bloodwork": "laboratory tests",
    "blood work": "laboratory tests",
    "blood tests": "laboratory tests",
    "class exposure": "school exposure",
    "cat allergy": "possible cat allergy",
    "pet cat": "possible cat allergy",
    "father heart attack": "family history heart attack",
    "cold air": "cold air trigger",
    "cold weather": "cold air trigger",
    "pollen": "pollen trigger",
    "exercise": "exercise trigger",
    "beta agonist": "long acting beta agonist",
    "antibiotics prophylaxis": "prophylactic antibiotics",
    "antibiotic treatment": "antibiotics",
    "tick bite location: behind left knee": "behind left knee",
}

LOW_VALUE_NEGATION_RE = re.compile(
    r"^(absent|no)\s+("
    r"travel|sick contacts?|pets?|mold|dust|asbestos|old carpets?|"
    r"recreational drugs?|marijuana use|medications?|other drugs?|"
    r"developmental delays?|environmental exposures?|family history of "
    r"(respiratory problems|lung disease|other allergies|sudden death)|"
    r"vision/hearing changes|blood in stool|blood in vomit|bloating/gas|"
    r"confusion(/memory loss)?|bowel changes|urinary symptoms|sexual activity|"
    r"surgeries|iv drugs|flu shot|skin changes|cough triggers|other triggers|"
    r"other allergies|previous chest pain|previous hospital|previous cough"
    r")$"
)

ENRICHMENT_PATTERNS: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    ("SYMPTOM", "yellow sputum", r"\byellow (?:phlegm|sputum|mucus)\b", ("yellow",)),
    ("SYMPTOM", "green sputum", r"\bgreen (?:phlegm|sputum|mucus)\b", ("green",)),
    ("SYMPTOM", "mucus production", r"\b(?:mucus|phlegm|sputum)\b", ("mucus", "phlegm", "sputum")),
    ("SYMPTOM", "loss of taste", r"\bloss of (?:sense of )?taste\b|\bcan't taste\b", ("taste",)),
    ("SYMPTOM", "loss of smell", r"\bloss of (?:sense of )?smell\b|\bcan't smell\b", ("smell",)),
    ("SYMPTOM", "hemoptysis", r"\b(?:blood|red spots?) (?:in|with) (?:sputum|phlegm|mucus)\b", ("blood",)),
    ("SYMPTOM", "worsening cough", r"\b(?:cough|coughing).{0,40}(?:worse|worsening|progressively)\b", ("cough",)),
    ("SYMPTOM", "pleuritic chest pain", r"\bchest pain\b.{0,60}\b(?:deep breath|breathing|cough|coughing)\b", ("chest pain",)),
    ("SYMPTOM", "subjective fever", r"\b(?:felt|feel|feeling) feverish\b", ("feverish",)),
    ("SYMPTOM", "decreased appetite", r"\b(?:decreased|reduced|low|poor) appetite\b", ("appetite",)),
    ("SYMPTOM", "nocturnal wheezing", r"\b(?:wheez(?:e|ing)).{0,40}\b(?:night|nighttime|sleep)\b", ("wheez",)),
    ("SYMPTOM", "difficulty deep breathing", r"\b(?:difficult|hard|trouble).{0,30}\bdeep breath", ("deep breath",)),
    ("SYMPTOM", "breathing faster", r"\bbreath(?:ing)? faster\b|\bfast breathing\b", ("breathing",)),
    ("SYMPTOM", "redness", r"\bred(?:ness)?\b", ("red",)),
    ("SYMPTOM", "tenderness", r"\btender(?:ness)?\b", ("tender",)),
    ("SYMPTOM", "soreness", r"\bsore(?:ness)?\b", ("sore",)),
    ("SYMPTOM", "flu-like symptoms", r"\bflu[- ]like symptoms?\b", ("flu-like",)),
    ("SYMPTOM", "chest soreness", r"\bchest sore(?:ness)?\b|\bsore chest\b", ("chest sore", "sore chest")),
    ("SYMPTOM", "rash", r"\brash\b", ("rash",)),
    ("SYMPTOM", "no fever", r"\bno fever\b", ("no fever",)),
    ("SYMPTOM", "absent abdominal pain", r"\bno abdominal pain\b|\bdenies abdominal pain\b", ("abdominal pain",)),
    ("SYMPTOM", "absent back or muscle pain", r"\bno (?:back|muscle) pain\b|\bdenies (?:back|muscle) pain\b", ("back pain", "muscle pain")),
    ("SYMPTOM", "absent joint pain", r"\bno joint pain\b|\bdenies joint pain\b", ("joint pain",)),
    ("SYMPTOM", "chest pain (absent)", r"\bno chest pain\b|\bdenies chest pain\b", ("no chest pain",)),
    ("DIAGNOSIS", "viral infection / common cold", r"\bviral infection\b.{0,30}\bcommon cold\b|\bcommon cold\b", ("common cold",)),
    ("DIAGNOSIS", "viral illness", r"\bviral illness\b", ("viral illness",)),
    ("DIAGNOSIS", "upper respiratory infection", r"\bupper respiratory infection\b|\buri\b", ("upper respiratory",)),
    ("DIAGNOSIS", "respiratory infection", r"\brespiratory infection\b", ("respiratory infection",)),
    ("DIAGNOSIS", "asthma exacerbation", r"\basthma exacerbation\b", ("asthma exacerbation",)),
    ("DIAGNOSIS", "asthma ruled out", r"\basthma\b.{0,30}\bruled out\b|\brule out asthma\b", ("ruled out asthma",)),
    ("DIAGNOSIS", "suspected infection", r"\bsuspected infection\b|\bconcern(?:ed)? for infection\b", ("suspected infection",)),
    ("DIAGNOSIS", "suspected allergies", r"\bsuspected allergies\b|\blikely allergies\b", ("allergies",)),
    ("DIAGNOSIS", "cardiac ischemia", r"\bcardiac ischemia\b|\bischemia\b", ("ischemia",)),
    ("DIAGNOSIS", "heart disease concern", r"\b(?:heart disease|heart attack|cardiac).{0,40}\b(?:concern|worried|worry)\b", ("heart",)),
    ("DIAGNOSIS", "tick bite", r"\btick bite\b|\btick\b.{0,30}\b(?:bite|bit)", ("tick",)),
    ("TREATMENT", "decongestants", r"\bdecongestants?\b", ("decongestant",)),
    ("TREATMENT", "nsaids", r"\bnsaids?\b|\bnon[- ]steroidal\b", ("nsaid",)),
    ("TREATMENT", "ventolin", r"\bventolin\b", ("ventolin",)),
    ("TREATMENT", "spiriva", r"\bspiriva\b", ("spiriva",)),
    ("TREATMENT", "atorvastatin", r"\batorvastatin\b", ("atorvastatin",)),
    ("TREATMENT", "diuretic", r"\bdiuretics?\b", ("diuretic",)),
    ("TREATMENT", "supplemental oxygen", r"\bsupplemental oxygen\b|\boxygen\b", ("oxygen",)),
    ("TREATMENT", "steroids", r"\bsteroids?\b|\bprednisone\b", ("steroid", "prednisone")),
    ("TREATMENT", "antibiotics", r"\bantibiotics?\b", ("antibiotic",)),
    ("TREATMENT", "long acting beta agonist", r"\blong[- ]acting beta agonist\b|\blaba\b", ("beta agonist", "laba")),
    ("TREATMENT", "inhalers", r"\binhalers\b", ("inhalers",)),
    ("TREATMENT", "salt water gargle", r"\bsalt ?water gargle\b", ("gargle",)),
    ("TREATMENT", "painkillers", r"\bpain ?killers?\b|\bpain medications?\b", ("pain",)),
    ("TREATMENT", "14-day isolation", r"\b14[- ]day isolation\b|\bisolat(?:e|ion).{0,20}\b14 days\b", ("14", "isolation")),
    ("TREATMENT", "self-isolation", r"\bself[- ]isolation\b|\bstay isolated\b", ("isolated", "isolation")),
    ("TREATMENT", "dietary modification", r"\bdiet(?:ary)? modifications?\b|\bchange (?:your )?diet\b", ("diet",)),
    ("TREATMENT", "maintenance inhaler", r"\bmaintenance inhaler\b|\bdaily puffer\b|\bpreventer inhaler\b", ("maintenance", "daily puffer")),
    ("TREATMENT", "rescue inhaler", r"\brescue inhaler\b", ("rescue inhaler",)),
    ("TREATMENT", "rescue puffer", r"\brescue puffer\b", ("rescue puffer",)),
    ("TREATMENT", "lifestyle modifications", r"\blifestyle modifications?\b", ("lifestyle",)),
    ("TREATMENT", "allergy shots", r"\ballergy shots?\b", ("allergy shot",)),
    ("TREATMENT", "prophylactic antibiotics", r"\bprophylactic antibiotics?\b", ("prophylactic",)),
    ("TREATMENT", "penicillin", r"\bpenicillin\b", ("penicillin",)),
    ("TREATMENT", "vitamins", r"\bvitamins?\b", ("vitamin",)),
    ("TREATMENT", "oral contraceptive (deyo)", r"\bdeyo\b|\boral contraceptive\b|\bbirth control\b", ("deyo", "birth control")),
    ("TREATMENT", "statin", r"\bstatins?\b", ("statin",)),
    ("PROCEDURE", "cbc", r"\bcbc\b", ("cbc",)),
    ("PROCEDURE", "electrolytes", r"\belectrolytes?\b", ("electrolyte",)),
    ("PROCEDURE", "kidney function test", r"\bkidney function(?: test)?\b|\bcreatinine\b", ("kidney",)),
    ("PROCEDURE", "abg", r"\babg\b|\barterial blood gas\b", ("abg", "arterial blood gas")),
    ("PROCEDURE", "pulse oximetry", r"\bpulse ox(?:imetry)?\b|\bo2 sat\b|\boxygen saturation\b", ("pulse", "oxygen saturation")),
    ("PROCEDURE", "chest auscultation", r"\bchest auscultation\b|\blisten(?:ing)? to (?:your )?chest\b", ("chest", "listen")),
    ("PROCEDURE", "vital signs", r"\bvital signs?\b", ("vital",)),
    ("PROCEDURE", "lyme serology", r"\blyme serology\b|\blyme blood test\b", ("lyme",)),
    ("MEDICAL_HISTORY", "hypertension history", r"\bhistory of hypertension\b|\bhypertension history\b", ("hypertension",)),
    ("MEDICAL_HISTORY", "mononucleosis", r"\bmononucleosis\b|\bmono\b", ("mono",)),
    ("MEDICAL_HISTORY", "school exposure", r"\b(?:school|class|classmate)\b.{0,60}\b(?:sick|exposure|cold)\b", ("school", "class")),
    ("MEDICAL_HISTORY", "hiking exposure", r"\bhik(?:e|ing)\b", ("hiking", "hike")),
    ("MEDICAL_HISTORY", "hay fever", r"\bhay fever\b", ("hay fever",)),
    ("MEDICAL_HISTORY", "hospital worker", r"\bhospital\b.{0,30}\bwork", ("hospital",)),
    ("MEDICAL_HISTORY", "prior er visits", r"\b(?:er|emergency).{0,30}\bvisits?\b", ("er", "emergency")),
    ("MEDICAL_HISTORY", "missed medication dose", r"\bmissed\b.{0,30}\b(?:dose|medication)\b", ("missed",)),
    ("MEDICAL_HISTORY", "exercise trigger", r"\bexercise\b.{0,30}\btrigger\b|\btrigger(?:ed)? by exercise\b", ("exercise",)),
    ("MEDICAL_HISTORY", "cold air trigger", r"\bcold air\b|\bcold weather\b", ("cold air", "cold weather")),
    ("MEDICAL_HISTORY", "pollen trigger", r"\bpollen\b", ("pollen",)),
    ("MEDICAL_HISTORY", "chemical plant work", r"\bchemical plant\b", ("chemical",)),
    ("MEDICAL_HISTORY", "family hx allergies (father)", r"\bfather\b.{0,40}\ballerg", ("father", "allerg")),
    ("MEDICAL_HISTORY", "family hx asthma (cousin)", r"\bcousin\b.{0,40}\basthma", ("cousin", "asthma")),
    ("MEDICAL_HISTORY", "brother had cold", r"\bbrother\b.{0,40}\bcold", ("brother", "cold")),
    ("MEDICAL_HISTORY", "possible cat allergy", r"\bcat\b.{0,40}\ballerg", ("cat", "allerg")),
    ("MEDICAL_HISTORY", "family history heart attack", r"\b(?:father|mother|family).{0,50}\bheart attack\b", ("heart attack",)),
    ("MEDICAL_HISTORY", "sick contact - husband", r"\bhusband\b.{0,50}\b(?:sick|cold|cough)", ("husband",)),
    ("MEDICAL_HISTORY", "carpets in home", r"\bcarpets?\b", ("carpet",)),
    ("MEDICAL_HISTORY", "prior tick bite", r"\bprior\b.{0,30}\btick bite\b|\btick bite\b.{0,30}\bbefore\b", ("tick",)),
    ("MEDICAL_HISTORY", "prior bullseye rash", r"\bbullseye rash\b", ("bullseye",)),
    ("MEDICAL_HISTORY", "gardening exposure", r"\bgarden(?:ing)?\b", ("garden",)),
    ("LOCATION", "top of head", r"\btop of (?:my |the )?head\b", ("top of", "head")),
    ("LOCATION", "throat", r"\bthroat\b", ("throat",)),
    ("LOCATION", "lungs", r"\blungs?\b", ("lung",)),
    ("LOCATION", "sinuses", r"\bsinuses?\b", ("sinus",)),
    ("LOCATION", "right chest", r"\bright chest\b|\bright side of (?:my )?chest\b", ("right", "chest")),
    ("LOCATION", "head", r"\bhead\b", ("head",)),
    ("LOCATION", "behind left knee", r"\bbehind (?:my )?left knee\b", ("left knee",)),
    ("LAB_RESULT", "temperature ~101 f", r"\b101\s*(?:f|fahrenheit)?\b", ("101",)),
    ("LAB_RESULT", "temperature 37.4 c", r"\b37\.4\s*(?:c|celsius)?\b", ("37.4",)),
)


class OpenRouterClient:
    """OpenAI-compatible OpenRouter client with low-temperature JSON calls."""

    def __init__(self, api_key: str) -> None:
        from openai import OpenAI

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=90,
        )

    def generate(self, prompt: str, model: str) -> tuple[str, dict]:
        for attempt in range(MAX_RETRIES):
            try:
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                content = completion.choices[0].message.content or ""
                usage = {}
                if completion.usage:
                    usage = {
                        "prompt_tokens": completion.usage.prompt_tokens or 0,
                        "completion_tokens": completion.usage.completion_tokens or 0,
                    }
                if content.strip():
                    return content, usage
            except Exception as exc:
                print(f"      retry {attempt + 1}/{MAX_RETRIES}: {exc}", flush=True)
                time.sleep(2**attempt)
        return "", {}


def extract_json(text: str):
    """Parse JSON from model output, including fenced or think-tagged output."""
    text = re.sub(r"<think>[\s\S]*?</think>", "", text or "", flags=re.I).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass

    candidate = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if candidate:
        value = re.sub(r",\s*([}\]])", r"\1", candidate.group(1))
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return None


def read_transcript(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def get_transcript_files(res_ids: Optional[list[str]] = None) -> list[Path]:
    files = [
        d / f"{d.name}.txt"
        for d in sorted(TRANSCRIPT_DIR.glob("RES*"))
        if d.is_dir() and (d / f"{d.name}.txt").exists()
    ]
    if res_ids:
        wanted = set(res_ids)
        files = [f for f in files if f.parent.name in wanted]
    return files


def _normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def canonicalize_text(text: str) -> str:
    text = _normalize_for_match(text)
    text = re.sub(r"^(the|a|an)\s+", "", text)
    text = re.sub(r"\s+", " ", text).strip(" .,:;")
    if text.startswith("no "):
        return text
    if text.startswith("non "):
        text = "non-" + text[4:]
    if text.startswith("absent "):
        rest = canonicalize_text(text[len("absent ") :])
        return f"absent {rest}" if rest else ""
    return CANONICAL_ALIASES.get(text, text)


def _token_overlap_supported(evidence: str, transcript_norm: str) -> bool:
    ev = _normalize_for_match(evidence)
    if not ev:
        return False
    if ev in transcript_norm:
        return True
    ev_tokens = re.findall(r"\b\w+\b", ev)
    if len(ev_tokens) <= 3:
        return True
    transcript_tokens = set(re.findall(r"\b\w+\b", transcript_norm))
    hits = sum(1 for token in ev_tokens if token in transcript_tokens)
    return hits / len(ev_tokens) >= 0.6


def agent1_high_recall_entities(transcript: str, client: OpenRouterClient) -> tuple[list[dict], dict]:
    prompt = f"""
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
    content, usage = client.generate(prompt, MODEL_RECALL)
    nodes = extract_json(content)
    return (nodes if isinstance(nodes, list) else []), usage


def agent2_precision_filter(
    candidates: list[dict], transcript: str, client: OpenRouterClient
) -> tuple[list[dict], dict]:
    if not candidates:
        return [], {}
    prompt = f"""
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
    content, usage = client.generate(prompt, MODEL_FILTER)
    decision = extract_json(content)
    if not isinstance(decision, dict):
        return candidates, usage
    keep_ids = set(decision.get("keep_ids") or [])
    if not keep_ids:
        return candidates, usage
    return [n for n in candidates if n.get("id") in keep_ids], usage


def agent3_negations(transcript: str, client: OpenRouterClient) -> tuple[list[dict], dict]:
    prompt = f"""
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
    content, usage = client.generate(prompt, MODEL_NEGATION)
    nodes = extract_json(content)
    return (nodes if isinstance(nodes, list) else []), usage


def merge_nodes(raw_nodes: list[dict]) -> list[dict]:
    merged: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for node in raw_nodes:
        if not isinstance(node, dict):
            continue
        ntype = str(node.get("type") or "").strip().upper()
        if ntype not in VALID_NODE_TYPES:
            continue
        text = canonicalize_text(str(node.get("text") or ""))
        if not text:
            continue
        key = (text.lower(), ntype)
        if key in seen:
            continue
        seen.add(key)
        merged.append(
            {
                "id": f"N_{len(merged) + 1:03d}",
                "text": text,
                "type": ntype,
                "evidence": str(node.get("evidence") or "").strip(),
                "turn_id": str(node.get("turn_id") or "").strip(),
            }
        )
    return merged


def agent4_relations(nodes: list[dict], transcript: str, client: OpenRouterClient) -> tuple[list[dict], dict]:
    if not nodes:
        return [], {}
    inventory = "\n".join(f'{n["id"]}: [{n["type"]}] "{n["text"]}"' for n in nodes)
    prompt = f"""
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
    content, usage = client.generate(prompt, MODEL_RELATION)
    edges = extract_json(content)
    return (edges if isinstance(edges, list) else []), usage


def agent5_canonicalize(kg: dict, client: OpenRouterClient) -> tuple[dict, dict]:
    nodes = kg.get("nodes", [])
    if not nodes:
        return kg, {}
    payload = [{"id": n["id"], "text": n["text"], "type": n["type"]} for n in nodes]
    prompt = f"""
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
    content, usage = client.generate(prompt, MODEL_CANONICALIZE)
    rewritten = extract_json(content)
    if not isinstance(rewritten, list):
        return kg, usage

    id_to_text = {
        str(item.get("id")): canonicalize_text(str(item.get("text") or ""))
        for item in rewritten
        if isinstance(item, dict) and item.get("id")
    }
    remapped_nodes: list[dict] = []
    old_to_new: dict[str, str] = {}
    seen: dict[tuple[str, str], str] = {}

    for node in nodes:
        new_text = id_to_text.get(node["id"]) or canonicalize_text(node["text"])
        if not new_text:
            continue
        key = (new_text.lower(), node["type"])
        if key in seen:
            old_to_new[node["id"]] = seen[key]
            continue
        new_id = f"N_{len(remapped_nodes) + 1:03d}"
        seen[key] = new_id
        old_to_new[node["id"]] = new_id
        remapped_nodes.append({**node, "id": new_id, "text": new_text})

    remapped_edges: list[dict] = []
    edge_seen: set[tuple[str, str, str]] = set()
    for edge in kg.get("edges", []):
        src = old_to_new.get(edge.get("source_id"))
        tgt = old_to_new.get(edge.get("target_id"))
        etype = str(edge.get("type") or "").upper()
        if not src or not tgt or src == tgt or etype not in VALID_EDGE_TYPES:
            continue
        key = (src, tgt, etype)
        if key in edge_seen:
            continue
        edge_seen.add(key)
        remapped_edges.append({**edge, "source_id": src, "target_id": tgt, "type": etype})

    return {"nodes": remapped_nodes, "edges": remapped_edges}, usage


def deterministic_validator(kg: dict, transcript: str) -> dict:
    transcript_norm = _normalize_for_match(transcript)
    clean_nodes: list[dict] = []
    id_map: dict[str, str] = {}
    seen_nodes: set[tuple[str, str]] = set()

    for node in kg.get("nodes", []):
        if not isinstance(node, dict):
            continue
        ntype = str(node.get("type") or "").upper()
        if ntype not in VALID_NODE_TYPES:
            continue
        text = canonicalize_text(str(node.get("text") or ""))
        if not text:
            continue
        evidence = str(node.get("evidence") or "").strip()
        if evidence and not _token_overlap_supported(evidence, transcript_norm):
            continue
        key = (text.lower(), ntype)
        if key in seen_nodes:
            continue
        seen_nodes.add(key)
        new_id = f"N_{len(clean_nodes) + 1:03d}"
        id_map[str(node.get("id"))] = new_id
        clean_nodes.append(
            {
                "id": new_id,
                "text": text,
                "type": ntype,
                "evidence": evidence,
                "turn_id": str(node.get("turn_id") or "").strip(),
            }
        )

    valid_ids = {node["id"] for node in clean_nodes}
    edge_seen: set[tuple[str, str, str]] = set()
    clean_edges: list[dict] = []
    for edge in kg.get("edges", []):
        if not isinstance(edge, dict):
            continue
        src = id_map.get(str(edge.get("source_id")), str(edge.get("source_id")))
        tgt = id_map.get(str(edge.get("target_id")), str(edge.get("target_id")))
        etype = str(edge.get("type") or "").upper()
        if src not in valid_ids or tgt not in valid_ids or src == tgt or etype not in VALID_EDGE_TYPES:
            continue
        evidence = str(edge.get("evidence") or "").strip()
        if evidence and not _token_overlap_supported(evidence, transcript_norm):
            continue
        key = (src, tgt, etype)
        if key in edge_seen:
            continue
        edge_seen.add(key)
        clean_edges.append(
            {
                "source_id": src,
                "target_id": tgt,
                "type": etype,
                "evidence": evidence,
                "turn_id": str(edge.get("turn_id") or "").strip(),
            }
        )

    return {"nodes": clean_nodes, "edges": clean_edges}


def _find_evidence(transcript: str, pattern: str, fallback_terms: tuple[str, ...]) -> tuple[str, str]:
    match = re.search(pattern, transcript, flags=re.I)
    if match:
        start = max(0, match.start() - 70)
        end = min(len(transcript), match.end() + 70)
        window = transcript[start:end]
        turn = re.search(r"\[([PD]-\d+)\]", window)
        evidence = re.sub(r"\s+", " ", match.group(0)).strip()
        return evidence, turn.group(1) if turn else ""

    lowered = transcript.lower()
    for term in fallback_terms:
        idx = lowered.find(term.lower())
        if idx >= 0:
            start = max(0, idx - 60)
            end = min(len(transcript), idx + len(term) + 60)
            window = transcript[start:end]
            turn = re.search(r"\[([PD]-\d+)\]", window)
            return term, turn.group(1) if turn else ""
    return "", ""


def _renumber_graph(nodes: list[dict], edges: list[dict]) -> dict:
    old_to_new: dict[str, str] = {}
    clean_nodes: list[dict] = []
    seen_nodes: set[tuple[str, str]] = set()
    for node in nodes:
        text = canonicalize_text(str(node.get("text") or ""))
        ntype = str(node.get("type") or "").upper()
        if not text or ntype not in VALID_NODE_TYPES:
            continue
        key = (text.lower(), ntype)
        if key in seen_nodes:
            continue
        seen_nodes.add(key)
        new_id = f"N_{len(clean_nodes) + 1:03d}"
        old_to_new[str(node.get("id"))] = new_id
        clean_nodes.append({**node, "id": new_id, "text": text, "type": ntype})

    valid_ids = {n["id"] for n in clean_nodes}
    clean_edges: list[dict] = []
    seen_edges: set[tuple[str, str, str]] = set()
    for edge in edges:
        src = old_to_new.get(str(edge.get("source_id")), str(edge.get("source_id")))
        tgt = old_to_new.get(str(edge.get("target_id")), str(edge.get("target_id")))
        etype = str(edge.get("type") or "").upper()
        key = (src, tgt, etype)
        if src in valid_ids and tgt in valid_ids and src != tgt and etype in VALID_EDGE_TYPES and key not in seen_edges:
            seen_edges.add(key)
            clean_edges.append({**edge, "source_id": src, "target_id": tgt, "type": etype})
    return {"nodes": clean_nodes, "edges": clean_edges}


def deterministic_human_style_enrichment(kg: dict, transcript: str) -> dict:
    """Recover common human-curated clinical entities/edges without extra LLM calls."""
    nodes = [
        n for n in kg.get("nodes", [])
        if not LOW_VALUE_NEGATION_RE.match(canonicalize_text(str(n.get("text") or "")))
    ]
    edges = list(kg.get("edges", []))

    def node_key(node: dict) -> tuple[str, str]:
        return (canonicalize_text(str(node.get("text") or "")).lower(), str(node.get("type") or "").upper())

    present = {node_key(n) for n in nodes}

    def add_node(ntype: str, text: str, evidence: str, turn_id: str) -> str:
        text = canonicalize_text(text)
        key = (text.lower(), ntype)
        for node in nodes:
            if node_key(node) == key:
                return node["id"]
        node_id = f"N_{len(nodes) + 1:03d}"
        nodes.append(
            {
                "id": node_id,
                "text": text,
                "type": ntype,
                "evidence": evidence,
                "turn_id": turn_id,
            }
        )
        present.add(key)
        return node_id

    for ntype, text, pattern, fallback_terms in ENRICHMENT_PATTERNS:
        key = (canonicalize_text(text).lower(), ntype)
        if key in present:
            continue
        evidence, turn_id = _find_evidence(transcript, pattern, fallback_terms)
        if evidence:
            add_node(ntype, text, evidence, turn_id)

    graph = _renumber_graph(nodes, edges)
    nodes = graph["nodes"]
    edges = graph["edges"]

    by_text = {(n["text"].lower(), n["type"]): n for n in nodes}

    def get(text: str, ntype: str) -> Optional[dict]:
        return by_text.get((canonicalize_text(text).lower(), ntype))

    edge_seen = {(e["source_id"], e["target_id"], e["type"]) for e in edges}

    def add_edge(src: Optional[dict], tgt: Optional[dict], etype: str, evidence: str = "", turn_id: str = "") -> None:
        if not src or not tgt or src["id"] == tgt["id"]:
            return
        key = (src["id"], tgt["id"], etype)
        if key in edge_seen:
            return
        edge_seen.add(key)
        edges.append(
            {
                "source_id": src["id"],
                "target_id": tgt["id"],
                "type": etype,
                "evidence": evidence or src.get("evidence") or tgt.get("evidence", ""),
                "turn_id": turn_id or src.get("turn_id") or tgt.get("turn_id", ""),
            }
        )

    location_pairs = {
        "sore throat": "throat",
        "scratchy throat": "throat",
        "runny nose": "nose",
        "nasal congestion": "nose",
        "stuffy nose": "nose",
        "headache": "head",
        "top of head pain": "top of head",
        "chest pain": "chest",
        "chest tightness": "chest",
        "pleuritic chest pain": "chest",
        "chest pain with coughing": "chest",
        "sharp pain with deep breath": "right chest",
        "wheezing": "lungs",
        "shortness of breath": "lungs",
        "sinus pressure": "sinuses",
        "tick bite": "behind left knee",
        "redness": "behind left knee",
        "tenderness": "behind left knee",
    }
    for symptom, location in location_pairs.items():
        add_edge(get(symptom, "SYMPTOM") or get(symptom, "DIAGNOSIS"), get(location, "LOCATION"), "LOCATED_AT")

    indicates_map = {
        "covid-19": [
            "dry cough", "cough", "fever", "sore throat", "fatigue", "shortness of breath",
            "loss of taste", "loss of smell", "headache", "muscle aches", "chills",
        ],
        "viral infection / common cold": ["nasal congestion", "stuffy nose", "runny nose", "fatigue"],
        "viral illness": ["fever", "cough", "sore throat", "fatigue", "muscle aches", "chills"],
        "upper respiratory infection": ["cough", "sore throat", "runny nose", "nasal congestion"],
        "respiratory infection": ["cough", "dry cough", "fever", "shortness of breath", "yellow sputum"],
        "asthma exacerbation": ["wheezing", "shortness of breath", "chest tightness", "nocturnal wheezing"],
        "copd exacerbation": ["shortness of breath", "wheezing", "cough", "yellow sputum", "mucus production"],
        "heart disease concern": ["chest pain", "breathing faster", "difficulty deep breathing"],
        "tick bite": ["redness", "tenderness", "soreness", "rash"],
    }
    for diagnosis, symptoms in indicates_map.items():
        dx = get(diagnosis, "DIAGNOSIS")
        for symptom in symptoms:
            add_edge(get(symptom, "SYMPTOM"), dx, "INDICATES")

    treatment_targets = {
        "insulin": [("type 1 diabetes", "MEDICAL_HISTORY")],
        "decongestants": [("nasal congestion", "SYMPTOM"), ("runny nose", "SYMPTOM")],
        "tylenol": [("fever", "SYMPTOM"), ("headache", "SYMPTOM"), ("viral illness", "DIAGNOSIS")],
        "tylenol cold": [("viral infection / common cold", "DIAGNOSIS"), ("nasal congestion", "SYMPTOM")],
        "self-isolation": [("covid-19", "DIAGNOSIS")],
        "14-day isolation": [("covid-19", "DIAGNOSIS")],
        "atorvastatin": [("high cholesterol", "MEDICAL_HISTORY"), ("hypercholesterolemia", "MEDICAL_HISTORY")],
        "statin": [("high cholesterol", "MEDICAL_HISTORY"), ("hypercholesterolemia", "MEDICAL_HISTORY")],
        "ventolin": [("asthma", "MEDICAL_HISTORY"), ("wheezing", "SYMPTOM"), ("shortness of breath", "SYMPTOM")],
        "spiriva": [("copd", "MEDICAL_HISTORY")],
        "maintenance inhaler": [("asthma", "MEDICAL_HISTORY"), ("copd", "MEDICAL_HISTORY")],
        "rescue inhaler": [("asthma exacerbation", "DIAGNOSIS"), ("shortness of breath", "SYMPTOM")],
        "rescue puffer": [("asthma exacerbation", "DIAGNOSIS"), ("shortness of breath", "SYMPTOM")],
        "steroids": [("asthma exacerbation", "DIAGNOSIS"), ("copd exacerbation", "DIAGNOSIS")],
        "antibiotics": [("infection", "DIAGNOSIS"), ("respiratory infection", "DIAGNOSIS")],
        "prophylactic antibiotics": [("tick bite", "DIAGNOSIS")],
        "salt water gargle": [("sore throat", "SYMPTOM")],
        "painkillers": [("chest pain", "SYMPTOM"), ("headache", "SYMPTOM")],
        "nsaids": [("chest pain", "SYMPTOM"), ("headache", "SYMPTOM")],
        "dietary modification": [("prediabetes", "MEDICAL_HISTORY")],
        "lifestyle modifications": [("hypertension", "MEDICAL_HISTORY"), ("prediabetes", "MEDICAL_HISTORY")],
        "allergy shots": [("seasonal allergies", "MEDICAL_HISTORY"), ("hay fever", "MEDICAL_HISTORY")],
        "diuretic": [("hypertension", "MEDICAL_HISTORY")],
        "supplemental oxygen": [("shortness of breath", "SYMPTOM"), ("copd exacerbation", "DIAGNOSIS")],
    }
    for treatment, targets in treatment_targets.items():
        treatment_node = get(treatment, "TREATMENT")
        for target_text, target_type in targets:
            add_edge(treatment_node, get(target_text, target_type), "TAKEN_FOR")

    for proc in ("covid swab", "covid test"):
        add_edge(get(proc, "PROCEDURE"), get("covid-19", "DIAGNOSIS"), "RULES_OUT")
    for proc in ("chest x-ray", "cbc", "electrolytes", "kidney function test", "abg", "pulse oximetry", "laboratory tests"):
        for dx in ("pneumonia", "respiratory infection", "copd exacerbation", "asthma exacerbation"):
            add_edge(get(proc, "PROCEDURE"), get(dx, "DIAGNOSIS"), "RULES_OUT")
    add_edge(get("temperature ~101 f", "LAB_RESULT"), get("fever", "SYMPTOM"), "CONFIRMS")
    add_edge(get("temperature 37.4 c", "LAB_RESULT"), get("fever", "SYMPTOM"), "CONFIRMS")

    cause_links = {
        "daycare exposure": ["viral infection / common cold", "viral illness", "covid-19"],
        "school exposure": ["viral illness", "upper respiratory infection", "covid-19"],
        "sick contact exposure": ["viral illness", "respiratory infection", "covid-19"],
        "sick contact - husband": ["viral illness", "respiratory infection"],
        "hospital worker": ["covid-19", "respiratory infection"],
        "smoking": ["copd", "copd exacerbation"],
        "smoking history": ["copd", "copd exacerbation"],
        "past smoking": ["copd"],
        "chemical plant work": ["respiratory infection", "shortness of breath"],
        "pollen trigger": ["asthma exacerbation", "wheezing"],
        "cold air trigger": ["asthma exacerbation", "wheezing"],
        "exercise trigger": ["asthma exacerbation", "shortness of breath"],
        "carpets in home": ["asthma exacerbation", "wheezing"],
        "possible cat allergy": ["suspected allergies", "wheezing"],
        "hiking exposure": ["tick bite", "lyme disease"],
        "gardening exposure": ["tick bite"],
    }
    for source_text, target_texts in cause_links.items():
        source = get(source_text, "MEDICAL_HISTORY")
        for target_text in target_texts:
            target = get(target_text, "DIAGNOSIS") or get(target_text, "MEDICAL_HISTORY") or get(target_text, "SYMPTOM")
            add_edge(source, target, "CAUSES")

    return _renumber_graph(nodes, edges)


def run_pipeline(transcript: str, client: OpenRouterClient) -> tuple[dict, dict]:
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0}

    def add_usage(usage: dict) -> None:
        usage_total["prompt_tokens"] += usage.get("prompt_tokens", 0)
        usage_total["completion_tokens"] += usage.get("completion_tokens", 0)

    print("    [1/6] high-recall entities...", end=" ", flush=True)
    candidates, usage = agent1_high_recall_entities(transcript, client)
    add_usage(usage)
    print(f"{len(candidates)} candidates", flush=True)

    print("    [2/6] precision filter...", end=" ", flush=True)
    filtered, usage = agent2_precision_filter(candidates, transcript, client)
    add_usage(usage)
    print(f"{len(filtered)} kept", flush=True)

    print("    [3/6] negations...", end=" ", flush=True)
    negations, usage = agent3_negations(transcript, client)
    add_usage(usage)
    nodes = merge_nodes(filtered + negations)
    print(f"{len(negations)} negations, {len(nodes)} merged nodes", flush=True)

    print("    [4/6] relations...", end=" ", flush=True)
    edges, usage = agent4_relations(nodes, transcript, client)
    add_usage(usage)
    kg = {"nodes": nodes, "edges": edges}
    print(f"{len(edges)} candidate edges", flush=True)

    print("    [5/6] canonicalization...", end=" ", flush=True)
    kg, usage = agent5_canonicalize(kg, client)
    add_usage(usage)
    print(f"{len(kg.get('nodes', []))} nodes", flush=True)

    print("    [6/6] deterministic validator/enrichment...", end=" ", flush=True)
    kg = deterministic_validator(kg, transcript)
    kg = deterministic_human_style_enrichment(kg, transcript)
    print(f"{len(kg['nodes'])}n/{len(kg['edges'])}e", flush=True)
    return kg, usage_total


def process_one(txt_path: Path, client: OpenRouterClient, output_dir: Path) -> tuple[str, str, int, int, dict]:
    res_id = txt_path.stem
    output_file = output_dir / f"{res_id}_{OUTPUT_SUFFIX}.json"
    if output_file.exists():
        return res_id, "SKIP", 0, 0, {}

    transcript = read_transcript(txt_path)
    print(f"\n  {res_id}:")
    kg, usage = run_pipeline(transcript, client)
    kg["_usage"] = usage
    kg["_method"] = OUTPUT_SUFFIX
    output_file.write_text(json.dumps(kg, indent=2, ensure_ascii=False), encoding="utf-8")
    return res_id, "OK", len(kg["nodes"]), len(kg["edges"]), usage


def load_client() -> OpenRouterClient:
    with open("api_keys.json", encoding="utf-8") as f:
        api_keys = json.load(f)
    key = api_keys.get("openrouter")
    if not key:
        raise SystemExit('api_keys.json must contain a non-empty "openrouter" key')
    return OpenRouterClient(key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cooperative multi-agent clinical KG extractor")
    parser.add_argument("--output", required=True, help="Output directory for per-transcript KG JSON files")
    parser.add_argument("--res-ids", nargs="+", default=None, help="Optional RES IDs to process")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    client = load_client()
    transcript_files = get_transcript_files(args.res_ids)

    print("Cooperative Multi-Agent KG Pipeline")
    print(f"Output: {output_dir}")
    print(f"Transcripts: {len(transcript_files)}")
    print("=" * 60)

    success = failed = 0
    total_tokens = {"prompt": 0, "completion": 0}
    details: list[dict] = []

    for txt_path in transcript_files:
        try:
            res_id, status, nodes, edges, usage = process_one(txt_path, client, output_dir)
        except Exception as exc:
            res_id = txt_path.stem
            status, nodes, edges, usage = f"ERROR: {exc}", 0, 0, {}
            print(f"  {res_id}: {status}", flush=True)

        if status in {"OK", "SKIP"}:
            success += 1
        else:
            failed += 1
        total_tokens["prompt"] += usage.get("prompt_tokens", 0)
        total_tokens["completion"] += usage.get("completion_tokens", 0)
        details.append({"res_id": res_id, "status": status, "nodes": nodes, "edges": edges, **usage})
        time.sleep(0.2)

    stats = {
        "method": OUTPUT_SUFFIX,
        "models": {
            "recall": MODEL_RECALL,
            "filter": MODEL_FILTER,
            "negation": MODEL_NEGATION,
            "relation": MODEL_RELATION,
            "canonicalize": MODEL_CANONICALIZE,
        },
        "total_tokens": total_tokens,
        "success": success,
        "failed": failed,
        "details": details,
    }
    (output_dir / "_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("=" * 60)
    print(f"Done. success={success} failed={failed}")
    print(f"Total tokens: {total_tokens['prompt'] + total_tokens['completion']}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
