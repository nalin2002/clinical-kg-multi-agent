"""
smoke_test_glm47_specialized_parallel_260420.py
================================================

Specialized-per-category parallel extraction framework.
Model: z-ai/glm-4.7-flash via OpenRouter.

Architecture (this is the long-term plan; only SYMPTOM is active for this
iteration):
  Stage 1 (parallel, ThreadPoolExecutor): all ACTIVE entity-category
          extractors fire together. Each is an EXPERT internist / pharmacist /
          anatomist / etc. with its own XML-delimited rule-based prompt.
  Stage 2 (parallel, ThreadPoolExecutor): all ACTIVE edge-category extractors
          fire together, taking stage-1 nodes as input so src/dst are known
          entity IDs.

This iteration's ACTIVE set:
  entity:  SYMPTOM
  edge:    (none yet; added iteratively)

Prompts:
  - Every prompt is XML-delimited: <role> / <purpose> / <rules> /
    <curated_labels> / <transcript> / <output_contract>.
  - Role declares EXPERT clinical role + "analyzing this transcript excerpt
    for the hospital".
  - Rules enforce: no inference/assumption, verbatim curated labels, no
    splitting combined labels, assertion gate for denials, evidence field
    per item (turn_id + verbatim quote), "when in doubt emit nothing".
  - Output contract emits objects with `{label|text, turn_id, quote}`.

Output:
  eir_results/smoke_test_glm47_specialized_parallel/
    RES0XXX/
      RES0XXX_SYMPTOM.json         (per-category result)
      ... (future categories will land here too)
      RES0XXX_all.json             (combined roll-up)

Run (one transcript, SYMPTOM only):
    python smoke_test_glm47_specialized_parallel_260420.py --res-id RES0200

Created: 260420
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import time
from pathlib import Path


EIR_ROOT = Path(__file__).parent
PROJECT_ROOT = EIR_ROOT.parent
TRANSCRIPT_DIR = PROJECT_ROOT / "data" / "transcripts"
CURATED_KB = EIR_ROOT / "curated_kb_260419.json"
API_KEYS_PATH = PROJECT_ROOT / "api_keys.json"
OUT_DIR = EIR_ROOT / "eir_results" / "smoke_test_glm47_specialized_parallel_v7"

OPENROUTER_MODEL = "z-ai/glm-4.7-flash"

# Per-category overrides. Four independent axes — mix and match per category.
# A category not listed in a given dict falls back to the default for that
# axis (OPENROUTER_MODEL / stream=True / default temperature / no reasoning).
# All 6 edge categories now run on the reasoning model with V12-style prompts.
_EDGE_CATS = {"INDICATES", "CAUSES", "CONFIRMS",
              "RULES_OUT", "TAKEN_FOR", "LOCATED_AT"}
MODEL_OVERRIDES_BY_CATEGORY = {
    c: "deepseek/deepseek-r1-distill-qwen-32b" for c in _EDGE_CATS
}
NONSTREAM_CATEGORIES = set(_EDGE_CATS)
TEMPERATURE_OVERRIDES_BY_CATEGORY = {c: 0.6 for c in _EDGE_CATS}
REASONING_TRACE_CATEGORIES = set(_EDGE_CATS)
MAX_RETRIES = 3

ENTITY_CATEGORIES = ["SYMPTOM", "DIAGNOSIS", "TREATMENT", "PROCEDURE",
                     "LOCATION", "MEDICAL_HISTORY", "LAB_RESULT"]
EDGE_CATEGORIES   = ["RULES_OUT", "LOCATED_AT", "TAKEN_FOR",
                     "INDICATES", "CAUSES", "CONFIRMS"]

# Toggle per category for iterative build-out.
ACTIVE: dict[str, bool] = {
    # entities
    "SYMPTOM":         True,
    "DIAGNOSIS":       False,
    "TREATMENT":       False,
    "PROCEDURE":       False,
    "LOCATION":        False,
    "MEDICAL_HISTORY": False,
    "LAB_RESULT":      False,
    # edges
    "RULES_OUT":       False,
    "LOCATED_AT":      False,
    "TAKEN_FOR":       False,
    "INDICATES":       False,
    "CAUSES":          False,
    "CONFIRMS":        False,
}


# ─────────────────────────────────────────────────────────────────────────────
# Prompts — only SYMPTOM is implemented for this iteration.
# Others are intentionally None — the code skips them unless ACTIVE[cat]=True,
# and if ACTIVE[cat]=True but prompt is None we fail loudly.
# ─────────────────────────────────────────────────────────────────────────────

SYMPTOM_PROMPT = """<role>
You are an EXPERT clinical internist documenting the patient's symptoms
for the medical chart. Your job is to extract every SYMPTOM the patient
reports, the patient denies when asked, or the clinician directly observes.
You are rigorous and conservative; clinical decision support depends on
your output being deterministic and grounded. When in doubt, emit nothing.
</role>

<purpose>
For each mention in the transcript, walk through these decisions to decide
whether and how to emit it as a SYMPTOM item:

1. IS THIS A SYMPTOM OR CURRENT-VISIT DIAGNOSIS?
   - Something the patient FEELS or EXPERIENCES (cough, pain, fatigue,
     nausea, shortness of breath) → symptom (EMIT).
   - Something the clinician DIAGNOSES as a disease (pneumonia, COPD
     exacerbation, COVID-19) → SKIP (goes to DIAGNOSIS category).
   - Observed signs the clinician names (wheezing heard on exam, rash
     on skin) → symptom (EMIT).

2. IS THE PATIENT AFFIRMING OR DENYING?
   - Affirming ("I have a cough", "my chest hurts") → emit the positive
     form.
   - Denying in response to a clinician question ("Any headaches?" /
     "No") → check <curated_labels> for the exact denial form to emit.
     Do NOT emit the positive form when the patient denies.
   - Denying without being asked → check <curated_labels> for the exact
     denial form; otherwise skip.

3. WHICH CURATED VARIANT DO I USE?
   Multiple curated forms may exist for similar content:
   - `fever`, `subjective fever`
   - `cough`, `dry cough`, `productive cough`, `worsening cough`
   - `chest pain`, `pleuritic chest pain`, `chest tightness`,
     `chest soreness`
   Pick the curated variant that best fits BOTH the transcript content
   AND the clinician's framing if the clinician restates the symptom.
   Example: if the patient says "I felt hot" but the clinician later
   says "your fevers", prefer `fever` over `subjective fever` because
   the clinician's framing is authoritative.

   PRIORITY RULE (applies to AFFIRMATIVE labels only — does NOT apply
   to denials): If the transcript phrase appears VERBATIM in
   <curated_labels>, emit THAT exact form — do NOT "upgrade" to a
   clinical synonym. Examples:
   - patient says "stuffy nose" AND `stuffy nose` is in curated →
     emit `stuffy nose`, NOT `nasal congestion`.
   - patient says "fever" AND `fever` is in curated → emit `fever`,
     NOT `subjective fever` (unless the patient's own framing clearly
     requires the subjective form, e.g. "felt hot but never took my
     temperature").

4. PRESERVE QUALIFIERS — do NOT strip:
   - `dry cough` is NOT the same as `cough` — emit `dry cough` if
     transcript says "dry".
   - `productive cough` is NOT the same as `cough` — emit the curated
     variant that fits.
   - `worsening cough` keeps "worsening"; `pleuritic chest pain` keeps
     "pleuritic"; `nocturnal wheezing` keeps "nocturnal".

   COLOR PRIORITY: When the patient describes multiple colors in one
   phrase (e.g. "green yellowy sputum"), pick the FIRST color mentioned.
   So "green yellowy" → `green sputum`, not `yellow sputum`.

5. SYNONYM MAPPING — patient layman terms → curated clinical terms.
   If the transcript describes a curated symptom using a lay synonym,
   copy the CURATED LABEL verbatim to matched. Examples:
   - "coughing blood", "spitting blood" → `hemoptysis`
   - "winded when I walk", "get short of breath climbing stairs" →
     `dyspnea on exertion`
   - "hurts when I breathe deep" → `pleuritic chest pain`
   - "can't smell anything" → `loss of smell`
   - "can't taste my food" → `loss of taste`
   - "sore all over", "overall achy", "body aches" → `muscle ache`
   - "feel crappy", "feel run down" → `malaise`
   - "face hurts", "sinuses feel full" → `sinus pressure`
   - "short of breath", "SOB", "breathless" → `shortness of breath`
   - "chest hurts", "chest pressure" → `chest pain` (or `chest tightness`
     if the transcript says "tight")
   - "not hungry", "not eating much" → `decreased appetite`

6. COMPREHENSIVE REVIEW.
   Before finalizing, walk through EVERY label in <curated_labels> and
   check whether the transcript supports it. Do not stop at the first
   few symptoms you notice. Patients often mention symptoms in passing
   ("and I couldn't really taste my breakfast") — those count.

7. COMPOUND SYMPTOMS — preserve the full phrase.
   Curated may contain multi-word symptom labels like:
   - `chest pain with coughing`, `sharp pain with deep breath`
   - `difficulty deep breathing`, `difficulty swallowing`
   If the transcript supports such a compound, emit the compound label
   verbatim — do NOT break it up.

   ONE QUOTE, MULTIPLE LABELS: A single transcript quote can support
   more than one curated label. Example: patient says "I'm bringing up
   yellow stuff when I cough." If both `productive cough` AND
   `yellow sputum` are in <curated_labels>, emit BOTH — one for the
   coughing aspect, one for the color aspect. Do not stop after
   emitting one match per quote.

Every emitted item MUST cite turn_id + verbatim quote from the transcript.
</purpose>

<rules>
1. "matched" — use curated labels from the list below. If the transcript
   describes a curated symptom using a synonym, layman phrasing, or
   abbreviation (e.g. "SOB" for `shortness of breath`, "coughing blood"
   for `hemoptysis`, "overall achy" for `muscle ache`, "can't taste" for
   `loss of taste`), copy the CURATED LABEL verbatim to matched. Cite
   the transcript phrasing in the quote. For denials, emit whichever
   denial form appears in <curated_labels> for that concept verbatim.
2. "other" — SYMPTOM entities in the transcript that are NOT in the
   <curated_labels> list. Use the actual words from the transcript.
3. "fine_grained" — SYMPTOM entities in the transcript that are finer
   grained than entries in the <curated_labels> list (e.g. "productive
   cough with yellow sputum" when curated has "yellow sputum"). Use the
   actual words from the transcript.
4. Do NOT assume or infer symptoms. Extract ONLY symptoms explicitly
   mentioned in the transcript.
</rules>

<curated_labels>
{curated_labels_bulleted}
</curated_labels>

<transcript>
{transcript}
</transcript>

<output_contract>
Emit ONE json object and nothing else:
{{
  "matched": [
    {{"label": "muscle ache", "turn_id": "P-20", "quote": "I feel kind of overall achy"}}
  ],
  "other": [
    {{"text": "laborious breathing", "turn_id": "P-3", "quote": "laborious"}}
  ],
  "fine_grained": [
    {{"text": "productive cough with yellow sputum", "turn_id": "P-6", "quote": "coughing up yellow stuff"}}
  ]
}}
</output_contract>"""

# ─────────────────────────────────────────────────────────────────────────────
# Entity prompts — each category gets a specialized EXPERT role + same 4-rule
# framework as SYMPTOM. fine_grained = two words, sequentially closest to an
# existing curated label.
# ─────────────────────────────────────────────────────────────────────────────

def _entity_prompt(category: str, expert_role: str, purpose: str,
                    pos_example: tuple, other_example: tuple, fine_example: tuple) -> str:
    # pos_example = (label, turn_id, quote); same shape for other/fine
    pos_l, pos_t, pos_q = pos_example
    oth_l, oth_t, oth_q = other_example
    fine_l, fine_t, fine_q = fine_example
    return f"""<role>
You are an EXPERT {expert_role} analyzing this doctor-patient transcript
excerpt for the hospital. Your sole responsibility is to extract {category}
entities explicitly stated in the transcript. You are rigorous and
conservative; clinical decision support depends on your output being
deterministic and grounded. When in doubt, emit nothing.
</role>

<purpose>
{purpose}
</purpose>

<rules>
1. "matched" — curated labels copied verbatim from the list below.
2. "other" — NO MORE THAN TWO WORDS, {category} entities in the transcript
   that are NOT in the <curated_labels> list below. Use the actual words
   from the transcript.
3. "fine_grained" — NO MORE THAN TWO WORDS, {category} entities in the
   transcript that are finer grained than entries in the <curated_labels>
   list below. Use the actual words from the transcript.
4. Do NOT assume or infer. Extract ONLY {category} entities explicitly
   mentioned in the transcript.
</rules>

<curated_labels>
{{curated_labels_bulleted}}
</curated_labels>

<transcript>
{{transcript}}
</transcript>

<output_contract>
Emit ONE json object and nothing else:
{{{{
  "matched": [
    {{{{"label": "{pos_l}", "turn_id": "{pos_t}", "quote": "{pos_q}"}}}}
  ],
  "other": [
    {{{{"text": "{oth_l}", "turn_id": "{oth_t}", "quote": "{oth_q}"}}}}
  ],
  "fine_grained": [
    {{{{"text": "{fine_l}", "turn_id": "{fine_t}", "quote": "{fine_q}"}}}}
  ]
}}}}
</output_contract>"""


DIAGNOSIS_PROMPT = """<role>
You are an EXPERT clinical internist analyzing this doctor-patient transcript
excerpt for the hospital. Your sole responsibility is to extract every
DIAGNOSIS the clinician is considering at THIS visit — confirmed, suspected,
possible, or explicitly ruled-out. You are rigorous and conservative;
clinical decision support depends on your output being deterministic and
grounded. When in doubt, emit nothing.
</role>

<purpose>
Extract every DIAGNOSIS the clinician considers in THIS visit. Include
confirmed diagnoses, suspected / possible / likely working hypotheses,
explicit rule-outs, and exacerbations of chronic conditions. Skip the
patient's pre-existing chronic conditions (those go to MEDICAL_HISTORY).
Skip symptoms and complaints (those go to SYMPTOM).

Every emitted item MUST cite turn_id + verbatim quote from the transcript.
</purpose>

<rules>
1. "matched" — use curated labels from the list below. If the transcript
   describes a curated diagnosis using a synonym, abbreviation, colloquial
   form, or a qualifier phrase that equals a curated label (e.g. "sounds
   like an infection" for `suspected infection`, "he doesn't have asthma"
   for `asthma ruled out`, "concerned for heart disease" for `heart
   disease concern`, "covid" for `covid-19`, "flu" for `influenza`),
   copy the CURATED LABEL verbatim to matched. Cite the transcript
   phrasing in the quote.
2. "other" — DIAGNOSIS entities in the transcript that are NOT in the
   <curated_labels> list. Use the actual words from the transcript.
3. "fine_grained" — DIAGNOSIS entities in the transcript that are finer
   grained than entries in the <curated_labels> list. Use the actual
   words from the transcript.
4. Do NOT assume or infer. Extract ONLY diagnoses the clinician
   EXPLICITLY considers in the transcript.
5. EXHAUSTIVE SCAN — before finalizing, walk through EVERY label in
   <curated_labels> and ask: is the clinician ACTIVELY WORKING on this
   diagnosis for THIS patient — asserting it, suspecting it, hedging
   on it, listing it on a differential ("add X to the list", "on the
   differential", "top of the differential"), stating it flatly,
   ruling it out after work-up, or acknowledging a patient-voiced
   concern?

   If yes, emit in matched with the curated label verbatim.

   Do NOT emit when the mention is:
     - a procedure / test mention ("covid swab", "chest X-ray" are
       PROCEDUREs; only extract the diagnosis if the doctor is actively
       working it up, not just because a test was ordered)
     - preventive talk ("prevent any infections")
     - screening negative WITHOUT work-up ("do you have asthma?" + "no"
       + no further clinical reasoning — that's not "ruled out")
     - patient's distant history ("had pneumonia as a child" →
       MEDICAL_HISTORY)
     - sick contact mention ("my husband had covid" → exposure, not
       THIS patient's diagnosis)
     - general health discussion unrelated to this patient's problem

6. PREFER QUALIFIED CURATED LABELS WHEN CONTEXT SUPPORTS THEM.
   When the doctor uses a base term (e.g. "infection", "asthma") but
   the patient's presentation makes the qualifier clear, prefer the
   qualified curated label:
     - Respiratory presentation (cough, sputum, shortness of breath,
       chest involvement) + "infection" → `respiratory infection`
     - URI presentation (runny nose, sore throat, nasal congestion)
       + "infection" → `upper respiratory infection`
     - Chronic condition with acute worsening
       (e.g. "asthma reoccurring", "COPD flare") → `X exacerbation`
   If the transcript provides no disambiguating context, use the base
   curated label.
</rules>

<curated_labels>
{curated_labels_bulleted}
</curated_labels>

<transcript>
{transcript}
</transcript>

<output_contract>
Emit ONE json object and nothing else:
{{
  "matched": [
    {{"label": "suspected infection", "turn_id": "D-45", "quote": "sounds like an infection"}}
  ],
  "other": [
    {{"text": "pneumonia", "turn_id": "D-22", "quote": "this looks like pneumonia"}}
  ],
  "fine_grained": [
    {{"text": "community acquired pneumonia", "turn_id": "D-22", "quote": "community acquired pneumonia"}}
  ]
}}
</output_contract>"""

TREATMENT_PROMPT = """<role>
You are an EXPERT clinical pharmacist analyzing this doctor-patient transcript
excerpt for the hospital. Your sole responsibility is to extract every
MEDICATION and TREATMENT explicitly mentioned in the transcript. Pay
particular attention to pharmaceuticals and medical devices, which are easy
to miss when transcript spelling is non-standard.

MEDICATIONS and TREATMENTs include:
  - medications (pharmaceuticals)
  - injections
  - inhalers
  - medical devices (CPAP, pacemakers, glucometers, infusion pumps, etc.)
  - vaccines
  - oxygen
  - behavioral instructions

EXCLUDE diagnostic procedures (chest X-ray, blood work, lab tests, swabs,
ABG, CBC, electrolytes panels) — those belong to the separate PROCEDURE
category, NOT to TREATMENT.

You are rigorous and conservative; clinical decision support depends on your
output being deterministic and grounded. When in doubt, emit nothing.
</role>

<purpose>
Extract every MEDICATION and TREATMENT explicitly mentioned in the transcript.
Pay specific attention to specific medications such as atorvastatin,
ventolin, Generic treatment names such as supplemental oxygen or diuretic.
Every emitted item MUST cite turn_id + verbatim quote from the transcript.
</purpose>

<rules>
1. "matched" — use curated labels from the list below. If the transcript
   names a curated treatment even with mangled spelling (e.g. "Torva Staten"
   for atorvastatin, "Ventilin" for ventolin) or a colloquial name (e.g.
   "water pill" for diuretic, "puffer" for an inhaler), copy the CURATED
   LABEL verbatim to matched. Cite the transcript phrasing in the quote.
2. "other" — TREATMENT entities in the transcript that are NOT in the
   <curated_labels> list. Use the actual words from the transcript.
3. "fine_grained" — TREATMENT entities in the transcript that are finer
   grained than entries in the <curated_labels> list (e.g. specific dose,
   route, or formulation). Use the actual words from the transcript.
4. Do NOT assume or infer. Extract ONLY TREATMENTs explicitly mentioned
   in the transcript.
</rules>

<curated_labels>
{curated_labels_bulleted}
</curated_labels>

<transcript>
{transcript}
</transcript>

<output_contract>
Emit ONE json object and nothing else:
{{
  "matched": [
    {{"label": "atorvastatin", "turn_id": "P-38", "quote": "Just a Torva Staten"}}
  ],
  "other": [
    {{"text": "fluids", "turn_id": "D-9", "quote": "drink plenty of fluids"}}
  ],
  "fine_grained": [
    {{"text": "inhaled albuterol", "turn_id": "D-18", "quote": "your albuterol inhaler"}}
  ]
}}
</output_contract>"""

PROCEDURE_PROMPT = _entity_prompt(
    category="PROCEDURE",
    expert_role="clinician",
    purpose=(
        "Extract every diagnostic, evaluative, or interventional PROCEDURE "
        "performed on or ordered for the patient (swab, chest X-ray, pulse "
        "ox, auscultation, biopsy). Include both performed and pending "
        "procedures. Do not include treatments."
    ),
    pos_example=("COVID swab", "D-30", "I'll order a COVID swab"),
    other_example=("lung exam", "D-25", "I'll listen to your lungs"),
    fine_example=("nasopharyngeal swab", "D-30", "nasopharyngeal swab for COVID"),
)

LOCATION_PROMPT = _entity_prompt(
    category="LOCATION",
    expert_role="clinical anatomist",
    purpose=(
        "Extract every ANATOMICAL location referenced in relation to a "
        "clinical finding (chest, throat, nose, lungs, left lower lobe, "
        "behind left knee). Must be anatomical — not institutional, "
        "temporal, or geographic. Include laterality when stated."
    ),
    pos_example=("left lower lobe", "D-25", "crackles in the left lower lobe"),
    other_example=("upper airway", "D-18", "your upper airway"),
    fine_example=("left lung", "D-25", "the left lung base"),
)

MEDICAL_HISTORY_PROMPT = """<role>
You are an EXPERT clinical internist documenting the patient's background
history for the medical chart. Your job is to distinguish the patient's
pre-existing history — chronic conditions, prior diseases, family history
and genetic predispositions, exposures, triggers, habits, and allergies —
from whatever the clinician is diagnosing at THIS visit. You are rigorous
and conservative; clinical decision support depends on your output being
deterministic and grounded. When in doubt, emit nothing.
</role>

<purpose>
For each mention in the transcript, walk through these decisions to decide
whether and how to emit it as a MEDICAL_HISTORY item:

1. BACKGROUND HISTORY vs CURRENT-VISIT DIAGNOSIS?
   - Pre-existing condition the patient has had before today → history (EMIT).
   - Chronic disease being managed long-term → history (EMIT).
   - Problem the clinician is diagnosing AT THIS visit → SKIP (goes to
     DIAGNOSIS category, not here).
   - When unclear: prefer history if the patient says "I have/had" and
     prefer DIAGNOSIS if the clinician is assessing it as today's problem.

2. PERSONAL or FAMILY history?
   - Patient's own condition ("I have asthma") → emit the condition name
     (or matching curated label).
   - Family member's condition ("my dad has X", "cousin with Y", "grandpa
     died of Z") → emit as "family history of X" or "family hx Y".
   - NEVER strip the family-history framing. "my dad has lung cancer"
     MUST become "family history of lung cancer", not "lung cancer".

3. Is this an EXPOSURE?
   Exposures ARE history items. Types include:
   - Occupational: farm work, chemical plant, hospital worker, daycare
     worker, construction, teaching.
   - Environmental: travel to specific places (e.g. "travel to sarnia"),
     hiking, gardening, carpets at home, pet ownership.
   - Social / contacts: sick contact (husband, wife, child, coworker),
     daycare exposure, school exposure, contact with elderly or young
     children, prior tick bite.
   If the clinician asks about an exposure as a possible cause of illness,
   emit the exposure — whether the answer was yes or no.

4. Is this a TRIGGER of a chronic condition?
   Triggers ARE history items. Examples: cold air, pollen, exercise, pets,
   dust, animal dander, stress, seasonal changes, specific foods.
   If the patient says "X makes my [condition] worse" or "I get it when Y
   happens", X/Y is a trigger — emit it.

5. Is this a HABIT or SOCIAL history?
   Include smoking history (current, former, past, never), alcohol use,
   cannabis / marijuana / vape use, other substance use.

6. Is this an ALLERGY?
   Include drug allergies, food allergies, environmental allergies. ALSO
   include "no allergies" or "no drug allergies" as positive statements
   of absence — the negated form IS the history item.

7. Is this a NEGATED STATEMENT?
   Many history items are phrased negatively by the patient. The negated
   form is clinically meaningful and MUST be preserved exactly as the
   curated list has it:
   - "I don't smoke" → emit "non-smoker" (or "absent smoking" if that is
     the curated form).
   - "No allergies" → emit "no allergies".
   - "No chronic conditions" → emit "no chronic conditions".
   - "I don't have asthma" with curated "absent asthma" → emit
     "absent asthma".
   DO NOT strip "absent", "no", or "non-" prefixes from curated labels.

8. Is this a VACCINE or IMMUNIZATION?
   - Specific named vaccine (coronavirus, shingles, pneumococcal, flu
     shot, HPV, etc.) → EXCLUDE (belongs to TREATMENT, not here).
   - General immunization status ("immunizations up to date", "all shots
     current") → INCLUDE here as history.

Do not duplicate today's DIAGNOSIS entries. Every emitted item MUST cite
turn_id + verbatim quote from the transcript.
</purpose>

<rules>
1. "matched" — use curated labels from the list below. If the transcript
   describes a curated history item using a clinical synonym or variant
   (e.g. "high cholesterol" for curated "hypercholesterolemia", "past
   smoking" for curated "smoking history", "my dad has lung cancer" for
   curated "family history lung cancer"), copy the CURATED LABEL verbatim
   to matched. Cite the transcript phrasing in the quote.
2. "other" — MEDICAL_HISTORY entities in the transcript that are NOT in
   the <curated_labels> list. Use the actual words from the transcript.
3. "fine_grained" — MEDICAL_HISTORY entities in the transcript that are
   finer grained than entries in the <curated_labels> list (e.g.
   "childhood asthma" when curated has "asthma"). Use the actual words
   from the transcript.
4. Do NOT assume or infer. Extract ONLY MEDICAL_HISTORY items explicitly
   mentioned in the transcript.
</rules>

<curated_labels>
{curated_labels_bulleted}
</curated_labels>

<transcript>
{transcript}
</transcript>

<output_contract>
Emit ONE json object and nothing else:
{{
  "matched": [
    {{"label": "family history of asthma", "turn_id": "P-33", "quote": "my dad has asthma"}}
  ],
  "other": [
    {{"text": "hiking exposure", "turn_id": "P-18", "quote": "I went hiking"}}
  ],
  "fine_grained": [
    {{"text": "childhood asthma", "turn_id": "P-8", "quote": "asthma since childhood"}}
  ]
}}
</output_contract>"""

LAB_RESULT_PROMPT = _entity_prompt(
    category="LAB_RESULT",
    expert_role="clinical laboratory specialist",
    purpose=(
        "Extract every LAB_RESULT — lab test, imaging study, vital-sign "
        "measurement, or a stated result. Include test name AND/OR numeric "
        "result if stated. Distinguish from PROCEDURE: PROCEDURE is the "
        "action; LAB_RESULT is the test identity or measured value."
    ),
    pos_example=("SpO2 94%", "D-40", "your oxygen saturation is 94 percent"),
    other_example=("BP 148/90", "D-41", "your BP was 148 over 90"),
    fine_example=("SpO2 low", "D-40", "oxygen saturation is low"),
)


# ─────────────────────────────────────────────────────────────────────────────
# Edge prompts — each edge type has a specialized EXPERT role + rules.
# Edges are populated as PROMPTS but Stage 2 wiring is deferred to v5.
# Output schema differs (edges have src/dst endpoints from Stage 1 nodes).
# The {{entities_json}} placeholder is used by Stage 2 to inject Stage 1 nodes.
# ─────────────────────────────────────────────────────────────────────────────

def _edge_prompt(edge_type: str, expert_role: str, purpose: str,
                  src_types: list, dst_types: list,
                  pos_example: tuple, neg_example: str) -> str:
    # pos_example = (src_id, dst_id, quote, turn_id)
    src, dst, quote, tid = pos_example
    src_types_str = ", ".join(src_types) if src_types else "any"
    dst_types_str = ", ".join(dst_types) if dst_types else "any"
    return f"""<role>
You are an EXPERT {expert_role} analyzing this doctor-patient transcript
excerpt for the hospital. Your sole responsibility is to extract {edge_type}
relationships between pre-extracted clinical entities. Deterministic,
grounded output required.
</role>

<purpose>
{purpose}
</purpose>

<allowed_endpoints>
src must be one of: {src_types_str}
dst must be one of: {dst_types_str}
</allowed_endpoints>

<rules>
1. Both endpoints MUST be entity_ids from <entities>. No free-text endpoints.
2. Every edge MUST cite turn_id + verbatim quote (<=25 tokens from that turn).
3. "matched" — emit here when a single turn has explicit {edge_type} intent
   language as described in <purpose> above.
4. "other" — emit here when {edge_type} is clinically supported by context
   across the visit but lacks a crisp intent phrase in one turn. When
   choosing src/dst node IDs for "other", prefer nodes whose text shares
   as many character-level tokens as possible with the nearest curated
   edge endpoint (so fuzzy string matching can still pair them).
5. "fine_grained" — emit here for tighter-qualification variants of a
   {edge_type} edge (more specific src or dst). When choosing src/dst node
   IDs for "fine_grained", prefer nodes whose text shares as many
   character-level tokens as possible with the nearest curated edge
   endpoint (so fuzzy string matching can still pair them).
6. Hedged or hypothetical mentions ("maybe", "might", "if", "consider") —
   do NOT emit in ANY bucket.
7. No duplicate edges across buckets (same src + dst + type).
8. When in doubt, emit nothing.
</rules>

<negative_example>
{neg_example}
</negative_example>

<entities>{{entities_json}}</entities>
<transcript>{{transcript}}</transcript>

<output_contract>
Emit ONE json object and nothing else:
{{{{
  "matched": [
    {{{{"src": "{src}", "dst": "{dst}", "turn_id": "{tid}", "quote": "{quote}"}}}}
  ],
  "other": [
    {{{{"src": "N_XXX", "dst": "N_YYY", "turn_id": "D-N", "quote": "<=25 tokens verbatim"}}}}
  ],
  "fine_grained": [
    {{{{"src": "N_XXX", "dst": "N_YYY", "turn_id": "D-N", "quote": "<=25 tokens verbatim"}}}}
  ]
}}}}
</output_contract>"""


RULES_OUT_PROMPT = """<role>
You are an EXPERT clinical internist extracting RULES_OUT edges from a
doctor-patient transcript. You receive the full entity list and the full
transcript. Emit edges in three buckets.
</role>

<entities>
{numbered_entities_list}
</entities>

<transcript>
{transcript}
</transcript>

<task>
A RULES_OUT edge means the transcript EXPLICITLY excludes a condition
with language such as: "negative for X", "rules out Y", "not consistent
with Z", "excludes", "ruled out X", "came back negative".

Valid src → dst type patterns:

  - procedure → diagnosis      (negative test rules out dx)
  - symptom → diagnosis        (absent presenting feature excludes dx)
  - symptom → medical_history  (absent feature excludes chronic condition)

Only emit pairs matching one of these type patterns. Arrow direction is
src → dst (evidence of absence → excluded condition). Never reverse.

DO NOT confuse with a working differential ("we should rule out covid"
is a candidate being evaluated, NOT a RULES_OUT — that's an INDICATES).
Only emit when the transcript states the exclusion has happened.

Err toward inclusion; precision is handled downstream.

For every eligible (src, dst) pair from <entities>, walk through:

  Step 1 — CANDIDATE WALK. For each DIAGNOSIS D and each eligible src S,
    decide ACCEPT / REJECT / UNCERTAIN for "S rules out D". One line each,
    ≤ 15 words cited from transcript. Do not skip pairs.
  Step 2 — RESOLVE UNCERTAIN as ACCEPT or REJECT.
  Step 3 — TRANSCRIPT WALK (for OTHER). Scan the transcript for diagnoses
    the clinician excluded that are NOT in <entities>. Emit src → that
    free-text dx.
  Step 4 — SUBTYPE CHECK (for FINE_GRAINED). For each ACCEPT pair, ask if
    transcript specifies a subtype of the ruled-out diagnosis.
  Step 5 — ANTI-COLLAPSE. A single exclusion may rule out multiple
    candidates — include all.
  Step 6 — VERIFY LABELS. MATCHED endpoints are character-for-character
    copies of text values from <entities>.

Three buckets:
  MATCHED      — both src and dst are <entities> entries.
  OTHER        — at least one endpoint is free-text, not in <entities>
                 but named in the transcript.
  FINE_GRAINED — src is <entities>, dst is free-text subtype.

NEGATION IS THE EMISSION CRITERION for this edge type — do not skip
denied/absent symptoms; those are valid src entities here.

DEDUPLICATE within each bucket.
</task>

<output>
Emit ONE JSON object, nothing else:
{{
  "rationale": "<=80 words summarizing coverage",
  "matched": [
    {{"src": "...", "dst": "...", "turn_id": "...", "quote": "...", "confidence": 0.0-1.0}}
  ],
  "other": [
    {{"src": "...", "dst": "...", "turn_id": "...", "quote": "...", "confidence": 0.0-1.0, "dst_in_candidates": false}}
  ],
  "fine_grained": [
    {{"src": "...", "coarse_candidate": "...", "dst": "...", "turn_id": "...", "quote": "...", "confidence": 0.0-1.0}}
  ]
}}
Any bucket may be []. Every bucket key must be present.
</output>
"""

LOCATED_AT_PROMPT = """<role>
You are an EXPERT clinical anatomist extracting LOCATED_AT edges from a
doctor-patient transcript. You receive the full entity list and the full
transcript. Emit edges in three buckets.
</role>

<entities>
{numbered_entities_list}
</entities>

<transcript>
{transcript}
</transcript>

<task>
A LOCATED_AT edge means a finding is anchored to an anatomical location.
Institutional ("hospital", "clinic") and geographic ("home", "Strathroy")
locations are NOT valid dst.

Valid src → dst type patterns:

  - symptom → location           (most common — "pain in my chest")
  - procedure → location         ("chest x-ray" → chest)
  - diagnosis → location         (dx anchored at body region)
  - medical_history → location   (chronic condition at body region —
                                  e.g. copd → lungs)

Only emit pairs matching one of these type patterns. dst must be
anatomical (e.g. "chest", "left lower lobe", "throat", "behind the
knee"). Include laterality when stated. Arrow direction is src → dst
(finding → anatomy). Never reverse.

Err toward inclusion; precision is handled downstream.

For every eligible (src, dst) pair from <entities>, walk through:

  Step 1 — CANDIDATE WALK. For each LOCATION entity L (anatomical only)
    and each eligible src S, decide ACCEPT / REJECT / UNCERTAIN for
    "S is located at L". One line each, ≤ 15 words cited from transcript.
  Step 2 — RESOLVE UNCERTAIN as ACCEPT or REJECT.
  Step 3 — TRANSCRIPT WALK (for OTHER). Scan for anatomical locations
    the clinician named that are NOT in <entities>. Emit src → that
    free-text location.
  Step 4 — SUBTYPE CHECK (for FINE_GRAINED). For each ACCEPT pair, ask
    if transcript specifies a tighter anatomical site (e.g. <entities>
    has "chest", transcript says "left lower chest" — emit FINE_GRAINED
    with coarse_candidate="chest", dst="left lower chest").
  Step 5 — ANTI-COLLAPSE. One symptom may localize to multiple sites;
    emit each.
  Step 6 — VERIFY LABELS. MATCHED endpoints are character-for-character
    copies of text values from <entities>.

Three buckets:
  MATCHED      — both src and dst are <entities> entries.
  OTHER        — at least one endpoint is free-text, not in <entities>
                 but named in the transcript.
  FINE_GRAINED — src is <entities>, dst is free-text anatomical refinement.

NEGATION BLOCKS EMISSION: if the src is denied/absent, do not emit.

DEDUPLICATE within each bucket.
</task>

<output>
Emit ONE JSON object, nothing else:
{{
  "rationale": "<=80 words summarizing coverage",
  "matched": [
    {{"src": "...", "dst": "...", "turn_id": "...", "quote": "...", "confidence": 0.0-1.0}}
  ],
  "other": [
    {{"src": "...", "dst": "...", "turn_id": "...", "quote": "...", "confidence": 0.0-1.0, "dst_in_candidates": false}}
  ],
  "fine_grained": [
    {{"src": "...", "coarse_candidate": "...", "dst": "...", "turn_id": "...", "quote": "...", "confidence": 0.0-1.0}}
  ]
}}
Any bucket may be []. Every bucket key must be present.
</output>
"""

TAKEN_FOR_PROMPT = """<role>
You are an EXPERT clinical pharmacist extracting TAKEN_FOR edges from a
doctor-patient transcript. You receive the full entity list and the full
transcript. Emit edges in three buckets.
</role>

<entities>
{numbered_entities_list}
</entities>

<transcript>
{transcript}
</transcript>

<task>
A TAKEN_FOR edge means the treatment is prescribed, taken, or used FOR
a target condition or symptom. Intent language: "for", "to treat",
"prescribed for", "helps with", "manages", "to control".

Valid src → dst type patterns:

  - treatment → medical_history  (chronic condition being managed — e.g.
                                  insulin → diabetes, spiriva → copd,
                                  atorvastatin → high cholesterol — THIS
                                  IS THE MOST COMMON PATTERN, do not skip)
  - treatment → diagnosis        (active dx being treated today)
  - treatment → symptom          (symptom relief — tylenol → headache)

Only emit pairs matching one of these type patterns. Arrow direction is
src → dst (treatment → target). Never reverse.

Mere co-mention is not enough. Drug-induced symptoms (treatment caused a
symptom as a side effect) go to CAUSES, not TAKEN_FOR.

Err toward inclusion; precision is handled downstream.

For every (src=TREATMENT, dst=SYMPTOM or DIAGNOSIS) pair from <entities>,
walk through:

  Step 1 — CANDIDATE WALK. For each TREATMENT T and each eligible dst D,
    decide ACCEPT / REJECT / UNCERTAIN for "T is taken for D". One line
    each, ≤ 15 words cited from transcript.
  Step 2 — RESOLVE UNCERTAIN as ACCEPT or REJECT.
  Step 3 — TRANSCRIPT WALK (for OTHER). Scan for treatments or target
    conditions the clinician named that are NOT in <entities>.
  Step 4 — SUBTYPE CHECK (for FINE_GRAINED). For each ACCEPT pair, ask
    if transcript specifies a subtype of the treatment or the target.
  Step 5 — ANTI-COLLAPSE. One treatment may be prescribed for multiple
    targets (e.g. prednisone for COPD exacerbation AND inflammation) —
    include each.
  Step 6 — VERIFY LABELS. MATCHED endpoints are character-for-character
    copies of text values from <entities>.

Three buckets:
  MATCHED      — both src and dst are <entities> entries.
  OTHER        — at least one endpoint is free-text, not in <entities>
                 but named in the transcript.
  FINE_GRAINED — src is <entities>, dst is free-text subtype.

NEGATION BLOCKS EMISSION: if the src or dst is denied/absent, do not emit.

DEDUPLICATE within each bucket.
</task>

<output>
Emit ONE JSON object, nothing else:
{{
  "rationale": "<=80 words summarizing coverage",
  "matched": [
    {{"src": "...", "dst": "...", "turn_id": "...", "quote": "...", "confidence": 0.0-1.0}}
  ],
  "other": [
    {{"src": "...", "dst": "...", "turn_id": "...", "quote": "...", "confidence": 0.0-1.0, "dst_in_candidates": false}}
  ],
  "fine_grained": [
    {{"src": "...", "coarse_candidate": "...", "dst": "...", "turn_id": "...", "quote": "...", "confidence": 0.0-1.0}}
  ]
}}
Any bucket may be []. Every bucket key must be present.
</output>
"""

# INDICATES_PROMPT_V12 — one-list V12 style, aligned with the 5 other edge prompts.
# All 6 edge prompts share the same shell; the <task> block differs per edge.
INDICATES_PROMPT = """<role>
You are an EXPERT clinical internist extracting INDICATES edges from a
doctor-patient transcript. You receive the full entity list and the full
transcript. Emit edges in three buckets.
</role>

<entities>
{numbered_entities_list}
</entities>

<transcript>
{transcript}
</transcript>

<task>
An INDICATES edge means evidence suggests / points to / is consistent
with a condition. Recommended src → dst type patterns:

  - symptom → diagnosis            (most common)
  - symptom → medical_history      (chronic condition a symptom points to)
  - procedure → diagnosis          (test ordered to evaluate a diagnosis)
  - medical_history → diagnosis    (exposure / family history pointing to dx)
  - lab_result → diagnosis
  - exposure → diagnosis

These patterns are guidance — emit other plausible clinical relationships
that match the edge meaning if you find them.

Err toward inclusion; precision is handled downstream.

For every eligible (src, dst) pair from <entities>, walk through:

  Step 1 — CANDIDATE WALK. For each DIAGNOSIS D and each evidence entity
    E, decide ACCEPT / REJECT / UNCERTAIN for "E supports D". One line
    each, ≤ 15 words cited from transcript. Do not skip pairs.
  Step 2 — RESOLVE UNCERTAIN as ACCEPT or REJECT.
  Step 3 — TRANSCRIPT WALK (for OTHER). Scan the transcript for
    diagnoses the clinician named or implied that are NOT in
    <entities>. Emit evidence → that free-text dx.
  Step 4 — SUBTYPE CHECK (for FINE_GRAINED). For each ACCEPT pair, ask
    if the transcript specifies a subtype of the diagnosis. If yes,
    emit with coarse_candidate = <entities> entry, dst = subtype
    (free text).
  Step 5 — ANTI-COLLAPSE. Specific AND generic diagnoses can both
    apply; include both (e.g. "covid-19" AND "suspected infection").
    Do not drop one for the other.
  Step 6 — VERIFY LABELS. MATCHED endpoints are character-for-character
    copies of text values from <entities>.

Three buckets:
  MATCHED      — both src and dst are <entities> entries.
  OTHER        — at least one endpoint is free-text, not in <entities>
                 but named or implied in the transcript.
  FINE_GRAINED — src is an <entities> entry, dst is a free-text subtype.

NEGATION BLOCKS EMISSION: if the src is denied/absent ("absent fever",
"no cough", "denies X"), do not emit any edge from it in any bucket.

DEDUPLICATE within each bucket — no repeated (src, dst) pairs.
</task>

<output>
Emit ONE JSON object, nothing else:
{{
  "rationale": "<=80 words summarizing coverage",
  "matched": [
    {{"src": "...", "dst": "...", "turn_id": "...", "quote": "...", "confidence": 0.0-1.0}}
  ],
  "other": [
    {{"src": "...", "dst": "...", "turn_id": "...", "quote": "...", "confidence": 0.0-1.0, "dst_in_candidates": false}}
  ],
  "fine_grained": [
    {{"src": "...", "coarse_candidate": "...", "dst": "...", "turn_id": "...", "quote": "...", "confidence": 0.0-1.0}}
  ]
}}
Any bucket may be []. Every bucket key must be present.
</output>
"""

CAUSES_PROMPT = """<role>
You are an EXPERT clinical internist extracting CAUSES edges from a
doctor-patient transcript. You receive the full entity list and the full
transcript. Emit edges in three buckets.
</role>

<entities>
{numbered_entities_list}
</entities>

<transcript>
{transcript}
</transcript>

<task>
A CAUSES edge means the transcript states or clearly implies etiology /
mechanism — a verb like "caused", "due to", "brought on by", "resulted
from", "led to", "triggered", "from". Co-occurrence is NOT enough.

Valid src → dst type patterns:

  - medical_history → diagnosis       (exposure causes the infection)
  - medical_history → medical_history (smoking → copd; fam hx → epilepsy)
  - medical_history → symptom         (trigger / missed dose causes symptom)
  - symptom → symptom                 (one symptom produces another)
  - diagnosis → diagnosis              (rare — upstream causes exacerbation)
  - treatment → symptom               (drug side effect)

Only emit pairs matching one of these type patterns. Arrow direction is
src → dst (upstream trigger → downstream condition). Never reverse.

DO NOT emit diagnosis → symptom. The diagnosis is the result of evidence,
not the cause of symptoms. "Infection causes fever" is WRONG in this
schema — fever is evidence for infection (INDICATES, not CAUSES).

Err toward inclusion; precision is handled downstream.

For every (src, dst) pair from <entities> where a causal relationship
is plausible, walk through:

  Step 1 — CANDIDATE WALK. For each plausible (src, dst), decide
    ACCEPT / REJECT / UNCERTAIN for "src causes dst". One line each,
    ≤ 15 words cited from transcript.
  Step 2 — RESOLVE UNCERTAIN as ACCEPT or REJECT.
  Step 3 — TRANSCRIPT WALK (for OTHER). Scan for causes or effects
    the clinician named that are NOT in <entities>.
  Step 4 — SUBTYPE CHECK (for FINE_GRAINED). For each ACCEPT pair,
    ask if transcript specifies a subtype of src or dst.
  Step 5 — ANTI-COLLAPSE. One cause may have multiple effects; one
    effect may have multiple contributing causes. Include each.
  Step 6 — VERIFY LABELS. MATCHED endpoints are character-for-character
    copies of text values from <entities>.

Three buckets:
  MATCHED      — both src and dst are <entities> entries.
  OTHER        — at least one endpoint is free-text, not in <entities>
                 but named in the transcript.
  FINE_GRAINED — src is <entities>, dst is free-text subtype.

NEGATION BLOCKS EMISSION: if src or dst is denied/absent, do not emit.

DEDUPLICATE within each bucket.
</task>

<output>
Emit ONE JSON object, nothing else:
{{
  "rationale": "<=80 words summarizing coverage",
  "matched": [
    {{"src": "...", "dst": "...", "turn_id": "...", "quote": "...", "confidence": 0.0-1.0}}
  ],
  "other": [
    {{"src": "...", "dst": "...", "turn_id": "...", "quote": "...", "confidence": 0.0-1.0, "dst_in_candidates": false}}
  ],
  "fine_grained": [
    {{"src": "...", "coarse_candidate": "...", "dst": "...", "turn_id": "...", "quote": "...", "confidence": 0.0-1.0}}
  ]
}}
Any bucket may be []. Every bucket key must be present.
</output>
"""

CONFIRMS_PROMPT = """<role>
You are an EXPERT clinical internist extracting CONFIRMS edges from a
doctor-patient transcript. You receive the full entity list and the full
transcript. Emit edges in three buckets.
</role>

<entities>
{numbered_entities_list}
</entities>

<transcript>
{transcript}
</transcript>

<task>
A CONFIRMS edge means a test or measurement established a condition.
Definitive verb required: "confirms", "diagnostic of", "positive for",
"demonstrates", "shows", "established by", "consistent on imaging".

Valid src → dst type patterns:

  - lab_result → diagnosis    (e.g. ECG confirmed atrial fibrillation)
  - lab_result → symptom      (e.g. temperature 101°F → fever)
  - procedure → diagnosis     (e.g. biopsy confirmed cancer)

Only emit pairs matching one of these type patterns. Arrow direction is
src → dst (test → established condition). Never reverse.

Never emit from patient self-report. A planned future test is NOT a
CONFIRMS — only emit when the result is in the transcript.

Err toward inclusion; precision is handled downstream.

For every (src=PROCEDURE/LAB_RESULT, dst=DIAGNOSIS) pair from <entities>,
walk through:

  Step 1 — CANDIDATE WALK. For each pair, decide ACCEPT / REJECT /
    UNCERTAIN for "src confirmed dst". One line each, ≤ 15 words cited
    from transcript.
  Step 2 — RESOLVE UNCERTAIN as ACCEPT or REJECT.
  Step 3 — TRANSCRIPT WALK (for OTHER). Scan for procedures, labs, or
    diagnoses the clinician named that are NOT in <entities>.
  Step 4 — SUBTYPE CHECK (for FINE_GRAINED). For each ACCEPT pair,
    ask if transcript specifies a subtype of the diagnosis or a
    tighter test variant.
  Step 5 — ANTI-COLLAPSE. One test may confirm multiple diagnoses;
    include each.
  Step 6 — VERIFY LABELS. MATCHED endpoints are character-for-character
    copies of text values from <entities>.

Three buckets:
  MATCHED      — both src and dst are <entities> entries.
  OTHER        — at least one endpoint is free-text, not in <entities>
                 but named in the transcript.
  FINE_GRAINED — src is <entities>, dst is free-text subtype.

NEGATION BLOCKS EMISSION: if the dst is denied (e.g. "test came back
negative for X"), that is a RULES_OUT edge, not a CONFIRMS edge. Do not
emit in this bucket.

DEDUPLICATE within each bucket.
</task>

<output>
Emit ONE JSON object, nothing else:
{{
  "rationale": "<=80 words summarizing coverage",
  "matched": [
    {{"src": "...", "dst": "...", "turn_id": "...", "quote": "...", "confidence": 0.0-1.0}}
  ],
  "other": [
    {{"src": "...", "dst": "...", "turn_id": "...", "quote": "...", "confidence": 0.0-1.0, "dst_in_candidates": false}}
  ],
  "fine_grained": [
    {{"src": "...", "coarse_candidate": "...", "dst": "...", "turn_id": "...", "quote": "...", "confidence": 0.0-1.0}}
  ]
}}
Any bucket may be []. Every bucket key must be present.
</output>
"""


# Placeholder registry — flip ACTIVE[cat]=True to enable at runtime via --categories.
CATEGORY_PROMPTS: dict[str, str | None] = {
    "SYMPTOM":         SYMPTOM_PROMPT,
    "DIAGNOSIS":       DIAGNOSIS_PROMPT,
    "TREATMENT":       TREATMENT_PROMPT,
    "PROCEDURE":       PROCEDURE_PROMPT,
    "LOCATION":        LOCATION_PROMPT,
    "MEDICAL_HISTORY": MEDICAL_HISTORY_PROMPT,
    "LAB_RESULT":      LAB_RESULT_PROMPT,
    "RULES_OUT":       RULES_OUT_PROMPT,
    "LOCATED_AT":      LOCATED_AT_PROMPT,
    "TAKEN_FOR":       TAKEN_FOR_PROMPT,
    "INDICATES":       INDICATES_PROMPT,
    "CAUSES":          CAUSES_PROMPT,
    "CONFIRMS":        CONFIRMS_PROMPT,
}


# ─────────────────────────────────────────────────────────────────────────────
# .env loader + credentials
# ─────────────────────────────────────────────────────────────────────────────
def _load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def get_openrouter_key() -> str:
    if API_KEYS_PATH.exists():
        try:
            k = json.loads(API_KEYS_PATH.read_text()).get("openrouter")
            if k:
                return k
        except Exception:
            pass
    _load_env_file(PROJECT_ROOT / ".env")
    k = os.environ.get("OPENROUTER_API_KEY")
    if not k:
        raise SystemExit(
            f"No OpenRouter key. Put it in {API_KEYS_PATH} (as 'openrouter') "
            f"or {PROJECT_ROOT / '.env'} (as OPENROUTER_API_KEY)."
        )
    return k


# ─────────────────────────────────────────────────────────────────────────────
# OpenRouter client — same as naive kg_extraction.py (no timeout, stream=True)
# ─────────────────────────────────────────────────────────────────────────────
class OpenRouterClient:
    def __init__(self, api_key: str, model: str = OPENROUTER_MODEL):
        from openai import OpenAI
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model = model

    def generate(self, prompt: str, category: str) -> tuple:
        """Return (content, usage, reasoning). reasoning is '' unless the
        category is in REASONING_TRACE_CATEGORIES."""
        model = MODEL_OVERRIDES_BY_CATEGORY.get(category, self.model)
        use_stream = category not in NONSTREAM_CATEGORIES
        temperature = TEMPERATURE_OVERRIDES_BY_CATEGORY.get(category)
        want_reasoning = category in REASONING_TRACE_CATEGORIES

        for attempt in range(MAX_RETRIES):
            try:
                print(f"    [TRACE {category}] attempt {attempt+1}/{MAX_RETRIES}  "
                      f"model={model}  stream={use_stream}  "
                      f"temp={temperature}  reasoning={want_reasoning}  "
                      f"prompt_chars={len(prompt):,}",
                      flush=True)
                t_send = time.time()
                print(f"    [TRACE {category}] PROMPT SENT @ {time.strftime('%H:%M:%S')}",
                      flush=True)

                call_kwargs = dict(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=use_stream,
                )
                if temperature is not None:
                    call_kwargs["temperature"] = temperature
                if want_reasoning:
                    call_kwargs["extra_body"] = {"include_reasoning": True}

                # ── non-streaming branch (any category in NONSTREAM_CATEGORIES) ──
                if not use_stream:
                    response = self.client.chat.completions.create(**call_kwargs)
                    t_end = time.time()
                    msg = response.choices[0].message
                    content = msg.content or ""
                    reasoning = getattr(msg, "reasoning", "") or ""
                    usage = None
                    if response.usage:
                        usage = {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                        }
                    print(f"    [TRACE {category}] NONSTREAM DONE  "
                          f"total_ms={int((t_end-t_send)*1000)}  "
                          f"content_len={len(content):,}  "
                          f"reasoning_len={len(reasoning):,}",
                          flush=True)
                    if usage:
                        print(f"    [TRACE {category}] USAGE  "
                              f"in={usage['prompt_tokens']} "
                              f"out={usage['completion_tokens']}",
                              flush=True)
                    if content:
                        return content, usage, reasoning
                    print(f"    [TRACE {category}] empty content — retrying",
                          flush=True)
                    continue

                # ── streaming branch (everything else) ──
                stream = self.client.chat.completions.create(**call_kwargs)
                t_opened = time.time()
                print(f"    [TRACE {category}] STREAM OPENED  "
                      f"stream_open_ms={int((t_opened-t_send)*1000)}",
                      flush=True)
                content = ""
                last_chunk = None
                n_chunks = 0
                n_bytes = 0
                t_first_chunk = None
                last_heartbeat = t_opened
                for chunk in stream:
                    last_chunk = chunk
                    n_chunks += 1
                    delta = chunk.choices[0].delta
                    if delta.content:
                        content += delta.content
                        n_bytes += len(delta.content)
                        if t_first_chunk is None:
                            t_first_chunk = time.time()
                            print(f"    [TRACE {category}] FIRST CHUNK  "
                                  f"ttfc_ms={int((t_first_chunk-t_send)*1000)}",
                                  flush=True)
                    now = time.time()
                    if now - last_heartbeat > 3.0:
                        print(f"    [TRACE {category}] streaming… chunks={n_chunks} "
                              f"bytes={n_bytes} elapsed={now-t_opened:.1f}s",
                              flush=True)
                        last_heartbeat = now
                t_end = time.time()
                usage = None
                if last_chunk and hasattr(last_chunk, "usage") and last_chunk.usage:
                    u = last_chunk.usage
                    usage = {
                        "prompt_tokens": u.prompt_tokens,
                        "completion_tokens": u.completion_tokens,
                    }
                print(f"    [TRACE {category}] STREAM CLOSED  chunks={n_chunks} "
                      f"bytes={n_bytes} stream_ms={int((t_end-t_opened)*1000)} "
                      f"total_ms={int((t_end-t_send)*1000)}",
                      flush=True)
                if usage:
                    print(f"    [TRACE {category}] USAGE  in={usage['prompt_tokens']} "
                          f"out={usage['completion_tokens']}",
                          flush=True)
                print(f"    [TRACE {category}] RESPONSE RETURNED  "
                      f"content_len={len(content):,}",
                      flush=True)
                if content:
                    return content, usage, ""
                print(f"    [TRACE {category}] empty content — retrying", flush=True)
            except Exception as e:
                print(f"    [TRACE {category}] ERROR {type(e).__name__}: {e} — "
                      f"retry {attempt+1}/{MAX_RETRIES}",
                      flush=True)
                time.sleep(2 ** attempt)
        print(f"    [TRACE {category}] retries exhausted", flush=True)
        return "", None, ""


# ─────────────────────────────────────────────────────────────────────────────
# JSON extraction
# ─────────────────────────────────────────────────────────────────────────────
def extract_json_object(text: str) -> dict | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    if text.strip().startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            inner = parts[1]
            if inner.startswith("json"):
                inner = inner[4:]
            try:
                return json.loads(inner.strip())
            except json.JSONDecodeError:
                pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        block = m.group(0)
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            fixed = re.sub(r",(\s*[}\]])", r"\1", block)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Curated labels by category
# ─────────────────────────────────────────────────────────────────────────────
def load_curated_by_category() -> dict[str, list[str]]:
    entries = json.loads(CURATED_KB.read_text())
    out: dict[str, list[str]] = {c: [] for c in ENTITY_CATEGORIES}
    for e in entries:
        etype = (e.get("entity_type") or "").upper()
        label = (e.get("label") or "").strip()
        if etype in out and label:
            out[etype].append(label)
    for k in out:
        out[k] = sorted(set(out[k]))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Per-category worker (called inside the ThreadPoolExecutor)
# ─────────────────────────────────────────────────────────────────────────────
def run_entity_category(
    cat: str,
    client: OpenRouterClient,
    transcript: str,
    curated_labels: list[str],
    out_per_cat_dir: Path,
) -> dict:
    """Run one entity-category extraction. Returns a result dict."""
    if not ACTIVE.get(cat):
        return {"category": cat, "status": "SKIPPED_INACTIVE"}
    prompt_template = CATEGORY_PROMPTS.get(cat)
    if prompt_template is None:
        raise RuntimeError(
            f"ACTIVE['{cat}']=True but CATEGORY_PROMPTS['{cat}'] is None. "
            "Fill in the prompt before activating."
        )
    t_cat = time.time()
    bulleted = "\n".join(f"- {l}" for l in curated_labels) if curated_labels else "(none)"
    prompt = prompt_template.format(
        curated_labels_bulleted=bulleted,
        transcript=transcript,
    )
    print(f"\n[TRACE {cat}] BEGIN  curated_labels={len(curated_labels)}  "
          f"prompt_chars={len(prompt):,}",
          flush=True)
    print(f"\n┌─── PROMPT ({cat}) ─── {len(prompt):,} chars ───", flush=True)
    print(prompt, flush=True)
    print(f"└─── END PROMPT ({cat}) ───\n", flush=True)

    content, usage, _reasoning = client.generate(prompt, cat)
    call_ms = int((time.time() - t_cat) * 1000)

    if not content:
        print(f"[TRACE {cat}] status=LLM_FAIL  category_ms={call_ms}", flush=True)
        return {"category": cat, "status": "LLM_FAIL",
                "category_ms": call_ms, "usage": usage,
                "matched": [], "other": [], "fine_grained": []}

    print(f"\n┌─── RESPONSE ({cat}) ─── {len(content):,} chars ───", flush=True)
    print(content, flush=True)
    print(f"└─── END RESPONSE ({cat}) ───\n", flush=True)

    parsed = extract_json_object(content)
    if parsed is None:
        print(f"[TRACE {cat}] status=PARSE_FAIL  category_ms={call_ms}", flush=True)
        payload = {"category": cat, "status": "PARSE_FAIL",
                   "category_ms": call_ms, "usage": usage,
                   "raw_content": content,
                   "matched": [], "other": [], "fine_grained": []}
        _write_category_file(out_per_cat_dir, cat, payload)
        return payload

    matched = parsed.get("matched") or []
    other = parsed.get("other") or []
    fine_grained = parsed.get("fine_grained") or []
    payload = {"category": cat, "status": "OK",
               "category_ms": call_ms, "usage": usage,
               "matched": matched, "other": other,
               "fine_grained": fine_grained,
               "raw_content": content}
    print(f"[TRACE {cat}] status=OK  matched={len(matched)} other={len(other)} "
          f"fine_grained={len(fine_grained)}  category_ms={call_ms}",
          flush=True)
    if matched:
        print(f"  MATCHED:      {matched}", flush=True)
    if other:
        print(f"  OTHER:        {other}", flush=True)
    if fine_grained:
        print(f"  FINE_GRAINED: {fine_grained}", flush=True)

    _write_category_file(out_per_cat_dir, cat, payload)
    return payload


def _write_category_file(out_dir: Path, cat: str, payload: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    file = out_dir / f"{out_dir.name}_{cat}.json"
    file.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[TRACE {cat}] saved {file}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 → Stage 2: flatten all entity results into a single node list with
# stable IDs. Used as <entities> input for edge-extraction prompts.
# ─────────────────────────────────────────────────────────────────────────────
def assemble_nodes_from_stage1(stage1_results: dict[str, dict]) -> list[dict]:
    nodes: list[dict] = []
    n = 1
    for cat in ENTITY_CATEGORIES:
        res = stage1_results.get(cat)
        if not res or res.get("status") != "OK":
            continue
        for item in res.get("matched", []):
            nodes.append({
                "id": f"N_{n:03d}",
                "type": cat,
                "text": item.get("label", ""),
                "turn_id": item.get("turn_id", ""),
                "bucket": "matched",
            })
            n += 1
        for item in res.get("other", []):
            nodes.append({
                "id": f"N_{n:03d}",
                "type": cat,
                "text": item.get("text", ""),
                "turn_id": item.get("turn_id", ""),
                "bucket": "other",
            })
            n += 1
        for item in res.get("fine_grained", []):
            nodes.append({
                "id": f"N_{n:03d}",
                "type": cat,
                "text": item.get("text", ""),
                "turn_id": item.get("turn_id", ""),
                "bucket": "fine_grained",
            })
            n += 1
    return nodes


def turn_text_map(transcript: str) -> dict[str, str]:
    return {m.group("turn"): m.group("text").strip()
            for m in re.finditer(
                r"\[(?P<turn>[DP]-\d+)\]\s*(?:[DP]:\s*)?(?P<text>.*?)(?=\[[DP]-\d+\]|\Z)",
                transcript, re.DOTALL)
            if m.group("text").strip()}


def quote_in_turn(quote: str, turn_text: str) -> bool:
    if not quote or not turn_text:
        return False
    nq = re.sub(r"\s+", " ", quote.lower().strip().strip(".,!?\"'"))
    nt = re.sub(r"\s+", " ", turn_text.lower())
    return nq in nt


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 per-edge-category worker
# ─────────────────────────────────────────────────────────────────────────────
def run_edge_category(
    cat: str,
    client: OpenRouterClient,
    transcript: str,
    turn_map: dict[str, str],
    nodes: list[dict],
    out_per_cat_dir: Path,
) -> dict:
    """Run one edge-category extraction. Returns a result dict."""
    if not ACTIVE.get(cat):
        return {"category": cat, "status": "SKIPPED_INACTIVE", "edges": []}
    if not nodes:
        return {"category": cat, "status": "NO_NODES", "edges": []}
    prompt_template = CATEGORY_PROMPTS.get(cat)
    if prompt_template is None:
        raise RuntimeError(
            f"ACTIVE['{cat}']=True but CATEGORY_PROMPTS['{cat}'] is None."
        )
    t_cat = time.time()
    # Build a numbered entities list — one line per Stage 1 node. All types
    # included, no per-edge filtering. The prompt text describes what src/dst
    # should be for each edge; the model routes pairs accordingly.
    numbered_entities_list = "\n".join(
        f'  {i}. "{n["text"]}" ({n["type"]}, turn {n.get("turn_id","")})'
        for i, n in enumerate(nodes, 1)
    )
    prompt = prompt_template.format(
        numbered_entities_list=numbered_entities_list,
        transcript=transcript,
    )
    print(f"\n[TRACE {cat}] BEGIN EDGE  nodes={len(nodes)}  "
          f"prompt_chars={len(prompt):,}",
          flush=True)
    print(f"\n┌─── PROMPT ({cat}) ─── {len(prompt):,} chars ───", flush=True)
    print(prompt, flush=True)
    print(f"└─── END PROMPT ({cat}) ───\n", flush=True)

    content, usage, reasoning = client.generate(prompt, cat)
    call_ms = int((time.time() - t_cat) * 1000)

    # Save reasoning trace alongside the category JSON (one file per transcript).
    if reasoning:
        reasoning_path = out_per_cat_dir / f"{out_per_cat_dir.name}_{cat}_reasoning.txt"
        reasoning_path.write_text(reasoning)
        print(f"    [TRACE {cat}] saved reasoning → {reasoning_path.name} "
              f"({len(reasoning):,} chars)",
              flush=True)

    if not content:
        print(f"[TRACE {cat}] status=LLM_FAIL  category_ms={call_ms}", flush=True)
        payload = {"category": cat, "status": "LLM_FAIL",
                   "category_ms": call_ms, "usage": usage,
                   "reasoning_chars": len(reasoning),
                   "matched": [], "other": [], "fine_grained": [], "edges": []}
        _write_category_file(out_per_cat_dir, cat, payload)
        return payload

    print(f"\n┌─── RESPONSE ({cat}) ─── {len(content):,} chars ───", flush=True)
    print(content, flush=True)
    print(f"└─── END RESPONSE ({cat}) ───\n", flush=True)

    parsed = extract_json_object(content)
    if parsed is None or not any(k in parsed for k in ("matched", "other", "fine_grained")):
        print(f"[TRACE {cat}] status=PARSE_FAIL  category_ms={call_ms}", flush=True)
        payload = {"category": cat, "status": "PARSE_FAIL",
                   "category_ms": call_ms, "usage": usage,
                   "reasoning_chars": len(reasoning),
                   "raw_content": content,
                   "matched": [], "other": [], "fine_grained": [], "edges": []}
        _write_category_file(out_per_cat_dir, cat, payload)
        return payload

    # Pass-through: no validation on edges. Drop only empty/self-loop edges.
    # The model's labels and quotes are saved as-is; downstream handles matching.
    def _passthrough(raw_list) -> list:
        out = []
        for e in raw_list or []:
            src = e.get("src", "")
            dst = e.get("dst", "")
            if not src or not dst or src == dst:
                continue
            out.append({
                "src": src, "dst": dst, "type": cat,
                "turn_id": e.get("turn_id", ""),
                "quote": (e.get("quote") or "")[:300],
            })
        return out

    matched = _passthrough(parsed.get("matched"))
    other = _passthrough(parsed.get("other"))
    fine_grained = _passthrough(parsed.get("fine_grained"))
    vstats = {"matched_kept": len(matched),
              "other_kept": len(other),
              "fine_grained_kept": len(fine_grained)}

    all_edges = matched + other + fine_grained

    payload = {"category": cat, "status": "OK",
               "category_ms": call_ms, "usage": usage,
               "reasoning_chars": len(reasoning),
               "matched": matched, "other": other, "fine_grained": fine_grained,
               "edges": all_edges, "validation": vstats,
               "raw_content": content}
    print(f"[TRACE {cat}] status=OK  matched={len(matched)} other={len(other)} "
          f"fine_grained={len(fine_grained)}  "
          f"(kept={vstats['matched_kept']+vstats['other_kept']+vstats['fine_grained_kept']})  "
          f"category_ms={call_ms}",
          flush=True)
    for bucket_name, bucket in (("matched", matched), ("other", other),
                                 ("fine_grained", fine_grained)):
        for e in bucket:
            print(f"  EDGE[{bucket_name}]: {e['src']} -[{e['type']}]-> {e['dst']}  "
                  f"turn={e['turn_id']}  quote={e['quote']!r}", flush=True)

    _write_category_file(out_per_cat_dir, cat, payload)
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# Per-transcript processor
# ─────────────────────────────────────────────────────────────────────────────
def process_transcript(
    res_id: str,
    client: OpenRouterClient,
    curated: dict[str, list[str]],
    out_base: Path,
    reuse_nodes_from: Path | None = None,
) -> dict:
    """Run Stage 1 (parallel entities) + Stage 2 (parallel edges) for one
    transcript. Writes per-category files + a combined roll-up. Returns the
    roll-up dict.

    If reuse_nodes_from is set, Stage 1 is skipped for this transcript and
    nodes are loaded from {reuse_nodes_from}/{res_id}/{res_id}_all.json."""
    t_script = time.time()
    transcript_path = TRANSCRIPT_DIR / res_id / f"{res_id}.txt"
    if not transcript_path.exists():
        return {"res_id": res_id, "status": "TRANSCRIPT_MISSING",
                "path": str(transcript_path)}
    transcript = transcript_path.read_text()
    n_turns = len(re.findall(r"\[[DP]-\d+\]", transcript))
    t_map = turn_text_map(transcript)

    out_root = out_base / res_id
    out_root.mkdir(parents=True, exist_ok=True)

    active_entities = [c for c in ENTITY_CATEGORIES if ACTIVE.get(c)]
    active_edges    = [c for c in EDGE_CATEGORIES    if ACTIVE.get(c)]

    print("=" * 72)
    print(f"  {res_id}  turns={n_turns} chars={len(transcript):,}")
    print(f"  active entities: {active_entities}")
    print(f"  active edges:    {active_edges}")
    if reuse_nodes_from:
        print(f"  reuse nodes from: {reuse_nodes_from}")
    print("=" * 72)

    # Stage 1: parallel entity extraction OR load nodes from previous run
    stage1_start = time.time()
    stage1_results: dict[str, dict] = {}
    nodes: list[dict] = []
    if reuse_nodes_from is not None:
        prev_all = reuse_nodes_from / res_id / f"{res_id}_all.json"
        if not prev_all.exists():
            print(f"[TRACE {res_id}] REUSE FAIL — no {prev_all}", flush=True)
            return {"res_id": res_id, "status": "REUSE_FAIL",
                    "path": str(prev_all)}
        prev = json.loads(prev_all.read_text())
        nodes = prev.get("nodes", []) or []
        stage1_results = prev.get("stage1_results", {}) or {}
        print(f"[TRACE {res_id}] REUSE: loaded {len(nodes)} nodes "
              f"from {prev_all.name}", flush=True)
        # Copy per-category entity files to new run dir for completeness
        for cat in ENTITY_CATEGORIES:
            src = reuse_nodes_from / res_id / f"{res_id}_{cat}.json"
            if src.exists():
                (out_root / src.name).write_text(src.read_text())
    elif active_entities:
        print(f"[TRACE {res_id}] Stage 1 — parallel entity extraction "
              f"({len(active_entities)} workers)", flush=True)
        with cf.ThreadPoolExecutor(max_workers=max(1, len(active_entities))) as pool:
            futs = {
                pool.submit(run_entity_category, cat, client, transcript,
                            curated.get(cat, []), out_root): cat
                for cat in active_entities
            }
            for fut in cf.as_completed(futs):
                cat = futs[fut]
                try:
                    stage1_results[cat] = fut.result()
                except Exception as e:
                    print(f"[TRACE {res_id}/{cat}] WORKER ERROR "
                          f"{type(e).__name__}: {e}", flush=True)
                    stage1_results[cat] = {"category": cat,
                                           "status": f"WORKER_ERROR: {e}",
                                           "matched": [], "other": [],
                                           "fine_grained": []}
    stage1_ms = int((time.time() - stage1_start) * 1000)
    print(f"[TRACE {res_id}] Stage 1 done  stage1_ms={stage1_ms}", flush=True)

    # Assemble nodes for Stage 2 (only if not already loaded via reuse)
    if reuse_nodes_from is None:
        nodes = assemble_nodes_from_stage1(stage1_results)
    print(f"[TRACE {res_id}] nodes for Stage 2: {len(nodes)}", flush=True)

    # Stage 2: parallel edge extraction
    stage2_start = time.time()
    stage2_results: dict[str, dict] = {}
    if active_edges and nodes:
        print(f"[TRACE {res_id}] Stage 2 — parallel edge extraction "
              f"({len(active_edges)} workers)", flush=True)
        with cf.ThreadPoolExecutor(max_workers=max(1, len(active_edges))) as pool:
            futs = {
                pool.submit(run_edge_category, cat, client, transcript,
                            t_map, nodes, out_root): cat
                for cat in active_edges
            }
            for fut in cf.as_completed(futs):
                cat = futs[fut]
                try:
                    stage2_results[cat] = fut.result()
                except Exception as e:
                    print(f"[TRACE {res_id}/{cat}] WORKER ERROR "
                          f"{type(e).__name__}: {e}", flush=True)
                    stage2_results[cat] = {"category": cat,
                                           "status": f"WORKER_ERROR: {e}",
                                           "edges": []}
    elif active_edges and not nodes:
        print(f"[TRACE {res_id}] Stage 2 skipped — no Stage 1 nodes", flush=True)
    stage2_ms = int((time.time() - stage2_start) * 1000)
    print(f"[TRACE {res_id}] Stage 2 done  stage2_ms={stage2_ms}", flush=True)

    # Per-transcript roll-up
    total_in = sum((r.get("usage") or {}).get("prompt_tokens", 0)
                   for r in list(stage1_results.values()) + list(stage2_results.values()))
    total_out = sum((r.get("usage") or {}).get("completion_tokens", 0)
                    for r in list(stage1_results.values()) + list(stage2_results.values()))
    total_ms = int((time.time() - t_script) * 1000)

    roll_up = {
        "res_id": res_id,
        "status": "OK",
        "model": OPENROUTER_MODEL,
        "active_entities": active_entities,
        "active_edges":    active_edges,
        "n_turns": n_turns,
        "n_chars": len(transcript),
        "n_nodes": len(nodes),
        "n_edges": sum(len(r.get("edges", [])) for r in stage2_results.values()),
        "stage1_ms": stage1_ms,
        "stage2_ms": stage2_ms,
        "total_ms": total_ms,
        "total_prompt_tokens": total_in,
        "total_completion_tokens": total_out,
        "stage1_results": stage1_results,
        "stage2_results": stage2_results,
        "nodes": nodes,
    }
    (out_root / f"{res_id}_all.json").write_text(
        json.dumps(roll_up, indent=2, ensure_ascii=False)
    )
    print(f"[TRACE {res_id}] total={total_ms}ms  "
          f"tokens in={total_in:,} out={total_out:,}  "
          f"nodes={len(nodes)}  edges={roll_up['n_edges']}", flush=True)
    return roll_up


def collect_all_res_ids() -> list[str]:
    return sorted(d.name for d in TRANSCRIPT_DIR.glob("RES*") if d.is_dir())


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res-ids", nargs="+", default=None,
                    help="Transcript ids (e.g. RES0200 RES0203). "
                         "If omitted, runs all 20.")
    ap.add_argument("--output", type=str, default=str(OUT_DIR))
    ap.add_argument("--categories", nargs="+", default=None,
                    help="Override ACTIVE — run only these categories. "
                         "Allowed: " + str(sorted(set(ENTITY_CATEGORIES) | set(EDGE_CATEGORIES))))
    ap.add_argument("--workers", type=int, default=4,
                    help="Number of transcripts to process in parallel "
                         "(default 4). Each transcript still runs its "
                         "categories in parallel internally.")
    ap.add_argument("--reuse-nodes-from", type=str, default=None,
                    help="Path to a previous run dir (e.g. "
                         "eir_results/..._v7/260421_130623). If set, Stage 1 "
                         "entity extraction is skipped and nodes are loaded "
                         "from that run's per-transcript {rid}_all.json files. "
                         "Useful for iterating on edge prompts without "
                         "re-running entities.")
    args = ap.parse_args()

    # --categories override
    if args.categories:
        all_known = set(ENTITY_CATEGORIES) | set(EDGE_CATEGORIES)
        unknown = [c for c in args.categories if c not in all_known]
        if unknown:
            sys.exit(f"ERROR: unknown --categories: {unknown}. "
                     f"Allowed: {sorted(all_known)}")
        for c in ACTIVE:
            ACTIVE[c] = c in args.categories
        print(f"[TRACE] --categories override: running only {args.categories}",
              flush=True)

    t_batch = time.time()
    print(f"[TRACE] batch start @ {time.strftime('%H:%M:%S')}", flush=True)

    print(f"[TRACE] loading OpenRouter API key …", flush=True)
    key = get_openrouter_key()
    print(f"[TRACE] key loaded", flush=True)

    print(f"[TRACE] loading curated KB by category …", flush=True)
    curated = load_curated_by_category()
    for c in ENTITY_CATEGORIES:
        flag = "ON " if ACTIVE.get(c) else "off"
        print(f"[TRACE]   {c:16} {flag}  labels={len(curated.get(c, []))}", flush=True)
    for c in EDGE_CATEGORIES:
        flag = "ON " if ACTIVE.get(c) else "off"
        print(f"[TRACE]   {c:16} {flag}", flush=True)

    print(f"[TRACE] constructing OpenRouter client …", flush=True)
    client = OpenRouterClient(key)

    # Target transcripts
    if args.res_ids:
        res_ids = args.res_ids
    else:
        res_ids = collect_all_res_ids()

    # Each batch lands in its own timestamped subdir so prior runs are
    # never overwritten.
    run_stamp = time.strftime("%y%m%d_%H%M%S")
    out_base = Path(args.output) / run_stamp
    out_base.mkdir(parents=True, exist_ok=True)
    print(f"[TRACE] batch: {len(res_ids)} transcripts, "
          f"workers={args.workers}, output={out_base}",
          flush=True)

    # Parallel per transcript (each transcript still runs its categories
    # in parallel internally). max_workers controlled by --workers.
    reuse_path = Path(args.reuse_nodes_from) if args.reuse_nodes_from else None
    if reuse_path:
        print(f"[TRACE] reuse-nodes-from: {reuse_path}", flush=True)
    per_run_summaries: list[dict] = []
    with cf.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = {
            pool.submit(process_transcript, rid, client, curated, out_base,
                        reuse_path): rid
            for rid in res_ids
        }
        for fut in cf.as_completed(futs):
            rid = futs[fut]
            try:
                summary = fut.result()
            except Exception as e:
                print(f"[TRACE {rid}] FATAL {type(e).__name__}: {e}",
                      flush=True)
                summary = {"res_id": rid,
                           "status": f"FATAL: {type(e).__name__}: {e}"}
            per_run_summaries.append(summary)

    # Batch-level aggregate
    total_ms = int((time.time() - t_batch) * 1000)
    total_in = sum(s.get("total_prompt_tokens", 0) for s in per_run_summaries)
    total_out = sum(s.get("total_completion_tokens", 0) for s in per_run_summaries)
    total_nodes = sum(s.get("n_nodes", 0) for s in per_run_summaries)
    total_edges = sum(s.get("n_edges", 0) for s in per_run_summaries)

    agg = {
        "model": OPENROUTER_MODEL,
        "n_transcripts": len(res_ids),
        "n_transcripts_ok": sum(1 for s in per_run_summaries if s.get("status") == "OK"),
        "n_transcripts_failed": sum(1 for s in per_run_summaries if s.get("status") != "OK"),
        "total_ms": total_ms,
        "total_prompt_tokens": total_in,
        "total_completion_tokens": total_out,
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "active_entities": [c for c in ENTITY_CATEGORIES if ACTIVE.get(c)],
        "active_edges":    [c for c in EDGE_CATEGORIES    if ACTIVE.get(c)],
        "per_transcript": per_run_summaries,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_base / "_batch_summary.json").write_text(
        json.dumps(agg, indent=2, ensure_ascii=False)
    )

    print("=" * 72)
    print(f"BATCH DONE  transcripts={len(res_ids)} "
          f"ok={agg['n_transcripts_ok']} failed={agg['n_transcripts_failed']}")
    print(f"  total time: {total_ms}ms ({total_ms/1000:.1f}s)")
    print(f"  tokens: in={total_in:,} out={total_out:,}")
    print(f"  nodes total: {total_nodes}  edges total: {total_edges}")
    print(f"  summary: {out_base / '_batch_summary.json'}")


if __name__ == "__main__":
    main()
