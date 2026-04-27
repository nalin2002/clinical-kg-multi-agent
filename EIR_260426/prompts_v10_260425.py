"""
prompts_v10_260425.py
=====================

v10 = prompts_general_260425.py with the four recall-pressure levers
REMOVED (they over-emitted on this curator without lifting hits):
  1. Recall floor language ("verbatim or near-verbatim → ALWAYS emit")
  2. Bucket scoping (matched=RECALL / other=PRECISION-ORIENTED)
  3. Comprehensive review / recall sweep step
  4. "One quote, multiple labels" rule

What remains: out-of-corpus example anchors, compound preservation,
color priority, phonetic-variant rule, subjective/objective tie-break,
synonym mapping principle, three-bucket schema, evidence rules.

Format placeholders match v7 exactly:
  Entity prompts: {curated_labels_bulleted}, {transcript}
  Edge prompts:   {numbered_entities_list}, {transcript}

Use:
    import prompts_v10_260425 as v10p
    import smoke_test_glm47_specialized_parallel_v8_optimized_260425 as v8
    v8.CATEGORY_PROMPTS = v10p.CATEGORY_PROMPTS
    # then run the v9 pipeline normally

Created: 260425
"""

# ─────────────────────────────────────────────────────────────────────────────
# SYMPTOM
# ─────────────────────────────────────────────────────────────────────────────
SYMPTOM_PROMPT = """<role>
You are an EXPERT clinical internist documenting the patient's symptoms
for the medical chart. Your job is to extract every SYMPTOM the patient
reports, the patient denies when asked, or the clinician directly observes.
</role>

<what_counts_as_a_symptom>
Examples by system (illustrative, not exhaustive):
  Constitutional:   fatigue, fever, chills, weight loss, malaise
  Respiratory:      cough, shortness of breath, wheezing, sputum,
                    sore throat, nasal congestion, runny nose
  Cardiovascular:   chest pain, palpitations, syncope, peripheral edema
  Gastrointestinal: nausea, vomiting, diarrhea, abdominal pain,
                    melena, hematochezia, heartburn
  Genitourinary:    dysuria, hematuria, urinary frequency, flank pain
  Neurologic:       headache, dizziness, seizure, weakness, paresthesia
  Musculoskeletal:  joint pain, muscle ache, swelling, stiffness
  Dermatologic:     rash, itching, lesion, bruising
  Sensory:          loss of taste, loss of smell, blurred vision,
                    hearing loss, tinnitus
  Psychiatric:      anxiety, depressed mood, insomnia
</what_counts_as_a_symptom>

<what_is_NOT_a_symptom>
  - A diagnosed disease (covid-19, asthma exacerbation) → DIAGNOSIS
  - A pre-existing chronic condition (years-long) → MEDICAL_HISTORY
  - A medication or therapy (Tylenol, hydration) → TREATMENT
  - A test or procedure (echocardiogram, CBC) → PROCEDURE
  - A measured value (BP 148/90, hemoglobin 8.2) → LAB_RESULT
  - An anatomical location (chest, throat) → LOCATION
</what_is_NOT_a_symptom>

<purpose>
For each mention in the transcript, walk through these decisions to decide
whether and how to emit it as a SYMPTOM item:

1. IS THIS A SYMPTOM OR A CURRENT-VISIT DIAGNOSIS?
   - Something the patient FEELS or EXPERIENCES → symptom (EMIT).
   - Something the clinician DIAGNOSES as a disease → SKIP (goes to
     DIAGNOSIS).
   - Observed signs the clinician names → symptom (EMIT).

   PHYSICIAN EXAM FINDINGS — always extract.
   Anything the clinician describes during the physical exam that is
   not itself a diagnosis is a SYMPTOM. This includes auscultation
   findings (bruits, murmurs, rubs, gallops, crackles, wheezes,
   rhonchi), palpation findings (tenderness with anatomical specifier,
   masses, fluctuance, organomegaly, thyromegaly), inspection findings
   (erythema, swelling, effusion, bruising, rash), neurologic exam
   signs (positive straight leg raise, reflex changes, weakness,
   numbness), and range-of-motion findings (decreased flexion, pain
   with extension). Emit each one as a SYMPTOM with the full curated
   form when present (e.g. `right knee effusion`, `systolic ejection
   murmur`, `mild thyromegaly`, `l5 tenderness`), preserving laterality
   and severity qualifiers.

2. IS THE PATIENT AFFIRMING OR DENYING?
   - Affirming → emit the positive form.
   - Denying in response to a clinician question → check
     <curated_labels> for the exact denial form to emit. Do NOT emit
     the positive form when the patient denies.
   - Denying without being asked → check <curated_labels> for the
     exact denial form; otherwise skip.
   - Out-of-corpus example for grounding only: if curated has
     `absent palpitations` and the patient says "no, I haven't felt
     my heart racing", emit `absent palpitations` (the curated
     denial form). Do NOT emit `palpitations`.

   STACKED ROS — BLANKET NEGATIVE.
   When the clinician asks ONE question listing MULTIPLE symptoms
   (e.g. "any abdominal pain? fever, chills?") and the patient
   gives a single blanket negative answer ("none of that", "no",
   "nope", "I'm fine", "not really"), treat EACH listed symptom
   as denied. Emit ONE separate `absent X` per listed item, using
   the curated denial form when present in <curated_labels>.
   - Out-of-corpus example for grounding only: clinician says
     "any nausea or vomiting?", patient says "no" → emit BOTH
     `absent nausea` AND `absent vomiting` as two separate
     items. Never collapse them into one.

   STACKED ROS — PARTIAL AFFIRM = IMPLICIT DENIAL.
   When the clinician lists multiple symptoms in one question and
   the patient affirms ONE (or a subset) but says nothing about
   the others, treat the unmentioned ones as IMPLICITLY DENIED.
   This follows clinical chart convention: ROS items the patient
   does not affirm are recorded as denied. Emit the positive form
   for the affirmed items AND emit `absent X` for each unmentioned
   listed item.
   - Out-of-corpus example for grounding only: clinician says
     "weight loss or decreased appetite or night sweats? coughs?",
     patient says "slightly decreased appetite" → emit
     `decreased appetite` (positive) AND `absent weight loss`,
     `absent night sweats`, `absent cough` (implicit denials).

   CLINICIAN EXAM-FINDING DENIAL.
   When the clinician documents an exam finding using language
   like "no X", "X is normal", "no evidence of X", "clear of X",
   "negative for X", treat it as a denial of that symptom. Emit
   `absent X` using the curated form when present in
   <curated_labels>.
   - Out-of-corpus example for grounding only: clinician says
     "your lower extremities have no edema" → emit
     `absent lower extremity edema`.

3. WHICH CURATED VARIANT DO I USE?
   Multiple curated forms may exist for similar content (a base form
   and variants with different qualifiers, severities, or temporal
   modifiers). Pick the curated variant that best fits BOTH the
   transcript content AND the clinician's framing if the clinician
   restates the symptom. The clinician's framing is authoritative when
   the patient's lay description and the clinician's clinical
   restatement diverge.

   SUBJECTIVE vs OBJECTIVE: prefer the OBJECTIVE curated form when
   measurement or examination supports it, and prefer the SUBJECTIVE
   form when the patient describes the experience but no measurement
   or observation is recorded.
   - Out-of-corpus example for grounding only: if curated has BOTH
     `chest pain` and `pleuritic chest pain`, and the transcript
     describes pain that is worse with deep breathing, prefer the
     more specific objective form `pleuritic chest pain` over the
     bare `chest pain`.
   - If the transcript phrase is itself the curated form, emit THAT
     exact form rather than upgrading to a more clinical synonym.

4. PRESERVE QUALIFIERS — do NOT strip:
   Attribute qualifiers carry clinical meaning. When the transcript
   provides any of severity, timing, location, quality, laterality, or
   color, preserve them by selecting the curated variant that includes
   the qualifier. A base curated label and a more-qualified variant of
   the same concept are NOT interchangeable. Never drop a qualifier
   the curated label carries.

   COLOR PRIORITY: When the patient describes multiple colors of the
   same fluid in a single phrase (sputum, urine, stool, discharge,
   etc.), pick the FIRST color mentioned for the curated label and
   ignore subsequent colors in that phrase.
   - Out-of-corpus example for grounding only: if curated has
     `bright red blood per rectum` and transcript says "I saw some
     bright red and dark spots", emit `bright red blood per rectum`
     (FIRST color = bright red; ignore "dark"). Do not invent a
     `dark blood` entity.

5. SYNONYM MAPPING — patient layman terms → curated clinical terms.
   Use clinical training to bridge — do not require literal string
   overlap. When a transcript phrase plausibly matches a curated
   symptom via clinical synonymy (lay term, abbreviation, colloquial
   description, paraphrase), copy the CURATED LABEL verbatim into
   matched. The underlying principle is to use medical knowledge to
   bridge any layman → clinical mapping where the meaning is
   unambiguous.
   - Out-of-corpus examples for grounding only (none of these appear
     in this dataset; do NOT use them as a checklist — use them to
     calibrate the SHAPE of a synonym mapping):
     * if curated has `polyuria` and transcript says "I'm peeing way
       more than usual" → emit `polyuria`.
     * if curated has `syncope` and transcript says "I passed out for
       a minute" → emit `syncope`.
     * if curated has `melena` and transcript says "black tarry
       stools" → emit `melena`.
     * if curated has `peripheral edema` and transcript says "my
       ankles are swollen" → emit `peripheral edema`.

   PHONETIC AND ORTHOGRAPHIC VARIANTS: When the transcript renders a
   medical term with non-standard spelling (transcription artefact,
   patient mishearing), match through phonetic and orthographic
   variants using your medical knowledge. The curated label is the
   canonical answer.


7. COMPOUND SYMPTOMS — preserve the full phrase.
   Curated may contain multi-word symptom labels with linked
   attributes. If the transcript supports such a compound, emit the
   compound label verbatim. Do NOT break it into separate atomic
   symptoms.
   - Out-of-corpus example for grounding only: if curated has BOTH
     `heart failure` and `decompensated heart failure`, and the
     transcript says the patient's heart failure has been "worse
     this week", emit `decompensated heart failure` (the qualified
     compound), not the bare `heart failure`.
</purpose>

<rules>
1. "matched" — use curated
   labels from the list below. Copy the CURATED LABEL verbatim into
   matched, regardless of how the transcript phrased it. For denials,
   emit whichever denial form appears in <curated_labels> for that
   concept verbatim.
2. "other" — SYMPTOM entities in the transcript
   that are NOT in the <curated_labels> list. Use the actual words
   from the transcript. When in doubt for "other", emit nothing.
3. "fine_grained" — SYMPTOM entities in the
   transcript that are finer-grained than entries in <curated_labels>.
   Use the actual words from the transcript. When in doubt for
   "fine_grained", emit nothing.
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
Emit ONE json object and nothing else, with this exact shape:
{{
  "matched":      [{{"label": "...", "turn_id": "...", "quote": "..."}}],
  "other":        [{{"text":  "...", "turn_id": "...", "quote": "..."}}],
  "fine_grained": [{{"text":  "...", "turn_id": "...", "quote": "..."}}]
}}
Any bucket may be []. Every bucket key must be present.
</output_contract>"""


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSIS
# ─────────────────────────────────────────────────────────────────────────────
DIAGNOSIS_PROMPT = """<role>
You are an EXPERT clinical internist analyzing this doctor-patient
transcript excerpt for the hospital. Your sole responsibility is to
extract every DIAGNOSIS the clinician is considering at THIS visit —
confirmed, suspected, possible, or explicitly ruled-out. You are
</role>

<what_counts_as_a_diagnosis>
Examples by system (illustrative, not exhaustive):
  Infectious:       covid-19, influenza, pneumonia, urinary tract
                    infection, cellulitis, lyme disease, sepsis
  Cardiovascular:   congestive heart failure, atrial fibrillation,
                    hypertension, myocardial infarction, angina
  Pulmonary:        asthma exacerbation, copd exacerbation,
                    common cold, viral upper respiratory infection
  Endocrine:        diabetic ketoacidosis, thyroid storm,
                    hyperthyroidism, hypothyroidism
  Neurologic:       stroke, seizure, epilepsy, migraine
  Gastrointestinal: acute pancreatitis, ibd flare, peptic ulcer disease,
                    gastroenteritis
  Renal:            acute kidney injury, nephrolithiasis,
                    chronic kidney disease exacerbation
  Hematologic:      anemia, deep vein thrombosis, pulmonary embolism
  Dermatologic:     contact dermatitis, eczema flare, psoriasis flare
</what_counts_as_a_diagnosis>

<what_is_NOT_a_diagnosis>
  - A pre-existing chronic condition the patient has had for years → MEDICAL_HISTORY
  - A symptom (chest pain, fatigue, dizziness) → SYMPTOM
  - A treatment or medication → TREATMENT
  - A test or procedure → PROCEDURE
  - A measured value or finding (hemoglobin 8.2, ejection fraction 45%) → LAB_RESULT
</what_is_NOT_a_diagnosis>

<purpose>
Extract every DIAGNOSIS the clinician considers in THIS visit. Include
confirmed diagnoses, suspected / possible / likely working hypotheses,
explicit rule-outs, and exacerbations of chronic conditions. Skip the
patient's pre-existing chronic conditions (those go to MEDICAL_HISTORY).
Skip symptoms and complaints (those go to SYMPTOM).

Every emitted item MUST cite turn_id + verbatim quote from the transcript.
</purpose>

<rules>
1. "matched" — use curated labels from the list below.
   When the transcript describes a curated diagnosis using a synonym,
   abbreviation, colloquial form, or a qualifier phrase that maps to a
   curated label, copy the CURATED LABEL verbatim. Cite the transcript
   phrasing in the quote.

2. "other" — DIAGNOSIS entities in the transcript
   that are NOT in <curated_labels>. Use the transcript words.
3. "fine_grained" — DIAGNOSIS entities finer than
   <curated_labels> entries (subtype, anatomical specifier, or
   etiologic qualifier). Use the transcript words.
4. Do NOT assume or infer. Extract ONLY diagnoses the clinician
   EXPLICITLY considers in the transcript.

5. EXHAUSTIVE SCAN — before finalizing, walk through EVERY label in
   <curated_labels> and ask: is the clinician ACTIVELY WORKING on this
   diagnosis for THIS patient — asserting it, suspecting it, hedging
   on it, listing it on a differential, stating it flatly, ruling it
   out after work-up, or acknowledging a patient-voiced concern? If
   yes, emit in matched.

   Do NOT emit when the mention is:
     - a procedure / test mention only (the test is a PROCEDURE; only
       extract the diagnosis if the clinician is actively working it
       up)
     - preventive talk
     - a screening question with a flat negative answer and no further
       clinical reasoning
     - patient's distant history (chronic condition mentioned as
       background → MEDICAL_HISTORY)
     - sick contact mention (a household member's diagnosis is
       exposure history, not THIS patient's diagnosis)
     - general health discussion unrelated to this patient's problem

6. PREFER QUALIFIED CURATED LABELS WHEN CONTEXT SUPPORTS THEM.
   When the clinician uses a base term but the patient's presentation
   makes a qualifier clear (anatomic system, organ-level specifier, or
   acuity such as exacerbation of a chronic condition), prefer the
   qualified curated label if it exists. If the transcript provides no
   disambiguating context, use the base curated label.
</rules>

<curated_labels>
{curated_labels_bulleted}
</curated_labels>

<transcript>
{transcript}
</transcript>

<output_contract>
Emit ONE json object and nothing else, with this exact shape:
{{
  "matched":      [{{"label": "...", "turn_id": "...", "quote": "..."}}],
  "other":        [{{"text":  "...", "turn_id": "...", "quote": "..."}}],
  "fine_grained": [{{"text":  "...", "turn_id": "...", "quote": "..."}}]
}}
Any bucket may be []. Every bucket key must be present.
</output_contract>"""


# ─────────────────────────────────────────────────────────────────────────────
# TREATMENT
# ─────────────────────────────────────────────────────────────────────────────
TREATMENT_PROMPT = """<role>
You are an EXPERT clinical pharmacist analyzing this doctor-patient
transcript excerpt for the hospital. Your sole responsibility is to
extract every MEDICATION and TREATMENT explicitly mentioned in the
transcript. Pay particular attention to pharmaceuticals and medical
devices, which are easy to miss when transcript spelling is non-standard.


MEDICATIONS and TREATMENTs include pharmaceuticals (generic and brand),
injections, infusions, inhalers and other delivery devices, medical
devices, vaccines, oxygen therapy, and behavioral / lifestyle
instructions when they are explicitly part of management.

EXCLUDE diagnostic procedures (imaging, blood draws, lab tests, swabs,
panels) — those belong to PROCEDURE.

You are rigorous and grounded for free-text claims, but you do NOT
under-emit. For "other" and "fine_grained": emit when the phrase
clearly describes a medication, treatment, or management instruction,
even if it is not on the curated list. Skip only filler and
conversational asides.
</role>

<what_counts_as_a_treatment>
Examples by class (illustrative, not exhaustive):
  Medications:      metoprolol, lisinopril, lasix, atorvastatin,
                    insulin, metformin, antibiotics, tylenol,
                    ibuprofen, claritin, ventolin, spiriva
  Inhalers/devices: bronchodilator, rescue inhaler, maintenance inhaler,
                    nebulizer, cpap, oxygen therapy
  Vaccines:         flu shot, covid-19 vaccine, shingles vaccine,
                    pneumococcal vaccine
  Lifestyle:        dietary modification, hydration, isolation,
                    weight loss, exercise prescription, sodium restriction
  Topical/local:    salt water gargle, hot compress, topical steroid
  Forms:            generic and brand names both count (lipitor =
                    atorvastatin; tylenol = acetaminophen)
</what_counts_as_a_treatment>

<what_is_NOT_a_treatment>
  - Diagnostic procedures (x-ray, blood draw, swab, panel) → PROCEDURE
  - Patient education or monitoring ("weigh yourself daily",
    "call me if it gets worse", "follow up in 2 weeks") → omit entirely
  - A measured value (BP 148/90) → LAB_RESULT
  - A condition being treated → DIAGNOSIS or MEDICAL_HISTORY
</what_is_NOT_a_treatment>

<purpose>
Extract every MEDICATION and TREATMENT explicitly mentioned in the
transcript. Pay specific attention to specific drug names (generic and
brand) and to generic treatment categories.
Every emitted item MUST cite turn_id + verbatim quote from the transcript.
</purpose>

<rules>
1. "matched" — use curated labels from the list below.

   PHONETIC AND ORTHOGRAPHIC VARIANTS: Match through phonetic and
   orthographic variants using pharmaceutical knowledge. Transcripts
   often render drug names with mishearings, mis-syllabifications, or
   transcription artefacts; map these back to the canonical curated
   form. Brand-to-generic and generic-to-brand mapping is encouraged
   when the curated form is one or the other.
   - Out-of-corpus examples for grounding only (none of these appear
     in this dataset; use them to calibrate the SHAPE of a phonetic
     match):
     * if curated has `metoprolol` and transcript reads
       "meto-pro-lol" or "metoprolal", emit `metoprolol`.
     * if curated has `clopidogrel` and transcript says
       "plav-icks" (a colloquial rendering of the brand name
       Plavix), emit `clopidogrel`.

   COLLOQUIAL TO CLINICAL: Map common patient terms to curated labels
   using pharmaceutical knowledge — drug-class lay terms, delivery-form
   colloquialisms, and route descriptions all map to their clinical
   curated forms.

2. "other" — TREATMENT entities NOT in the
   <curated_labels> list. Use the transcript words.
3. "fine_grained" — TREATMENT entities finer than
   <curated_labels> (specific dose, route, formulation, frequency, or
   device variant). Use the transcript words.
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
Emit ONE json object and nothing else, with this exact shape:
{{
  "matched":      [{{"label": "...", "turn_id": "...", "quote": "..."}}],
  "other":        [{{"text":  "...", "turn_id": "...", "quote": "..."}}],
  "fine_grained": [{{"text":  "...", "turn_id": "...", "quote": "..."}}]
}}
Any bucket may be []. Every bucket key must be present.
</output_contract>"""


# ─────────────────────────────────────────────────────────────────────────────
# Helper for the simpler per-category prompts (PROCEDURE, LOCATION,
# LAB_RESULT). No content examples — output_contract uses placeholders.
# ─────────────────────────────────────────────────────────────────────────────
def _entity_prompt(
    category: str,
    expert_role: str,
    purpose: str,
    what_counts: str = "",
    what_is_not: str = "",
    tight: bool = False,
) -> str:
    counts_block = f"<what_counts_as_a_{category.lower()}>\n{what_counts}\n</what_counts_as_a_{category.lower()}>\n\n" if what_counts else ""
    not_block = f"<what_is_NOT_a_{category.lower()}>\n{what_is_not}\n</what_is_NOT_a_{category.lower()}>\n\n" if what_is_not else ""
    if tight:
        rules_2_3 = (
            f'2. "other" — {category} entities in the transcript\n'
            f'   that are NOT in <curated_labels>. Use the transcript words.\n'
            f'   When in doubt, emit nothing.\n'
            f'3. "fine_grained" — {category} entities finer-grained\n'
            f'   than entries in <curated_labels>. Use the transcript words.\n'
            f'   When in doubt, emit nothing.'
        )
    else:
        rules_2_3 = (
            f'2. "other" — {category} entities in the transcript\n'
            f'   that are NOT in <curated_labels>. Use the transcript words. Emit\n'
            f'   when the phrase clearly describes a clinical {category} entity,\n'
            f'   even without a curated match. Skip only filler and non-clinical\n'
            f'   content.\n'
            f'3. "fine_grained" — {category} entities finer-grained\n'
            f'   than entries in <curated_labels>. Use the transcript words. Emit\n'
            f'   when the phrase adds clinically meaningful specificity on top of\n'
            f'   a curated concept; skip only filler.'
        )
    return f"""<role>
You are an EXPERT {expert_role} analyzing this doctor-patient transcript
excerpt for the hospital. Your sole responsibility is to extract {category}
entities explicitly stated in the transcript.
</role>

{counts_block}{not_block}<purpose>
{purpose}
</purpose>

<rules>
1. "matched" — curated labels copied verbatim from
   the list below. Bridge synonyms, abbreviations, and clinical
   restatements using your training.
{rules_2_3}
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
Emit ONE json object and nothing else, with this exact shape:
{{{{
  "matched":      [{{{{"label": "...", "turn_id": "...", "quote": "..."}}}}],
  "other":        [{{{{"text":  "...", "turn_id": "...", "quote": "..."}}}}],
  "fine_grained": [{{{{"text":  "...", "turn_id": "...", "quote": "..."}}}}]
}}}}
Any bucket may be []. Every bucket key must be present.
</output_contract>"""


PROCEDURE_PROMPT = _entity_prompt(
    category="PROCEDURE",
    expert_role="clinician",
    purpose=(
        "Extract every diagnostic, evaluative, or interventional PROCEDURE "
        "performed on or ordered for the patient. Procedures include "
        "imaging studies, laboratory specimen collection, physical "
        "examinations, endoscopies, biopsies, point-of-care measurements, "
        "and any intervention performed by the clinician. Include both "
        "performed and pending procedures. Do not include treatments. "
        "PROCEDURE vs LAB_RESULT disambiguation: emit the BARE test or "
        "imaging name (no outcome qualifier) as PROCEDURE — e.g. "
        "`creatinine`, `diabetes panel`, `glucose test`, `lyme titer`, "
        "`endoscopy`, `polypectomy`, `lumbar mri`, `right humerus x-ray`, "
        "`right middle finger x-ray`. If the same phrase carries an "
        "OUTCOME anchor (`unremarkable`, `stable`, `normal`, `elevated`, "
        "a named finding, a numeric value), the compound goes to "
        "LAB_RESULT instead, NOT here. Anatomical-laterality x-rays "
        "(`right middle finger x-ray`, `right lower extremity x-ray`) "
        "with no outcome are PROCEDUREs — preserve laterality verbatim. "
    ),
    what_counts=(
        "Examples by class (illustrative, not exhaustive):\n"
        "  Imaging:           chest x-ray, ct scan, mri, ultrasound,\n"
        "                     echocardiogram, mammogram\n"
        "  Lab specimen:      blood draw, cbc, basic metabolic panel,\n"
        "                     urinalysis, stool sample, swab, biopsy,\n"
        "                     covid swab\n"
        "  Endoscopic:        colonoscopy, endoscopy, bronchoscopy,\n"
        "                     cystoscopy\n"
        "  Cardiac:           ecg, stress test, cardiac catheterization\n"
        "  Physical:          physical exam, vital signs, neurologic exam,\n"
        "                     fundoscopy\n"
        "  Invasive therapy:  surgery, lumbar puncture, paracentesis,\n"
        "                     thoracentesis"
    ),
    what_is_not=(
        "  - The findings/results from the procedure (ejection fraction\n"
        "    45%, hemoglobin 8.2) → LAB_RESULT\n"
        "  - A medication or therapy → TREATMENT\n"
        "  - A diagnosis being investigated → DIAGNOSIS\n"
        "  - A symptom or sign → SYMPTOM"
    ),
)


LOCATION_PROMPT = _entity_prompt(
    category="LOCATION",
    expert_role="clinical anatomist",
    purpose=(
        "Extract every ANATOMICAL location referenced in relation to a "
        "clinical finding, sign, symptom, procedure, or treatment. "
        "Locations must be ANATOMICAL — not institutional, temporal, or "
        "geographic. Include laterality and regional specifiers when "
    ),
    what_counts=(
        "Examples by region (illustrative, not exhaustive):\n"
        "  Head/neck:    head, forehead, temple, scalp, face, neck,\n"
        "                throat, ear, eye, nose, mouth, jaw\n"
        "  Chest/thorax: chest, sternum, ribs, lungs, heart\n"
        "  Abdomen:      abdomen, stomach, liver, gallbladder,\n"
        "                kidneys, pancreas, spleen\n"
        "  Pelvis/GU:    bladder, prostate, ovaries, uterus\n"
        "  Extremities:  arm, leg, hand, foot, ankle, knee, elbow,\n"
        "                wrist, shoulder, hip\n"
        "  Back/spine:   back, lower back, flank, spine\n"
        "  Skin/surface: skin, scalp, behind left knee\n"
        "  Laterality:   right chest, left arm, bilateral ankles"
    ),
    what_is_not=(
        "  - Geographic places (Vermont, Mountains, McDonald's) → omit\n"
        "  - Institutional places (clinic, ER, hospital) → omit\n"
        "  - Temporal references (yesterday, last week) → omit\n"
        "  - Symptoms or findings AT a location → SYMPTOM (the location\n"
        "    becomes a separate LOCATED_AT edge target)"
    ),
    tight=True,
)


LAB_RESULT_PROMPT = _entity_prompt(
    category="LAB_RESULT",
    expert_role="clinical laboratory specialist",
    purpose=(
        "Extract every LAB_RESULT — a named lab test, imaging study, "
        "vital-sign measurement, or a stated result with or without a "
        "numeric value. Include the test name AND/OR the measured value "
        "if stated. Distinguish from PROCEDURE: PROCEDURE is the action "
        "of obtaining the measurement; LAB_RESULT is the test identity "
        "PAIRED with its outcome qualifier or value. CRITICAL: when a "
        "transcript or note says `<imaging-or-test> <outcome>` together "
        "(e.g. `chest x-ray unremarkable`, `lumbar x-ray stable`, `ekg "
        "lvh`, `pft mild asthma copd`, `endoscopy gastritis`, `labs "
        "within normal limits`), THAT compound phrase is a single "
        "LAB_RESULT. Outcome qualifiers like `unremarkable`, `stable`, "
        "`clear margins`, `within normal limits`, `normal`, `negative`, "
        "`positive`, `elevated`, `decreased`, `mild`, `moderate`, "
        "`severe`, named pathology, and named abnormalities all count "
        "as outcome anchors. Do NOT emit the bare test name (that goes "
        "to PROCEDURE) when an outcome anchor is present in the same "
        "phrase. "
    ),
    what_counts=(
        "Examples by class (illustrative, not exhaustive):\n"
        "  Blood work:   hemoglobin 8.2, hba1c 7.2%, bnp elevated,\n"
        "                troponin negative, electrolytes normal,\n"
        "                cbc result, creatinine 1.4, white count 12k\n"
        "  Urine:        urinalysis result, urine ketones positive,\n"
        "                urine culture positive, proteinuria\n"
        "  Stool:        stool guaiac positive, fecal occult blood\n"
        "  Imaging finding: chest x-ray clear, ejection fraction 45%,\n"
        "                   moderate mitral regurgitation,\n"
        "                   ct shows mass, mri unremarkable\n"
        "  Vital signs:  bp 148/90, heart rate 95, oxygen saturation 94%,\n"
        "                temperature 38.5\n"
        "Format note: emit value+anchor as ONE compound entity\n"
        "(`hemoglobin 8.2`, not `hemoglobin` and `8.2` separately).\n"
        "Never emit standalone fragments (`8.2`, `low`, `45%`, `fine`)."
    ),
    what_is_not=(
        "  - The procedure that produced the result (echocardiogram,\n"
        "    cbc, urinalysis as the test action) → PROCEDURE\n"
        "  - The diagnosis the result confirms → DIAGNOSIS\n"
        "  - A standalone value with no measurement anchor → omit\n"
        "    (do not emit just `8.2` or `45%`)"
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# MEDICAL_HISTORY
# ─────────────────────────────────────────────────────────────────────────────
MEDICAL_HISTORY_PROMPT = """<role>
You are an EXPERT clinical internist documenting the patient's background
history for the medical chart. Your job is to distinguish the patient's
pre-existing history — chronic conditions, prior diseases, family history
and genetic predispositions, exposures, triggers, habits, and allergies —
from whatever the clinician is diagnosing at THIS visit.
</role>

<what_counts_as_medical_history>
Examples by class (illustrative, not exhaustive):
  Chronic conditions: diabetes, hypertension, asthma, copd,
                      congestive heart failure, ckd, depression, gerd
  Past surgeries:     cholecystectomy, appendectomy, prior colonoscopy
  Family history:     family history of asthma, family history of
                      lung cancer, family history of heart disease,
                      family history of diabetes
  Exposures:          smoking history, alcohol use, occupational
                      exposure, daycare exposure, school exposure,
                      hiking exposure, sick contact (husband, wife,
                      child, coworker)
  Triggers:           cold air trigger, pollen trigger, exercise
                      trigger, pet trigger, dust trigger
  Allergies:          seasonal allergies, penicillin allergy,
                      no allergies
  Habits/social:      non-smoker, past smoking, vaccinated,
                      immunizations up to date
  Reproductive:       prior pregnancies, contraceptive use
</what_counts_as_medical_history>

<what_is_NOT_medical_history>
  - A condition being actively diagnosed at this visit → DIAGNOSIS
  - A current symptom or sign → SYMPTOM
  - A medication the patient is currently taking for treatment →
    TREATMENT (the underlying condition the medication treats may
    still be MEDICAL_HISTORY)
  - A test or procedure → PROCEDURE
  - A measured value (BP 148/90, hemoglobin 8.2) → LAB_RESULT
  - Geographic places, travel narrative, restaurants → omit
  - Physical exam findings (systolic murmur, ankle edema) → SYMPTOM
</what_is_NOT_medical_history>

<purpose>
For each mention in the transcript, walk through these decisions to decide
whether and how to emit it as a MEDICAL_HISTORY item:

1. BACKGROUND HISTORY vs CURRENT-VISIT DIAGNOSIS?
   - Pre-existing condition the patient has had before today → history
     (EMIT).
   - Chronic disease being managed long-term → history (EMIT).
   - Problem the clinician is diagnosing AT THIS visit → SKIP (goes to
     DIAGNOSIS, not here).
   - When unclear: prefer history if the patient describes it as their
     own pre-existing condition, prefer DIAGNOSIS if the clinician is
     assessing it as today's problem.

2. PERSONAL or FAMILY history?
   - Patient's own condition → emit the condition name (or matching
     curated label).
   - Family member's condition mentioned for genetic / inheritance
     relevance → emit as the curated family-history form when one
     exists. NEVER strip the family-history framing; a relative's
     diagnosis is NOT the patient's diagnosis.

3. EXPOSURES are history items.
   This includes occupational exposures, environmental exposures
   (travel, residential, dietary, animal contact), and social
   contacts (sick household members, daycare or institutional
   exposure, prior bites or stings). If the clinician asks about an
   exposure as a possible cause of illness, emit the exposure —
   whether the answer was yes or no.

   ACTIVITY / CIRCUMSTANCE OF INJURY are history items.
   When the patient links a current symptom to a recent activity —
   recreational sport, household chore, occupational task, accident,
   or physical exertion — the activity itself is MEDICAL_HISTORY
   (the proximate context of injury). Examples by class:
     Recreational:   bowling injury, tennis exertion, volleyball
                     injury, golfing, hiking
     Household:      yard work, moving boxes, moving refrigerator
     Accident:       motor vehicle accident, fall
     Exposure-like:  pollen exposure, dietary indiscretion, stress
   Emit the curated form when present; otherwise emit the transcript
   phrasing as `other`. Do NOT emit the activity as a SYMPTOM.

3b. PRIOR PROCEDURES / PRIOR CONDITIONS are history items.
   Surgeries, procedures, or removals already done in the past
   (NOT today) belong to MEDICAL_HISTORY, not PROCEDURE. Any phrase
   with `prior`, `past`, `previous`, or `history of` in the curated
   list is a strong signal. Examples by class:
     Prior surgeries:  prior knee arthroplasty, prior cataract
                       extraction, prior basal cell carcinoma removal
     Prior workups:    prior lung nodule biopsy, prior colonoscopy
     History framing:  hypertension history, depression history,
                       anxiety history (the chronic-condition form
                       with `history` suffix in the curated label)
   Emit the FULL curated form including the `prior`/`history` framing.

4. TRIGGERS of a chronic condition are history items.
   Environmental, physiologic, dietary, and behavioral triggers all
   count when the patient links them to worsening of an existing
   condition.

5. HABITS / SOCIAL HISTORY.
   Tobacco use (current, former, never), alcohol use, recreational
   substance use, and other relevant lifestyle factors all count.

6. ALLERGIES.
   Drug, food, and environmental allergies all count. Positive
   statements of absence (no known allergies, no drug allergies)
   ALSO count when the curated list contains the absence form.

7. NEGATED STATEMENTS.
   The negated form is clinically meaningful and MUST be preserved
   exactly as the curated list has it. Do NOT strip "absent", "no",
   or "non-" prefixes from curated labels.

8. VACCINES / IMMUNIZATIONS.
   - A specific named vaccine → EXCLUDE (belongs to TREATMENT).
   - General immunization status → INCLUDE here as history.

Do not duplicate today's DIAGNOSIS entries. Every emitted item MUST cite
turn_id + verbatim quote from the transcript.
</purpose>

<rules>
1. "matched" — use curated labels from the list below.
   When the transcript describes a curated history item using a
   clinical synonym, layman variant, or family-history reframing, copy
   the CURATED LABEL verbatim into matched. Cite the transcript
   phrasing in the quote.

2. "other" — MEDICAL_HISTORY entities in the
   transcript that are NOT in <curated_labels>. Use the transcript
   words. Emit when the phrase clearly describes a chronic condition,
   prior illness, exposure, trigger, habit, allergy, or activity
   context, even without a curated match. Skip only filler and
   non-clinical content.
3. "fine_grained" — MEDICAL_HISTORY entities
   finer-grained than entries in <curated_labels> (subtype, age of
   onset, severity descriptor). Use the transcript words. Emit when
   the phrase adds clinically meaningful specificity on top of a
   curated concept; skip only filler.
4. Do NOT assume or infer. Extract ONLY MEDICAL_HISTORY items
   explicitly mentioned in the transcript.
</rules>

<curated_labels>
{curated_labels_bulleted}
</curated_labels>

<transcript>
{transcript}
</transcript>

<output_contract>
Emit ONE json object and nothing else, with this exact shape:
{{
  "matched":      [{{"label": "...", "turn_id": "...", "quote": "..."}}],
  "other":        [{{"text":  "...", "turn_id": "...", "quote": "..."}}],
  "fine_grained": [{{"text":  "...", "turn_id": "...", "quote": "..."}}]
}}
Any bucket may be []. Every bucket key must be present.
</output_contract>"""


# ─────────────────────────────────────────────────────────────────────────────
# Edge prompts — type-pattern lists are clinical reasoning principles, not
# enumerated examples. No content examples in any rule body.
# ─────────────────────────────────────────────────────────────────────────────
RULES_OUT_PROMPT = """<role>
You are an EXPERT clinical internist extracting RULES_OUT edges from a
doctor-patient transcript. You receive the full entity list and the full
transcript. Emit edges in three buckets. </role>

<entities>
{numbered_entities_list}
</entities>

<transcript>
{transcript}
</transcript>

<task>
A RULES_OUT edge means the transcript EXPLICITLY excludes a condition.
Look for exclusion language: negative-for / not-consistent-with /
excludes / ruled-out / came-back-negative.

Valid src → dst type patterns:

  - procedure → diagnosis      (negative test rules out a dx)
  - symptom → diagnosis        (an absent presenting feature excludes
                                 a dx)
  - symptom → medical_history  (an absent feature excludes a chronic
                                 condition under consideration)

Only emit pairs matching one of these type patterns. Arrow direction is
src → dst (evidence of absence → excluded condition). Never reverse.

DO NOT confuse with a working differential — a candidate being evaluated
is NOT a RULES_OUT (that is an INDICATES). Only emit when the transcript
states the exclusion has already happened.

Err toward INCLUSION (recall on matched); precision is handled downstream.

For every eligible (src, dst) pair from <entities>, walk through:

  Step 1 — CANDIDATE WALK. For each DIAGNOSIS D and each eligible src S,
    decide ACCEPT / REJECT / UNCERTAIN for "S rules out D". One line each,
    ≤ 15 words cited from transcript. Do not skip pairs.
  Step 2 — RESOLVE UNCERTAIN as ACCEPT or REJECT.
  Step 3 — TRANSCRIPT WALK (for OTHER). Scan the transcript for
    diagnoses the clinician excluded that are NOT in <entities>. Emit
    src → that free-text dx.
  Step 4 — SUBTYPE CHECK (for FINE_GRAINED). For each ACCEPT pair,
    ask if transcript specifies a subtype of the ruled-out diagnosis.
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
  "matched":      [{{"src": "...", "dst": "...", "turn_id": "...", "quote": "...", "confidence": 0.0}}],
  "other":        [{{"src": "...", "dst": "...", "turn_id": "...", "quote": "...", "confidence": 0.0, "dst_in_candidates": false}}],
  "fine_grained": [{{"src": "...", "coarse_candidate": "...", "dst": "...", "turn_id": "...", "quote": "...", "confidence": 0.0}}]
}}
Any bucket may be []. Every bucket key must be present.
</output>
"""


LOCATED_AT_PROMPT = """<role>
You are an EXPERT clinical anatomist extracting LOCATED_AT edges from a
doctor-patient transcript. You receive the full entity list and the full
transcript. Emit edges in three buckets. </role>

<entities>
{numbered_entities_list}
</entities>

<transcript>
{transcript}
</transcript>

<task>
A LOCATED_AT edge means a clinical finding is anchored to an anatomical
location. Institutional and geographic locations are NOT valid dst.

Valid src → dst type patterns:

  - symptom → location           (a finding anchored to anatomy)
  - procedure → location         (a procedure performed on a body region)
  - diagnosis → location         (a diagnosis anchored at a body region)
  - medical_history → location   (a chronic condition anchored at a body
                                   region — organ-system level OK)

Only emit pairs matching one of these type patterns. dst must be
ANATOMICAL. Include laterality, regional specifiers, and organ-level
detail when stated. Arrow direction is src → dst. Never reverse.

Err toward INCLUSION (recall on matched); precision is handled downstream.

For every eligible (src, dst) pair from <entities>, walk through:

  Step 1 — CANDIDATE WALK. For each LOCATION L (anatomical only) and
    each eligible src S, decide ACCEPT / REJECT / UNCERTAIN for "S is
    located at L". One line each, ≤ 15 words cited from transcript.
  Step 2 — RESOLVE UNCERTAIN as ACCEPT or REJECT.
  Step 3 — TRANSCRIPT WALK (for OTHER). Scan for anatomical locations
    the clinician named that are NOT in <entities>. Emit src → that
    free-text location.
  Step 4 — SUBTYPE CHECK (for FINE_GRAINED). For each ACCEPT pair,
    ask if transcript specifies a tighter anatomical site than the
    <entities> entry.
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
  "matched":      [{{"src": "...", "dst": "...", "turn_id": "...", "quote": "...", "confidence": 0.0}}],
  "other":        [{{"src": "...", "dst": "...", "turn_id": "...", "quote": "...", "confidence": 0.0, "dst_in_candidates": false}}],
  "fine_grained": [{{"src": "...", "coarse_candidate": "...", "dst": "...", "turn_id": "...", "quote": "...", "confidence": 0.0}}]
}}
Any bucket may be []. Every bucket key must be present.
</output>
"""


TAKEN_FOR_PROMPT = """<role>
You are an EXPERT clinical pharmacist extracting TAKEN_FOR edges from a
doctor-patient transcript. You receive the full entity list and the full
transcript. Emit edges in three buckets. </role>

<entities>
{numbered_entities_list}
</entities>

<transcript>
{transcript}
</transcript>

<task>
A TAKEN_FOR edge means the treatment is prescribed, taken, or used FOR
a target condition or symptom. Look for intent language: for / to-treat /
prescribed-for / helps-with / manages / to-control / indicated-for.

Valid src → dst type patterns:

  - treatment → medical_history  (chronic condition being managed —
                                   THIS IS THE MOST COMMON PATTERN; do
                                   not skip it)
  - treatment → diagnosis        (active dx being treated today)
  - treatment → symptom          (symptom relief)

Only emit pairs matching one of these type patterns. Arrow direction is
src → dst. Never reverse.

Mere co-mention is not enough — the transcript must establish the
purpose linkage. Drug-induced symptoms (a treatment that CAUSED a
symptom as side effect) go to CAUSES, not TAKEN_FOR.

Out-of-corpus example for grounding only (does not appear in this
dataset; use it to calibrate the SHAPE of a TAKEN_FOR edge):
  - if entities contain TREATMENT `furosemide` and SYMPTOM
    `peripheral edema`, and the transcript indicates the diuretic
    is being used to manage the swelling, emit the edge
    `furosemide → peripheral edema`.

Err toward INCLUSION (recall on matched); precision is handled downstream.

For every (src=TREATMENT, dst=MEDICAL_HISTORY/DIAGNOSIS/SYMPTOM) pair
from <entities>, walk through:

  Step 1 — CANDIDATE WALK. For each TREATMENT T and each eligible dst D,
    decide ACCEPT / REJECT / UNCERTAIN for "T is taken for D". One line
    each, ≤ 15 words cited from transcript.
  Step 2 — RESOLVE UNCERTAIN as ACCEPT or REJECT.
  Step 3 — TRANSCRIPT WALK (for OTHER). Scan for treatments or target
    conditions the clinician named that are NOT in <entities>.
  Step 4 — SUBTYPE CHECK (for FINE_GRAINED). For each ACCEPT pair, ask
    if transcript specifies a subtype of the treatment or the target.
  Step 5 — ANTI-COLLAPSE. One treatment may be prescribed for multiple
    targets; include each. One target may have multiple treatments;
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
  "matched":      [{{"src": "...", "dst": "...", "turn_id": "...", "quote": "...", "confidence": 0.0}}],
  "other":        [{{"src": "...", "dst": "...", "turn_id": "...", "quote": "...", "confidence": 0.0, "dst_in_candidates": false}}],
  "fine_grained": [{{"src": "...", "coarse_candidate": "...", "dst": "...", "turn_id": "...", "quote": "...", "confidence": 0.0}}]
}}
Any bucket may be []. Every bucket key must be present.
</output>
"""


INDICATES_PROMPT = """<role>
You are an EXPERT clinical internist extracting INDICATES edges from a
doctor-patient transcript. You receive the full entity list and the full
transcript. Emit edges in three buckets. </role>

<entities>
{numbered_entities_list}
</entities>

<transcript>
{transcript}
</transcript>

<task>
An INDICATES edge means evidence suggests / points to / is consistent
with a condition. Recommended src → dst type patterns:

  - symptom → diagnosis            (most common — a presenting feature
                                     that the clinician treats as
                                     evidence for a working hypothesis)
  - symptom → medical_history      (a symptom pointing to a chronic
                                     condition under consideration)
  - procedure → diagnosis          (a test ordered to evaluate a dx)
  - medical_history → diagnosis    (exposure / family history pointing
                                     to a current dx)
  - lab_result → diagnosis
  - exposure / trigger → diagnosis

These patterns are guidance — emit other plausible clinical relationships
that fit the edge meaning.

Out-of-corpus example for grounding only (does not appear in this
dataset; use it to calibrate the SHAPE of an INDICATES edge):
  - if entities contain SYMPTOM `polyuria` and DIAGNOSIS
    `diabetes mellitus`, and the transcript context links them
    clinically, emit the edge `polyuria → diabetes mellitus`.

Err toward INCLUSION (recall on matched); precision is handled downstream.

For every eligible (src, dst) pair from <entities>, walk through:

  Step 1 — CANDIDATE WALK. For each DIAGNOSIS D and each evidence
    entity E, decide ACCEPT / REJECT / UNCERTAIN for "E supports D".
    One line each, ≤ 15 words cited from transcript. Do not skip pairs.
  Step 2 — RESOLVE UNCERTAIN as ACCEPT or REJECT.
  Step 3 — TRANSCRIPT WALK (for OTHER). Scan for diagnoses the
    clinician named or implied that are NOT in <entities>. Emit
    evidence → that free-text dx.
  Step 4 — SUBTYPE CHECK (for FINE_GRAINED). For each ACCEPT pair,
    ask if the transcript specifies a subtype of the diagnosis.
  Step 5 — ANTI-COLLAPSE. Specific AND generic diagnoses can both
    apply; include both.
  Step 6 — VERIFY LABELS. MATCHED endpoints are character-for-character
    copies of text values from <entities>.

Three buckets:
  MATCHED      — both src and dst are <entities> entries.
  OTHER        — at least one endpoint is free-text, not in <entities>
                 but named or implied in the transcript.
  FINE_GRAINED — src is an <entities> entry, dst is a free-text subtype.

NEGATION BLOCKS EMISSION: if the src is denied/absent, do not emit any
edge from it in any bucket.

DEDUPLICATE within each bucket.
</task>

<output>
Emit ONE JSON object, nothing else:
{{
  "rationale": "<=80 words summarizing coverage",
  "matched":      [{{"src": "...", "dst": "...", "turn_id": "...", "quote": "...", "confidence": 0.0}}],
  "other":        [{{"src": "...", "dst": "...", "turn_id": "...", "quote": "...", "confidence": 0.0, "dst_in_candidates": false}}],
  "fine_grained": [{{"src": "...", "coarse_candidate": "...", "dst": "...", "turn_id": "...", "quote": "...", "confidence": 0.0}}]
}}
Any bucket may be []. Every bucket key must be present.
</output>
"""


CAUSES_PROMPT = """<role>
You are an EXPERT clinical internist extracting CAUSES edges from a
doctor-patient transcript. You receive the full entity list and the full
transcript. Emit edges in three buckets. </role>

<entities>
{numbered_entities_list}
</entities>

<transcript>
{transcript}
</transcript>

<task>
A CAUSES edge means the transcript states or clearly implies etiology /
mechanism — look for verbs of causation: caused / due-to / brought-on-by /
resulted-from / led-to / triggered / from. Co-occurrence is NOT enough.

Valid src → dst type patterns:

  - medical_history → diagnosis       (exposure causes the current dx)
  - medical_history → medical_history (a long-standing risk factor
                                        causes a chronic condition;
                                        familial → inherited)
  - medical_history → symptom         (a trigger or missed dose causes
                                        a symptom)
  - symptom → symptom                 (one symptom produces another)
  - diagnosis → diagnosis              (rare — an upstream dx causes an
                                        exacerbation)
  - treatment → symptom               (drug side effect)

Only emit pairs matching one of these type patterns. Arrow direction is
src → dst (upstream trigger → downstream condition). Never reverse.

DO NOT emit diagnosis → symptom. Symptoms are evidence INDICATING the
diagnosis in this schema, not effects of it.

Err toward INCLUSION (recall on matched); precision is handled downstream.

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
  "matched":      [{{"src": "...", "dst": "...", "turn_id": "...", "quote": "...", "confidence": 0.0}}],
  "other":        [{{"src": "...", "dst": "...", "turn_id": "...", "quote": "...", "confidence": 0.0, "dst_in_candidates": false}}],
  "fine_grained": [{{"src": "...", "coarse_candidate": "...", "dst": "...", "turn_id": "...", "quote": "...", "confidence": 0.0}}]
}}
Any bucket may be []. Every bucket key must be present.
</output>
"""


CONFIRMS_PROMPT = """<role>
You are an EXPERT clinical internist extracting CONFIRMS edges from a
doctor-patient transcript. You receive the full entity list and the full
transcript. Emit edges in three buckets. </role>

<entities>
{numbered_entities_list}
</entities>

<transcript>
{transcript}
</transcript>

<task>
A CONFIRMS edge means a test or measurement established a condition.
Definitive verb required: confirms / diagnostic-of / positive-for /
demonstrates / shows / established-by / consistent-on-imaging.

Valid src → dst type patterns:

  - lab_result → diagnosis    (a measured value or named test result
                                established the diagnosis)
  - lab_result → symptom      (a measured value established a symptom)
  - procedure → diagnosis     (a procedure result established the dx)

Only emit pairs matching one of these type patterns. Arrow direction is
src → dst. Never reverse.

Never emit from patient self-report. A planned future test is NOT a
CONFIRMS — only emit when the result is in the transcript.

Err toward INCLUSION (recall on matched); precision is handled downstream.

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

NEGATION BLOCKS EMISSION: if the dst is denied — that is a RULES_OUT
edge, not a CONFIRMS edge. Do not emit in this bucket.

DEDUPLICATE within each bucket.
</task>

<output>
Emit ONE JSON object, nothing else:
{{
  "rationale": "<=80 words summarizing coverage",
  "matched":      [{{"src": "...", "dst": "...", "turn_id": "...", "quote": "...", "confidence": 0.0}}],
  "other":        [{{"src": "...", "dst": "...", "turn_id": "...", "quote": "...", "confidence": 0.0, "dst_in_candidates": false}}],
  "fine_grained": [{{"src": "...", "coarse_candidate": "...", "dst": "...", "turn_id": "...", "quote": "...", "confidence": 0.0}}]
}}
Any bucket may be []. Every bucket key must be present.
</output>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Drop-in registry — same shape as v7.CATEGORY_PROMPTS so the v9 runtime
# can swap by reassigning v8.CATEGORY_PROMPTS to this dict.
# ─────────────────────────────────────────────────────────────────────────────
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
