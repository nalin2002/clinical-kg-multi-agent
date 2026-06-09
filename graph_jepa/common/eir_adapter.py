"""Adapter to the existing EIR 13-agent LLM extractor (``EIR_260426/``).

We do NOT reimplement EIR's Stage-1 (7 entity agents) / Stage-2 (6 edge agents)
fan-out. This module is a thin integration wrapper that:

1. finds already-extracted EIR per-patient KGs on disk and converts them into the
   canonical :class:`Graph` schema (the cheap, default path — no API calls), and
2. optionally shells out to the EIR entry script to extract fresh KGs when
   ``eir.run_eir: true`` (best-effort; EIR is a heavy batch pipeline that needs
   an OpenRouter key — see graph_jepa/common/README notes).

EIR emits per-patient JSON shaped like
``{"nodes":[{"id","text","type"}], "edges":[{"source_id","target_id","type"}]}``
with filenames such as ``RESD2N001_curated.json`` /
``RES0198_cooperative_multi_agent.json``; :func:`match_patient` maps those to
patient folder ids like ``RES_D2N001``.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from .graph_schema import Graph
from .io_utils import LOG, Config, Patient, read_json

_ALNUM = re.compile(r"[^A-Z0-9]")


def _key(s: str) -> str:
    """Uppercase, strip non-alphanumerics: ``RES_D2N001`` -> ``RESD2N001``."""
    return _ALNUM.sub("", s.upper())


def match_patient(filename_stem: str, patient_ids: List[str]) -> Optional[str]:
    """Best-effort map an EIR output filename stem to a patient id."""
    fkey = _key(filename_stem)
    best = None
    for pid in patient_ids:
        pkey = _key(pid)
        if pkey and pkey in fkey:
            # Prefer the longest patient key that matches (most specific).
            if best is None or len(_key(best)) < len(pkey):
                best = pid
    return best


def find_existing_eir_graphs(cfg: Config, patient_ids: List[str]) -> Dict[str, Graph]:
    """Scan ``eir.existing_kg_dirs`` for per-patient EIR KGs and convert them."""
    out: Dict[str, Graph] = {}
    for rel in cfg.get("eir.existing_kg_dirs", []) or []:
        root = cfg.resolve(rel)
        if not root.exists():
            continue
        for jf in sorted(root.rglob("*.json")):
            if any(jf.name.endswith(s) for s in ("_meta.json", "_stats.json", "_usage.json")):
                continue
            pid = match_patient(jf.stem, patient_ids)
            if pid is None or pid in out:
                continue
            try:
                d = read_json(jf)
                if "nodes" not in d:
                    continue
                out[pid] = Graph.from_eir_dict(d, patient_id=pid, source="llm_transcript")
            except Exception as exc:  # noqa: BLE001
                LOG.warning("EIR graph %s unreadable: %s", jf.name, exc)
        LOG.info("found %d EIR graphs under %s", len(out), root)
    return out


def run_eir_extraction(cfg: Config, patient_ids: List[str], out_dir: Path) -> Dict[str, Graph]:
    """Best-effort: invoke the EIR entry script as a subprocess to produce KGs.

    This honors "wrap, don't rewrite". Because EIR is a batch pipeline tied to
    its own transcript dir and an OpenRouter key, the recommended workflow is to
    run EIR yourself and point ``eir.existing_kg_dirs`` at its output. This hook
    exists for convenience and fails loudly with guidance if the script's
    interface doesn't accept our arguments.
    """
    eir_root = cfg.resolve(cfg.get("eir.root", "EIR_260426"))
    entry = eir_root / cfg.get("eir.entry_script", "smoke_test_v10_aci_bench_260425.py")
    if not entry.exists():
        LOG.error("EIR entry script not found: %s", entry)
        return {}
    res_ids = [p for p in patient_ids]
    cmd = [sys.executable, str(entry), "--output", str(out_dir),
           "--workers", str(cfg.get("eir.workers", 3)), "--no-score"]
    if res_ids:
        cmd += ["--res-ids", *res_ids]
    LOG.info("running EIR: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, cwd=str(eir_root), check=True)
    except (subprocess.CalledProcessError, OSError) as exc:
        LOG.error("EIR subprocess failed (%s). Run EIR manually and set "
                  "eir.existing_kg_dirs to its output dir.", exc)
        return {}
    # Collect whatever per-patient KGs the run produced.
    produced: Dict[str, Graph] = {}
    for jf in sorted(Path(out_dir).rglob("*.json")):
        pid = match_patient(jf.stem, patient_ids)
        if pid and pid not in produced:
            try:
                produced[pid] = Graph.from_eir_dict(read_json(jf), patient_id=pid)
            except Exception:  # noqa: BLE001
                continue
    return produced


def get_llm_graphs(cfg: Config, patients: List[Patient], out_dir: Path) -> Dict[str, Graph]:
    """Resolve LLM-extracted graphs for the given patients.

    Order: existing EIR outputs first; if ``eir.run_eir`` is true, fill gaps by
    running EIR. Patients with no available graph are skipped (logged), so a
    missing extraction never crashes the pipeline.
    """
    patient_ids = [p.patient_id for p in patients]
    graphs = find_existing_eir_graphs(cfg, patient_ids)

    missing = [pid for pid in patient_ids if pid not in graphs]
    if missing and cfg.get("eir.run_eir", False):
        LOG.info("running EIR for %d patients without existing graphs", len(missing))
        graphs.update(run_eir_extraction(cfg, missing, out_dir))

    still_missing = [pid for pid in patient_ids if pid not in graphs]
    if still_missing:
        LOG.warning("no LLM graph for %d/%d patients (e.g. %s). Run EIR and point "
                    "eir.existing_kg_dirs at its output to cover them.",
                    len(still_missing), len(patient_ids), still_missing[:3])
    return graphs
