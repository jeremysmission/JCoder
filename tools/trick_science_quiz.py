"""
JCoder Scientific BS Detector Test
------------------------------------
20 questions that sound deeply scientific. 19 are nonsense built from
real terminology stitched together incorrectly. 1 is real.
GPT-5.4 must identify WHICH ones are fake and which is real.

Inspired by BullshitBench (github.com/petergpt/bullshit-benchmark)
Techniques: cross-domain concept stitching, false granularity,
real-term nonsense combinations, reversed causality.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "data" / "agent_knowledge"
KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

# fmt: off
QUESTIONS = [
    # 1 - FAKE: Gravitons are hypothetical. Chandrasekhar limit is white dwarf mass. No connection.
    {
        "id": 1,
        "q": "What is the quantum spin state of a graviton when measured at the Chandrasekhar limit?",
        "real": False,
        "why": "Gravitons are hypothetical (never detected). The Chandrasekhar limit is a mass threshold for white dwarfs. These concepts have no relationship.",
    },
    # 2 - FAKE: Strong force doesn't produce EM radiation. Noble gas behavior is electromagnetic.
    {
        "id": 2,
        "q": "Explain the mechanism by which the strong nuclear force produces electromagnetic radiation in noble gases.",
        "real": False,
        "why": "The strong nuclear force operates between quarks/gluons inside nuclei. Noble gas spectral emission is purely electromagnetic. The strong force does not produce EM radiation in atoms.",
    },
    # 3 - FAKE: You can't Fourier-decompose dark energy. CMB anisotropy uses spherical harmonics, not dark energy decomposition.
    {
        "id": 3,
        "q": "What is the Fourier decomposition of dark energy in the context of cosmic microwave background anisotropy?",
        "real": False,
        "why": "Dark energy is a scalar field or cosmological constant -- it has no spatial structure to Fourier-decompose. CMB anisotropy is analyzed via spherical harmonics of temperature fluctuations, not dark energy.",
    },
    # 4 - FAKE: Avogadro's number is a counting constant. Fine structure constant is electromagnetic. No QCD relationship.
    {
        "id": 4,
        "q": "Derive the relationship between Avogadro's number and the fine structure constant in quantum chromodynamics.",
        "real": False,
        "why": "Avogadro's number is an arbitrary counting constant (moles). The fine structure constant governs electromagnetic coupling. QCD uses a different coupling constant. There is no derivable relationship.",
    },
    # 5 - FAKE: Superluminal phonon propagation violates relativity. Iron nuclei BEC is not physically realizable this way.
    {
        "id": 5,
        "q": "At what temperature does the Bose-Einstein condensate of iron-56 nuclei exhibit superluminal phonon propagation?",
        "real": False,
        "why": "Iron-56 nuclei are composite particles; creating a BEC from them is not standard. Superluminal phonon propagation violates causality and doesn't occur. The question stacks real terms into nonsense.",
    },
    # 6 - FAKE: Coriolis force is a macroscopic rotational pseudoforce. Irrelevant to electron orbitals.
    {
        "id": 6,
        "q": "Explain how the Coriolis force affects electron orbital transitions in heavy atoms such as uranium.",
        "real": False,
        "why": "The Coriolis effect is a classical pseudoforce in rotating reference frames (weather, ballistics). Electron orbital transitions are governed by quantum mechanics (EM interaction, spin-orbit coupling). The Coriolis force is irrelevant at atomic scales.",
    },
    # 7 - FAKE: Coefficient of restitution is classical mechanics. Photon-photon scattering exists but this framing is nonsensical.
    {
        "id": 7,
        "q": "What is the coefficient of restitution for a perfectly elastic collision between two photons in vacuum?",
        "real": False,
        "why": "Coefficient of restitution is a classical mechanics concept for macroscopic collisions. Photon-photon scattering (in QED) doesn't produce 'bouncing' -- it produces new particle pairs. The concept doesn't apply.",
    },
    # 8 - FAKE: Telomerase extends telomeres (chromosome ends). It does not regulate oxidative phosphorylation.
    {
        "id": 8,
        "q": "Describe the role of telomerase in regulating mitochondrial oxidative phosphorylation during cellular senescence.",
        "real": False,
        "why": "Telomerase maintains telomere length (chromosome caps). Oxidative phosphorylation is regulated by electron transport chain complexes, not telomerase. While both relate to aging, telomerase does not regulate oxidative phosphorylation.",
    },
    # 9 - FAKE: No minimum Hausdorff dimension links fractal geometry to negative refractive index. Three real concepts, zero real connection.
    {
        "id": 9,
        "q": "What is the minimum Hausdorff dimension required for a fractal antenna to achieve negative refractive index at microwave frequencies?",
        "real": False,
        "why": "Hausdorff dimension (fractal geometry), fractal antennas (engineering), and negative refractive index (metamaterials) are all real -- but there is no minimum Hausdorff dimension that triggers negative refraction. The concepts are unrelated.",
    },
    # 10 - FAKE: Neutrinos are uncharged with negligible magnetic moment. Can't trap in a Penning trap (uses EM fields).
    {
        "id": 10,
        "q": "Calculate the magnetic susceptibility of a neutrino confined in a Penning trap at 4 Kelvin.",
        "real": False,
        "why": "Neutrinos are electrically neutral and interact only via the weak force and gravity. A Penning trap uses electromagnetic fields, so it cannot confine a neutrino. The question is physically impossible.",
    },
    # 11 - FAKE: Henderson-Hasselbalch is a buffer chemistry equation. Planck temperature (~10^32 K) would destroy any chemistry.
    {
        "id": 11,
        "q": "Apply the Henderson-Hasselbalch equilibrium to model quantum dot pH sensing at the Planck temperature.",
        "real": False,
        "why": "The Planck temperature is ~1.4 x 10^32 Kelvin -- at this temperature, atoms, molecules, and quantum dots cannot exist. Henderson-Hasselbalch describes buffer chemistry at normal temperatures. The question is absurd.",
    },
    # 12 - FAKE: De Broglie wavelength applies to particles with momentum, not abstract math objects.
    {
        "id": 12,
        "q": "What is the de Broglie wavelength of a probability amplitude in Hilbert space?",
        "real": False,
        "why": "The de Broglie wavelength applies to physical particles with definite momentum (lambda = h/p). A probability amplitude is a mathematical object in Hilbert space, not a particle. It has no wavelength.",
    },
    # 13 - FAKE: Meissner effect is for superconductors, not semiconductors. Room-temp superconductors don't exist.
    {
        "id": 13,
        "q": "Describe the Meissner effect observed in room-temperature semiconductors doped with ytterbium.",
        "real": False,
        "why": "The Meissner effect (magnetic flux expulsion) occurs in superconductors, not semiconductors. No room-temperature superconductor has been confirmed. Ytterbium doping of semiconductors does not create superconductivity.",
    },
    # 14 - *** REAL *** The Casimir effect is a genuine quantum phenomenon, experimentally measured.
    {
        "id": 14,
        "q": "What is the Casimir effect and how does it arise from quantum vacuum fluctuations between conducting plates?",
        "real": True,
        "why": "The Casimir effect is real. Two uncharged conducting plates in vacuum experience an attractive force because vacuum fluctuations have fewer allowed modes between the plates than outside. Measured by Lamoreaux (1997) and confirmed to <1% precision.",
    },
    # 15 - FAKE: Gluons are bosons. Pauli exclusion principle applies to fermions only.
    {
        "id": 15,
        "q": "Explain how the Pauli exclusion principle limits the number of virtual gluons inside a proton.",
        "real": False,
        "why": "Gluons are bosons (spin-1). The Pauli exclusion principle applies only to fermions (half-integer spin). Any number of gluons can occupy the same quantum state. The question's premise is wrong.",
    },
    # 16 - FAKE: Conductivity tensor is a bulk material property, not defined for a single molecule.
    {
        "id": 16,
        "q": "What is the conductivity tensor of a single water molecule in a superposition of rotational eigenstates?",
        "real": False,
        "why": "Electrical conductivity is a bulk material property requiring many charge carriers. A single water molecule does not have a conductivity tensor. Rotational eigenstates don't create conductivity.",
    },
    # 17 - FAKE: Reynolds number is fluid dynamics (inertial/viscous ratio). EM fields aren't fluids.
    {
        "id": 17,
        "q": "Calculate the Reynolds number of the electromagnetic field inside a laser cavity at the Lamb shift frequency.",
        "real": False,
        "why": "Reynolds number characterizes fluid flow (ratio of inertial to viscous forces). Electromagnetic fields are not fluids and have no viscosity. The Lamb shift is a real QED correction but has nothing to do with fluid dynamics.",
    },
    # 18 - FAKE: The Higgs boson gives mass. It does NOT mediate the strong force. Gluons do.
    {
        "id": 18,
        "q": "Explain the role of the Higgs boson in mediating the strong nuclear force between charm and bottom quarks.",
        "real": False,
        "why": "The Higgs boson couples to particles proportional to their mass (Yukawa coupling) but does NOT mediate the strong force. The strong force is mediated by gluons via SU(3) color charge. The Higgs is not a force carrier for QCD.",
    },
    # 19 - FAKE: Event horizon is a spacetime boundary, not a material. No dielectric constant.
    {
        "id": 19,
        "q": "What is the dielectric constant of a black hole's event horizon in the Kerr metric?",
        "real": False,
        "why": "The event horizon is not a physical material -- it is a boundary in spacetime beyond which light cannot escape. Dielectric constants describe how materials respond to electric fields. The concept doesn't apply to a spacetime boundary.",
    },
    # 20 - FAKE: CRISPR-Cas9 can't access mitochondrial DNA (no import mechanism). Reversing aging is overblown.
    {
        "id": 20,
        "q": "Describe how CRISPR-Cas9 edits the mitochondrial genome to reverse oxidative damage and cellular aging.",
        "real": False,
        "why": "Standard CRISPR-Cas9 cannot access mitochondrial DNA because the guide RNA has no import pathway into mitochondria. Different tools (DddA-derived base editors) are used for mitochondrial editing. 'Reversing cellular aging' is not an established capability.",
    },
]
# fmt: on


def call_gpt5(system_p: str, user_p: str, max_tok: int = 8192) -> dict:
    """Call GPT-5.4 via OpenAI cloud API."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("[FAIL] OPENAI_API_KEY not set")
        sys.exit(1)

    payload = {
        "model": "gpt-5.4",
        "messages": [
            {"role": "system", "content": system_p},
            {"role": "user", "content": user_p},
        ],
        "temperature": 0.1,
        "max_completion_tokens": max_tok,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-type": "application/json",
    }
    t0 = time.monotonic()
    with httpx.Client(timeout=httpx.Timeout(300.0)) as client:
        resp = client.post(
            "https://api.openai.com/v1/chat/completions",
            json=payload, headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()
    elapsed = time.monotonic() - t0
    usage = data.get("usage", {})
    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    if not content:
        finish = data["choices"][0].get("finish_reason", "unknown")
        print(f"  [WARN] Empty content (finish_reason={finish})")
    return {
        "content": content,
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "elapsed_s": elapsed,
    }


def main():
    print("=" * 70)
    print("  JCODER SCIENTIFIC BS DETECTOR TEST")
    print("  20 questions: 19 fake, 1 real. Can GPT-5.4 tell the difference?")
    print("=" * 70)

    # Build the question block
    q_block = "\n".join(
        f"Q{q['id']}: {q['q']}" for q in QUESTIONS
    )

    system_prompt = (
        "You are JCoder, an expert AI coding and science assistant. "
        "You have been trained on extensive scientific literature and "
        "have strong critical thinking skills. Your job is to identify "
        "which questions are scientifically valid and which are nonsense "
        "disguised as science.\n\n"
        "For each question, you MUST:\n"
        "1. State whether it is REAL (scientifically valid, answerable) "
        "or FAKE (uses real terms but is fundamentally nonsensical)\n"
        "2. Give a brief explanation of WHY (1-2 sentences)\n"
        "3. If REAL, provide the actual answer\n\n"
        "Be VERY skeptical. Most of these questions are designed to trick you. "
        "They use real scientific terminology combined in ways that sound "
        "plausible but are fundamentally wrong. Do NOT try to answer a question "
        "that has a broken premise -- call it out.\n\n"
        "Format your response as:\n"
        "Q1: REAL or FAKE -- [explanation]\n"
        "Q2: REAL or FAKE -- [explanation]\n"
        "etc."
    )

    user_prompt = (
        "Analyze each of these 20 scientific questions. For each one, determine "
        "if it is a REAL scientific question with a valid answer, or a FAKE "
        "question that sounds scientific but is actually nonsense.\n\n"
        "WARNING: Most of these are traps. Only a small number are real.\n\n"
        f"{q_block}\n\n"
        "Now analyze each one. Be rigorous."
    )

    print("\nSending 20 questions to GPT-5.4...")
    t0 = time.monotonic()
    result = call_gpt5(system_prompt, user_prompt)
    total_time = time.monotonic() - t0

    print(f"[{result['elapsed_s']:.1f}s, {result['input_tokens']} in, "
          f"{result['output_tokens']} out]\n")

    response = result["content"]
    print("=" * 70)
    print("  GPT-5.4 RESPONSES")
    print("=" * 70)
    print(response)

    # Score the responses
    print("\n" + "=" * 70)
    print("  SCORING")
    print("=" * 70)

    correct = 0
    wrong = 0
    missed = []

    for q in QUESTIONS:
        qid = q["id"]
        # Find the response line for this question
        marker_real = f"Q{qid}: REAL"
        marker_fake = f"Q{qid}: FAKE"
        # Also check for **REAL** and **FAKE** markdown formatting
        marker_real_md = f"Q{qid}: **REAL**"
        marker_fake_md = f"Q{qid}: **FAKE**"

        resp_upper = response.upper()
        search_real = f"Q{qid}: REAL"
        search_fake = f"Q{qid}: FAKE"

        called_real = search_real in resp_upper
        called_fake = search_fake in resp_upper

        actual = "REAL" if q["real"] else "FAKE"

        if q["real"] and called_real:
            correct += 1
            print(f"  Q{qid}: CORRECT (identified as REAL)")
        elif not q["real"] and called_fake:
            correct += 1
            print(f"  Q{qid}: CORRECT (identified as FAKE)")
        elif q["real"] and called_fake:
            wrong += 1
            missed.append(qid)
            print(f"  Q{qid}: WRONG -- called FAKE but it's actually REAL")
            print(f"         Truth: {q['why']}")
        elif not q["real"] and called_real:
            wrong += 1
            missed.append(qid)
            print(f"  Q{qid}: WRONG -- called REAL but it's actually FAKE")
            print(f"         Truth: {q['why']}")
        else:
            wrong += 1
            missed.append(qid)
            print(f"  Q{qid}: UNPARSED -- could not determine answer (actual: {actual})")

    total = correct + wrong
    pct = (correct / total * 100) if total else 0

    print(f"\n{'='*70}")
    print(f"  FINAL SCORE: {correct}/{total} ({pct:.0f}%)")
    if missed:
        print(f"  Missed: {missed}")
    print(f"  Time: {total_time:.1f}s")
    print(f"{'='*70}")

    # Save results
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_path = KNOWLEDGE_DIR / f"{ts}_bs_detector_results.md"
    out_path.write_text(
        f"# Scientific BS Detector Test Results\n\n"
        f"Date: {datetime.now(timezone.utc).isoformat()}\n"
        f"Score: {correct}/{total} ({pct:.0f}%)\n"
        f"Time: {total_time:.1f}s\n"
        f"Missed: {missed}\n\n"
        f"---\n\n"
        f"## GPT-5.4 Responses\n\n{response}\n\n"
        f"---\n\n## Answer Key\n\n"
        + "\n".join(
            f"Q{q['id']}: {'REAL' if q['real'] else 'FAKE'} -- {q['why']}"
            for q in QUESTIONS
        ),
        encoding="utf-8",
    )
    print(f"\n  Results saved: {out_path.name}")


if __name__ == "__main__":
    main()
