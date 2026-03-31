"""Curated benchmark prompts for MoE routing diversity analysis.

Each category targets a distinct semantic domain to exercise different expert
specializations in MoE models.  Prompts are length-stratified: ``_short``
(~50-80 tokens), base (~120-200 tokens), and ``_long`` (300+ tokens) variants
exist for key domains, enabling routing comparison across context lengths
without changing domain content.

Routing diversity benchmarking dimensions:
  1. **Domain** — code, science, legal, etc. exercise different vocabulary
     distributions and expert specializations.
  2. **Length** — short vs long prompts test whether routing patterns change
     as context grows (early layers tend to be length-sensitive).
  3. **Structure** — repetitive vs diverse vocabulary tests routing collapse
     under low-entropy inputs.

Usage::

    from soda.moe.prompts import (
        MOE_BENCHMARK_PROMPTS,
        ROUTING_SWEEP_CONFIGS,
        build_sweep_matrix,
    )

    # All prompts, organized by domain
    for name, text in MOE_BENCHMARK_PROMPTS.items():
        tokenized = tokenizer(text, ...)

    # Sweep across seq_len × domain
    for config in build_sweep_matrix():
        run_routing_capture(config["prompt_name"], config["seq_len"], ...)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Prompt corpus
# ---------------------------------------------------------------------------

MOE_BENCHMARK_PROMPTS: Dict[str, str] = {

    # ── Code ───────────────────────────────────────────────────────────
    "code_python_short": (
        "Write a Python function to check if a string is a palindrome. "
        "Handle edge cases like empty strings and single characters."
    ),
    "code_python": (
        "Write a Python function that implements a least-recently-used (LRU) "
        "cache using an OrderedDict. The cache should support get and put "
        "operations in O(1) time. Include type hints, docstrings, and edge "
        "case handling for capacity zero. Then write unit tests using pytest "
        "that cover insertion, eviction, and capacity-one behavior."
    ),
    "code_python_long": (
        "Design and implement a complete Python library for a task scheduler "
        "that supports priority queues, delayed execution, and periodic tasks. "
        "The scheduler should use a min-heap for O(log n) insertion and "
        "extraction, support cancellation via task IDs, and handle concurrent "
        "task submission from multiple threads using a reader-writer lock. "
        "Implement a decorator @scheduled(interval='5m') that registers "
        "functions as periodic tasks. Include comprehensive error handling: "
        "tasks that raise exceptions should be retried up to 3 times with "
        "exponential backoff, then moved to a dead-letter queue. Write a "
        "full test suite covering: basic scheduling, priority ordering, "
        "cancellation races, periodic task drift correction, and the "
        "dead-letter queue overflow behavior. Document the public API with "
        "NumPy-style docstrings and provide usage examples for both the "
        "programmatic interface and the decorator syntax."
    ),
    "code_cpp": (
        "Implement a thread-safe bounded queue in C++17 using "
        "std::condition_variable and std::mutex. The queue should support "
        "blocking push and pop operations with an optional timeout parameter. "
        "Include move semantics, RAII lock management, and a try_pop method "
        "that returns std::optional. Show example producer-consumer usage."
    ),

    # ── Science ────────────────────────────────────────────────────────
    "science_physics_short": (
        "Explain the photoelectric effect and why classical wave theory "
        "cannot explain the observed threshold frequency."
    ),
    "science_physics": (
        "Explain the relationship between entropy, temperature, and free "
        "energy in thermodynamic systems. Derive the Helmholtz free energy "
        "from the partition function and show how it connects to the second "
        "law of thermodynamics. Discuss implications for spontaneous "
        "processes at constant temperature and volume, providing concrete "
        "examples from ideal gas expansion and phase transitions."
    ),
    "science_physics_long": (
        "Present a detailed derivation of the Boltzmann transport equation "
        "starting from the Liouville equation in phase space. Show how the "
        "BBGKY hierarchy leads to the molecular chaos assumption and the "
        "collision integral. Derive the H-theorem and prove that entropy "
        "production is non-negative under the Stosszahlansatz. Then apply "
        "the Chapman-Enskog expansion to obtain the Navier-Stokes equations "
        "as the hydrodynamic limit, showing explicitly how viscosity and "
        "thermal conductivity emerge as transport coefficients. Discuss the "
        "regime of validity: Knudsen number, mean free path, and where "
        "continuum mechanics breaks down. Compare with the direct simulation "
        "Monte Carlo (DSMC) method for rarefied gas dynamics and discuss "
        "computational cost scaling."
    ),
    "science_biology": (
        "Describe the molecular mechanism of CRISPR-Cas9 gene editing, "
        "including guide RNA design, PAM sequence recognition, and "
        "double-strand break repair pathways (NHEJ vs HDR). Compare the "
        "efficiency and off-target profiles of SpCas9 versus high-fidelity "
        "variants like eSpCas9 and HiFi Cas9. Discuss recent advances in "
        "base editing and prime editing that avoid double-strand breaks."
    ),

    # ── Dialogue ───────────────────────────────────────────────────────
    "dialogue_casual": (
        "Hey! So I was thinking about getting a new laptop for college. "
        "My budget is around $800 and I mainly need it for taking notes, "
        "watching lectures, and maybe some light photo editing. Do you think "
        "I should go with a MacBook Air or a ThinkPad? Also, is 8GB of RAM "
        "still enough in 2025, or should I aim for 16GB? What about battery "
        "life — that matters a lot since I'm on campus all day."
    ),
    "dialogue_formal": (
        "Dear Professor Chen, I am writing to respectfully inquire about "
        "the possibility of joining your research group for the upcoming "
        "academic year. I recently completed a course in statistical machine "
        "learning under Dr. Martinez, where my final project examined "
        "variational inference methods for latent variable models. I believe "
        "this background aligns well with your group's work on probabilistic "
        "deep learning and uncertainty quantification."
    ),

    # ── Legal ──────────────────────────────────────────────────────────
    "legal_short": (
        "This Agreement is effective as of the Effective Date and shall "
        "continue for a period of twelve months unless terminated earlier."
    ),
    "legal": (
        "The party of the first part (hereinafter 'Licensor') hereby grants "
        "to the party of the second part (hereinafter 'Licensee') a "
        "non-exclusive, non-transferable, royalty-bearing license to use, "
        "reproduce, and distribute the Licensed Material solely within the "
        "Territory defined in Schedule A, subject to the terms, conditions, "
        "and limitations set forth herein. This Agreement shall be governed "
        "by and construed in accordance with the laws of the State of "
        "Delaware, without regard to its conflict of laws provisions."
    ),
    "legal_long": (
        "WHEREAS, the Company (hereinafter 'Disclosing Party') possesses "
        "certain confidential and proprietary information relating to its "
        "business operations, technical know-how, trade secrets, customer "
        "lists, financial data, product roadmaps, and strategic plans "
        "(collectively, 'Confidential Information'); and WHEREAS, the "
        "Receiving Party desires to receive access to such Confidential "
        "Information for the sole purpose of evaluating a potential business "
        "relationship between the parties; NOW, THEREFORE, in consideration "
        "of the mutual covenants and agreements herein contained, and for "
        "other good and valuable consideration, the receipt and sufficiency "
        "of which are hereby acknowledged, the parties agree as follows: "
        "Section 1. Definition of Confidential Information. For purposes "
        "of this Agreement, Confidential Information shall mean all "
        "information, whether written, oral, electronic, or visual, that "
        "is disclosed by the Disclosing Party to the Receiving Party, "
        "including but not limited to: (a) technical data, inventions, "
        "processes, designs, drawings, engineering specifications, software "
        "source code, algorithms, and hardware configurations; (b) business "
        "information including marketing plans, pricing strategies, customer "
        "and supplier lists, and financial projections; (c) any information "
        "that a reasonable person would understand to be confidential given "
        "the nature of the information and circumstances of disclosure."
    ),

    # ── Multilingual ───────────────────────────────────────────────────
    "multilingual_french": (
        "Analysez les principales causes sociales, économiques et politiques "
        "de la Révolution française de 1789. Comment les idées des Lumières "
        "ont-elles influencé les revendications du tiers état? Discutez le "
        "rôle de la crise financière de la monarchie et de l'inégalité "
        "fiscale dans le déclenchement des États généraux. Comparez les "
        "objectifs des différentes factions révolutionnaires."
    ),
    "multilingual_chinese": (
        "请详细解释量子计算的基本原理，包括量子比特、量子叠加和量子纠缠的概念。"
        "比较超导量子比特和离子阱量子计算机的优缺点。讨论量子计算在密码学、"
        "药物发现和优化问题中的潜在应用。目前量子计算面临的主要技术挑战是什么？"
    ),

    # ── Mathematics ────────────────────────────────────────────────────
    "math_proof": (
        "Prove that every continuous function on a closed interval [a, b] "
        "is uniformly continuous. Begin by stating the epsilon-delta "
        "definition of uniform continuity and contrast it with pointwise "
        "continuity. Use the Heine-Cantor theorem approach: extract a finite "
        "subcover from the open cover generated by pointwise continuity "
        "deltas. Discuss why the result fails on open intervals with a "
        "concrete counterexample."
    ),
    "math_applied": (
        "Derive the backpropagation algorithm for a two-layer neural network "
        "with ReLU activation and cross-entropy loss. Start from the chain "
        "rule, compute gradients with respect to weights and biases at each "
        "layer, and express the update rules in matrix form. Discuss the "
        "vanishing gradient problem and how ReLU mitigates it compared to "
        "sigmoid activation."
    ),

    # ── Creative ───────────────────────────────────────────────────────
    "creative_story": (
        "Write the opening chapter of a science fiction story set in a "
        "generation ship traveling to Proxima Centauri. The protagonist is "
        "a botanist who discovers that the ship's hydroponic gardens are "
        "producing plants with unexpected mutations. The narrative should "
        "build tension through environmental details, interpersonal conflict "
        "with the ship's AI administrator, and foreshadowing of a larger "
        "mystery about the ship's true mission."
    ),

    # ── Technical documentation ────────────────────────────────────────
    "technical_docs": (
        "Write comprehensive API documentation for a REST endpoint that "
        "handles batch inference requests for a large language model. The "
        "endpoint accepts POST requests with a JSON body containing an array "
        "of prompts, temperature, max_tokens, and top_p parameters. Document "
        "request/response schemas, rate limits (100 RPM, 10K TPM), error "
        "codes (400, 401, 429, 500), and include curl examples for both "
        "single and batch requests with streaming responses."
    ),

    # ── Routing stress tests ───────────────────────────────────────────
    # These target specific routing behaviors rather than domains.

    "stress_repetitive": (
        "the the the the the the the the the the the the the the the the "
        "the the the the the the the the the the the the the the the the "
        "the the the the the the the the the the the the the the the the "
        "the the the the the the the the the the the the the the the the "
    ),
    "stress_mixed_domain": (
        "Prove that f(x)=x^2 is continuous. Hey, what's up? "
        "Section 4.2(b) of the Agreement notwithstanding, the Licensor "
        "hereby grants permission. Implement quicksort in Rust. "
        "La liberté guidant le peuple est un tableau d'Eugène Delacroix. "
        "The mitochondria is the powerhouse of the cell. SELECT * FROM "
        "users WHERE active = true ORDER BY created_at DESC LIMIT 10;"
    ),
    "stress_numbers": (
        "Calculate: 3.14159 * 2.71828 = ? Then compute the determinant of "
        "the matrix [[1, 2, 3], [4, 5, 6], [7, 8, 9]]. Find the eigenvalues "
        "of [[4, 1], [2, 3]]. Evaluate the integral of x^2 from 0 to 1. "
        "The Fibonacci sequence: 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89."
    ),
}

# Categories grouped by domain for analysis convenience.
PROMPT_CATEGORIES: Dict[str, List[str]] = {
    "code": ["code_python_short", "code_python", "code_python_long", "code_cpp"],
    "science": [
        "science_physics_short", "science_physics", "science_physics_long",
        "science_biology",
    ],
    "dialogue": ["dialogue_casual", "dialogue_formal"],
    "legal": ["legal_short", "legal", "legal_long"],
    "multilingual": ["multilingual_french", "multilingual_chinese"],
    "math": ["math_proof", "math_applied"],
    "creative": ["creative_story"],
    "technical": ["technical_docs"],
    "stress": ["stress_repetitive", "stress_mixed_domain", "stress_numbers"],
}


# ---------------------------------------------------------------------------
# Sequence-length sweep configuration
# ---------------------------------------------------------------------------

# Canonical sweep points.  The pipeline pads/truncates each prompt to these
# lengths and captures routing at each, enabling cross-length comparison
# while holding domain constant.
ROUTING_SWEEP_SEQ_LENS: List[int] = [64, 128, 256, 512, 1024, 2048]

# Representative prompts for sweeps — one per domain, chosen for medium
# natural length so both truncation and padding occur across the sweep.
SWEEP_REPRESENTATIVE_PROMPTS: List[str] = [
    "code_python",
    "science_physics",
    "legal",
    "multilingual_french",
    "math_proof",
    "creative_story",
    "stress_mixed_domain",
]


def build_sweep_matrix(
    seq_lens: Optional[List[int]] = None,
    prompt_names: Optional[List[str]] = None,
) -> List[Dict]:
    """Build a sweep matrix of (prompt_name, seq_len) configurations.

    Args:
        seq_lens: Sequence lengths to sweep.  Defaults to
            ``ROUTING_SWEEP_SEQ_LENS``.
        prompt_names: Prompts to include.  Defaults to
            ``SWEEP_REPRESENTATIVE_PROMPTS``.

    Returns:
        List of dicts, each with keys ``prompt_name``, ``prompt_text``,
        ``seq_len``.
    """
    if seq_lens is None:
        seq_lens = ROUTING_SWEEP_SEQ_LENS
    if prompt_names is None:
        prompt_names = SWEEP_REPRESENTATIVE_PROMPTS

    matrix = []
    for name in prompt_names:
        text = MOE_BENCHMARK_PROMPTS.get(name)
        if text is None:
            continue
        for sl in seq_lens:
            matrix.append({
                "prompt_name": name,
                "prompt_text": text,
                "seq_len": sl,
            })
    return matrix


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_prompt_names() -> List[str]:
    """Return all prompt names in deterministic order."""
    return sorted(MOE_BENCHMARK_PROMPTS.keys())


def get_prompts_for_categories(
    categories: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    """Return (name, prompt_text) pairs, optionally filtered by domain category.

    Args:
        categories: If provided, only return prompts from these domain groups
            (e.g., ["code", "science"]).  If None, return all prompts.

    Returns:
        List of (prompt_name, prompt_text) tuples.
    """
    if categories is None:
        return sorted(MOE_BENCHMARK_PROMPTS.items())

    names = []
    for cat in categories:
        names.extend(PROMPT_CATEGORIES.get(cat, []))

    return [(n, MOE_BENCHMARK_PROMPTS[n]) for n in sorted(set(names)) if n in MOE_BENCHMARK_PROMPTS]
