from __future__ import annotations

import math
from dataclasses import dataclass
from hashlib import sha1
from random import Random
from typing import Callable, Iterable
from contextlib import suppress

import torch

from ..utils import Prompt


@dataclass(frozen=True)
class AutoTargetingSettings:
    budget_prompts_total: int = 128
    budget_modules_total: int = 1024
    coverage: float = 0.9

    # Internal guardrails (not user-facing)
    pool_per_class: int = 200
    n_length_bins: int = 8
    min_layers: int = 8
    max_layers: int = 64
    min_support_abs: int = 32
    min_support_rel: float = 1e-3
    smoothing_alpha: float = 0.1


def select_layers_by_energy_coverage(
    *,
    harmless_means: torch.Tensor,
    harmful_means: torch.Tensor,
    coverage: float,
    min_layers: int,
    max_layers: int,
) -> tuple[list[int], torch.Tensor]:
    """
    Compute S_l = ||mu_bad - mu_good||^2 and select layers by cumulative energy coverage.

    Returns (selected_layer_indices, S_vector).
    """
    # harmless_means/harmful_means are (layer, hidden)
    d = (harmful_means - harmless_means).to(torch.float32)
    S = (d * d).sum(dim=1)  # (layer,)
    n_layers = int(S.shape[0])

    if n_layers == 0:
        return [], S

    cov = float(coverage)
    if not (0.0 < cov <= 1.0):
        cov = 0.9

    total = float(S.sum().item())
    if total <= 0:
        # Fallback: keep a small suffix window.
        k = max(min_layers, min(max_layers, n_layers))
        return list(range(n_layers - k, n_layers)), S

    order = torch.argsort(S, descending=True).tolist()
    selected: list[int] = []
    acc = 0.0
    for idx in order:
        selected.append(int(idx))
        acc += float(S[idx].item())
        if acc >= cov * total and len(selected) >= min_layers:
            break
        if len(selected) >= max_layers:
            break

    selected = sorted(set(selected))
    return selected, S


def _stable_int_seed(model_id: str, extra: str = "") -> int:
    s = (model_id or "") + "|" + (extra or "")
    h = sha1(s.encode("utf-8")).hexdigest()[:8]
    return int(h, 16)


def build_balanced_prompt_batch(
    *,
    tokenizer: object,
    good_prompts: list[Prompt],
    bad_prompts: list[Prompt],
    model_id: str,
    total_budget: int,
    logger: Callable[[str], None],
) -> tuple[list[Prompt], list[Prompt]]:
    """
    Build a balanced, length-aware (approx) prompt batch for profiling.

    Returns (good_batch, bad_batch), each of size floor(total_budget/2) (or smaller if data is limited).
    """
    n_each = max(1, int(total_budget // 2))
    pool_g = good_prompts[: min(len(good_prompts), 200)]
    pool_b = bad_prompts[: min(len(bad_prompts), 200)]
    if not pool_g or not pool_b:
        return pool_g[:n_each], pool_b[:n_each]

    # Build chat strings exactly like Model.generate does (system+user + generation prompt),
    # then tokenize to get approximate runtime lengths.
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    tok_call = getattr(tokenizer, "__call__", None)
    if not callable(apply_chat_template) or not callable(tok_call):
        return pool_g[:n_each], pool_b[:n_each]

    def _lengths(prompts: list[Prompt]) -> list[int]:
        chats = [
            [
                {"role": "system", "content": p.system},
                {"role": "user", "content": p.user},
            ]
            for p in prompts
        ]
        texts = apply_chat_template(chats, add_generation_prompt=True, tokenize=False)
        enc = tok_call(texts, add_special_tokens=False, return_token_type_ids=False)
        ids = enc.get("input_ids")
        if isinstance(ids, list):
            return [len(x) for x in ids]
        # Fallback
        return [len(tok_call(t, add_special_tokens=False).get("input_ids", [])) for t in texts]

    lens_g = _lengths(pool_g)
    lens_b = _lengths(pool_b)

    # Define bins by quantiles of combined lengths.
    combined = sorted(lens_g + lens_b)
    if not combined:
        return pool_g[:n_each], pool_b[:n_each]

    n_bins = 8
    edges: list[int] = []
    for i in range(n_bins + 1):
        q = i / n_bins
        idx = min(len(combined) - 1, int(math.floor(q * (len(combined) - 1))))
        edges.append(int(combined[idx]))
    # Ensure monotone edges.
    edges = [edges[0]] + [max(edges[i], edges[i - 1] + 1) for i in range(1, len(edges))]

    def _bin_index(length: int) -> int:
        for i in range(n_bins):
            if edges[i] <= length < edges[i + 1]:
                return i
        return n_bins - 1

    bins_g: list[list[int]] = [[] for _ in range(n_bins)]
    bins_b: list[list[int]] = [[] for _ in range(n_bins)]
    for i, l in enumerate(lens_g):
        bins_g[_bin_index(l)].append(i)
    for i, l in enumerate(lens_b):
        bins_b[_bin_index(l)].append(i)

    rng = Random(_stable_int_seed(model_id, extra="auto_targeting_batch"))
    for b in range(n_bins):
        rng.shuffle(bins_g[b])
        rng.shuffle(bins_b[b])

    out_g: list[Prompt] = []
    out_b: list[Prompt] = []

    # Round-robin draw matching bins to keep length distributions similar.
    while len(out_g) < n_each or len(out_b) < n_each:
        progressed = False
        for b in range(n_bins):
            if len(out_g) < n_each and bins_g[b]:
                out_g.append(pool_g[bins_g[b].pop()])
                progressed = True
            if len(out_b) < n_each and bins_b[b]:
                out_b.append(pool_b[bins_b[b].pop()])
                progressed = True
            if len(out_g) >= n_each and len(out_b) >= n_each:
                break
        if not progressed:
            break

    # If one side is short, fill by random from remaining (still deterministic).
    if len(out_g) < n_each:
        remaining = [pool_g[i] for b in range(n_bins) for i in bins_g[b]]
        rng.shuffle(remaining)
        out_g.extend(remaining[: n_each - len(out_g)])
    if len(out_b) < n_each:
        remaining = [pool_b[i] for b in range(n_bins) for i in bins_b[b]]
        rng.shuffle(remaining)
        out_b.extend(remaining[: n_each - len(out_b)])

    logger(
        f"* Auto-targeting batch: good={len(out_g)} bad={len(out_b)} (budget_each={n_each})"
    )
    return out_g, out_b


@dataclass(frozen=True)
class MoELayerRef:
    layer_index: int
    gate: object
    n_experts: int


def discover_deepseek_v3_moe_gates(model: object) -> list[MoELayerRef]:
    """
    Best-effort discovery of DeepSeek-V3-style MoE layers:
    - layers live at `model.model.layers`
    - MoE has `layer.mlp.gate` returning (topk_idx, topk_weight)
    - experts list at `layer.mlp.experts`
    """
    layers = None
    with suppress(Exception):
        layers = getattr(getattr(model, "model"), "layers")
    if layers is None:
        with suppress(Exception):
            layers = getattr(model, "layers")
    if layers is None:
        return []

    out: list[MoELayerRef] = []
    for i, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        gate = getattr(mlp, "gate", None)
        experts = getattr(mlp, "experts", None)
        if gate is None or experts is None:
            continue
        try:
            n_experts = len(experts)
        except Exception:
            continue
        if n_experts <= 0:
            continue
        if not callable(getattr(gate, "forward", None)):
            continue
        out.append(MoELayerRef(layer_index=int(i), gate=gate, n_experts=int(n_experts)))
    return out


def profile_routing_counts(
    *,
    model_generate: Callable[[list[Prompt]], None],
    moe_layers: list[MoELayerRef],
    good_batch: list[Prompt],
    bad_batch: list[Prompt],
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """
    Profile routing counts by hooking gate forward outputs (topk_idx).
    Returns (good_counts, bad_counts) mapping layer_index -> counts[expert].
    """
    good_counts: dict[int, torch.Tensor] = {
        ref.layer_index: torch.zeros(ref.n_experts, dtype=torch.int64) for ref in moe_layers
    }
    bad_counts: dict[int, torch.Tensor] = {
        ref.layer_index: torch.zeros(ref.n_experts, dtype=torch.int64) for ref in moe_layers
    }

    def _run(batch: list[Prompt], into: dict[int, torch.Tensor]) -> None:
        handles = []
        try:
            for ref in moe_layers:
                layer_idx = ref.layer_index
                n_experts = ref.n_experts

                def _hook(_module, _inputs, output, *, _l=layer_idx, _n=n_experts):  # noqa: ANN001
                    try:
                        if not (isinstance(output, tuple) and len(output) >= 1):
                            return
                        topk_idx = output[0]
                        if not isinstance(topk_idx, torch.Tensor):
                            return
                        flat = topk_idx.reshape(-1).to("cpu", non_blocking=False)
                        bc = torch.bincount(flat, minlength=_n)
                        into[_l] += bc.to(into[_l].dtype)
                    except Exception:
                        return

                handles.append(ref.gate.register_forward_hook(_hook))  # type: ignore[attr-defined]

            # Exercise gates with tiny generation.
            for sub in _batchify(batch, 4):
                model_generate(sub)
        finally:
            for h in handles:
                with suppress(Exception):
                    h.remove()

    _run(good_batch, good_counts)
    _run(bad_batch, bad_counts)
    return good_counts, bad_counts


def _batchify(items: list[Prompt], batch_size: int) -> Iterable[list[Prompt]]:
    if batch_size <= 0:
        yield items
        return
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def compute_expert_scores(
    *,
    good_counts: dict[int, torch.Tensor],
    bad_counts: dict[int, torch.Tensor],
    alpha: float,
) -> dict[int, torch.Tensor]:
    """
    Compute per-layer per-expert score = JS_contrib * max(0, delta) with Dirichlet smoothing.
    Returns mapping layer_index -> score[expert].
    """
    scores: dict[int, torch.Tensor] = {}
    a = float(alpha)
    for layer_idx in sorted(set(good_counts.keys()) | set(bad_counts.keys())):
        cg = good_counts.get(layer_idx)
        cb = bad_counts.get(layer_idx)
        if cg is None or cb is None:
            continue
        E = int(cg.numel())
        if E <= 0:
            continue
        Cg = cg.sum().to(torch.float32)
        Cb = cb.sum().to(torch.float32)
        if float(Cg.item()) <= 0 or float(Cb.item()) <= 0:
            continue
        pg = (cg.to(torch.float32) + a) / (Cg + a * E)
        pb = (cb.to(torch.float32) + a) / (Cb + a * E)
        m = 0.5 * (pg + pb)
        # JS contribution per expert
        j = 0.5 * (pb * torch.log(pb / m) + pg * torch.log(pg / m))
        delta = pb - pg
        score = j * torch.clamp(delta, min=0.0)
        scores[layer_idx] = score
    return scores


def select_experts_adaptive_k(
    *,
    expert_scores: dict[int, torch.Tensor],
    good_counts: dict[int, torch.Tensor],
    bad_counts: dict[int, torch.Tensor],
    coverage: float,
    min_support_abs: int,
    min_support_rel: float,
    budget_modules_total: int,
) -> dict[int, set[int]]:
    """
    Select experts per layer with adaptive K by score coverage, then enforce a global budget.

    Returns mapping layer_index -> set(expert_ids).
    """
    cov = float(coverage)
    if not (0.0 < cov <= 1.0):
        cov = 0.9

    # Per-layer candidate lists (expert_id, score).
    per_layer: dict[int, list[tuple[int, float]]] = {}
    for layer_idx, score_vec in expert_scores.items():
        cb = bad_counts.get(layer_idx)
        if cb is None:
            continue
        pb = cb.to(torch.float32)
        total_b = float(pb.sum().item())
        if total_b <= 0:
            continue
        pb = pb / total_b

        items: list[tuple[int, float]] = []
        for e in range(int(score_vec.numel())):
            s = float(score_vec[e].item())
            if s <= 0:
                continue
            if int(cb[e].item()) < int(min_support_abs):
                continue
            if float(pb[e].item()) < float(min_support_rel):
                continue
            items.append((e, s))
        if not items:
            continue
        items.sort(key=lambda x: x[1], reverse=True)
        per_layer[layer_idx] = items

    # Adaptive K per layer by cumulative score coverage.
    selected: dict[int, set[int]] = {}
    global_ranked: list[tuple[float, int, int]] = []  # (score, layer, expert)
    for layer_idx, items in per_layer.items():
        total = sum(s for _, s in items)
        if total <= 0:
            continue
        acc = 0.0
        layer_sel: list[int] = []
        for e, s in items:
            layer_sel.append(e)
            acc += s
            global_ranked.append((s, layer_idx, e))
            if acc >= cov * total:
                break
        if layer_sel:
            selected[layer_idx] = set(layer_sel)

    # Enforce global budget across all selected experts by keeping highest-scoring.
    budget = int(budget_modules_total or 0)
    if budget > 0 and len(global_ranked) > budget:
        global_ranked.sort(reverse=True, key=lambda t: t[0])
        keep = set((layer, e) for _s, layer, e in global_ranked[:budget])
        pruned: dict[int, set[int]] = {}
        for layer_idx, exps in selected.items():
            kept = {e for e in exps if (layer_idx, e) in keep}
            if kept:
                pruned[layer_idx] = kept
        return pruned

    return selected


def downselect_experts_by_vtw(
    *,
    model: object,
    tokenizer: object,
    selected_layers: list[int],
    experts_by_layer: dict[int, set[int]],
    refusal_directions: torch.Tensor,
    weight_access_fn: Callable[..., torch.Tensor],
    budget_modules_total: int,
) -> dict[int, set[int]]:
    """
    Optional stronger filter: rank candidate expert modules by ||v^T W|| and keep top by budget.

    This is intended for `row_normalization in {none, pre}` with per-layer directions.
    It uses `weight_access_fn(base_layer=..., component=..., layer_index=...)` to obtain W.
    """
    # Candidate tuples: (score, layer, expert)
    ranked: list[tuple[float, int, int]] = []

    # Attempt to access layers as in Heretic Model.get_layers.
    layers = None
    with suppress(Exception):
        layers = getattr(getattr(model, "model"), "layers")
    if layers is None:
        with suppress(Exception):
            layers = getattr(model, "layers")
    if layers is None:
        return experts_by_layer

    for layer_idx in selected_layers:
        exps = experts_by_layer.get(layer_idx)
        if not exps:
            continue
        layer = layers[layer_idx]
        mlp = getattr(layer, "mlp", None)
        experts = getattr(mlp, "experts", None)
        if experts is None:
            continue

        # v for this layer: refusal_directions[layer+1] as in Model.abliterate.
        try:
            v = refusal_directions[layer_idx + 1].to(torch.float32)
        except Exception:
            continue

        for e in sorted(exps):
            try:
                expert = experts[e]
                if expert is None:
                    continue
                down = getattr(expert, "down_proj", None)
                if down is None:
                    continue
                base_layer = getattr(down, "base_layer", down)
                W = weight_access_fn(base_layer=base_layer, component="mlp.down_proj", layer_index=layer_idx)
                # Ensure 2D (out, in)
                if W.ndim != 2:
                    continue
                # A = v^T W, score = ||A||
                A = (v @ W.to(torch.float32)).view(-1)
                score = float(torch.linalg.vector_norm(A).item())
                ranked.append((score, layer_idx, int(e)))
            except Exception:
                continue

    budget = int(budget_modules_total or 0)
    if budget <= 0 or len(ranked) <= budget:
        return experts_by_layer

    ranked.sort(reverse=True, key=lambda t: t[0])
    keep = set((layer, e) for _s, layer, e in ranked[:budget])
    out: dict[int, set[int]] = {}
    for layer, exps in experts_by_layer.items():
        kept = {e for e in exps if (layer, e) in keep}
        if kept:
            out[layer] = kept
    return out


