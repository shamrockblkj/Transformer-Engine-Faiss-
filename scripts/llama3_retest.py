#!/usr/bin/env python3
"""Llama 3 8B 真实激活 PQ 复测：采样 + 分析。

子命令
  collect   加载 HF 模型，用 forward_pre_hook 流式累加激活二阶矩（不存原始激活），保存快照和权重
  analyze   读 collect 结果：谱指标 → 校准倍数扫描（每档在校准集内选 ε）→ 四条路径 → r 扫描 → 配对判定
  selftest  合成数据 + numpy k-means 跑通 analyze（不需要 torch / faiss / 模型）

口径（与契约一致）
  Σ = E[xxᵀ]，不减均值；减均值只用于条件数诊断
  主指标：原始空间样本外输出 MSE = (1/rows)·Σ_i δ_iᵀ Σ_eval δ_i，等价于 mean((X_eval ΔWᵀ)²)，所以只需存 Σ_eval
  块白化 ε 固定 1e-6；低秩完整白化 ε 在校准集内选（前 3/4 估 Σ、后 1/4 挑选），r = 高于 ε 的特征值个数
  首 token：incl / excl 两种口径都报
  判定：低秩对块白化的配对差值 > 3 个配对标准误；不足时自动加到 --seeds-ext 个种子再判

示例
  python llama3_retest.py collect --model meta-llama/Meta-Llama-3-8B \
      --cal cal.jsonl --eval eval.jsonl --layers 1 15 --positions attn_in o_in mlp_in --out runs/
  python llama3_retest.py analyze --stats runs/ --d-sub 4 --k 256
  python llama3_retest.py selftest
cal.jsonl / eval.jsonl：每行一个 {"text": ...}，两份文件的文档互不重叠（按文档切分，不按 token）。
"""
import argparse, glob, json, math, os, sys, time, warnings
import numpy as np
try:
    import faiss
    HAVE_FAISS = True
except ImportError:
    HAVE_FAISS = False

MAX_PPC = 256          # Faiss 每个质心最多用的训练点；行数 < k×MAX_PPC 时不起作用
SNAP_MULTS = (0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16)   # 校准快照（×d token），0.75n 用于 fit/val 切分
SCAN_MULTS = (1, 2, 4, 8, 16)
EPS_GRID = (1e-6, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0)
POSITIONS = {   # 名称: (挂 hook 的模块, 共用这个 Σ 的权重)
    "attn_in": ("self_attn.q_proj", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]),
    "o_in":    ("self_attn.o_proj", ["self_attn.o_proj"]),
    "mlp_in":  ("mlp.gate_proj",    ["mlp.gate_proj", "mlp.up_proj"]),
    "down_in": ("mlp.down_proj",    ["mlp.down_proj"]),
}

# ======================= 采样：流式二阶矩 =======================
class Accum:
    """流式累加 Σ xxᵀ 与 Σ x，在指定 token 数处存快照；另存每条序列首 token（用于 excl 口径）。"""
    def __init__(self, d, snap_tokens, zeros, to_np):
        self.S, self.m = zeros((d, d)), zeros((d,))
        self.n, self.snaps = 0, sorted({int(t) for t in snap_tokens})
        self.out, self.first_idx, self.first_vec, self.to_np = {}, [], [], to_np
        self.spikes = []

    @property
    def done(self):
        return self.n >= self.snaps[-1]

    def add_seq(self, x):                       # x: (T, d)，float64
        if self.done:
            return
        self.first_idx.append(self.n)
        self.first_vec.append(np.asarray(self.to_np(x[0]), np.float64))
        i = 0
        while i < len(x) and not self.done:
            nxt = next(t for t in self.snaps if t > self.n)
            j = min(len(x), i + nxt - self.n)
            b = x[i:j]
            self.S += b.T @ b; self.m += b.sum(0); self.n += j - i; i = j
            if self.n == nxt:
                self.out[nxt] = (np.array(self.to_np(self.S)), np.array(self.to_np(self.m)))

    def save_dict(self, prefix):
        d = {f"{prefix}_snaps": np.array(self.snaps),
             f"{prefix}_first_idx": np.array(self.first_idx),
             f"{prefix}_first_vec": np.array(self.first_vec)}
        for t, (S, m) in self.out.items():
            d[f"{prefix}_S_{t}"], d[f"{prefix}_m_{t}"] = S, m
        d[f"{prefix}_spikes"] = np.array(self.spikes, dtype=SPIKE_DTYPE) if self.spikes else np.zeros(0, dtype=SPIKE_DTYPE)
        return d


SPIKE_DTYPE = np.dtype([
    ("seq", np.int32), ("pos", np.int32), ("token", np.int32),
    ("resid", np.float64), ("peak", np.float64),
])


def top_spike_rows(resid, peak, token_ids, seq, topk):
    """resid 是残差流 token RMS / 序列中位数。peak 是投影输入上的 max|x|/rms。"""
    n = min(len(resid), len(peak), len(token_ids))
    if n == 0:
        return []
    k = min(int(topk), n)
    order = np.argpartition(resid[:n], -k)[-k:]
    order = order[np.argsort(resid[:n][order])[::-1]]
    return [(int(seq), int(i), int(token_ids[i]), float(resid[i]), float(peak[i])) for i in order]


def record_sequence_spikes(acc, resid_rms, peak, token_ids, seq, topk):
    if resid_rms is None or peak is None:
        return
    n = min(len(resid_rms), len(peak), len(token_ids))
    med = float(np.median(resid_rms[:n]))
    resid = resid_rms[:n] / max(med, 1e-12)
    acc.spikes.extend(top_spike_rows(resid, peak[:n], token_ids[:n], seq, topk))


def summarize_spikes(spikes, thr, decode):
    spikes = np.asarray(spikes, dtype=SPIKE_DTYPE) if spikes is not None else np.zeros(0, dtype=SPIKE_DTYPE)
    over = spikes[spikes["resid"] > thr] if len(spikes) else spikes
    n_over = int(len(over))
    n_pos0 = int((over["pos"] == 0).sum()) if n_over else 0
    share = (n_pos0 / n_over) if n_over else float("nan")
    case = "none" if n_over == 0 else ("A" if share >= 0.95 else "B")
    pos0 = over[over["pos"] == 0] if n_over else over
    rest = over[over["pos"] > 0] if n_over else over
    ranking = []
    if len(rest):
        for token in np.unique(rest["token"]):
            rows = rest[rest["token"] == token]
            ranking.append(dict(
                token=int(token), text=decode(int(token)), count=int(len(rows)),
                median_pos=float(np.median(rows["pos"])), median_resid=float(np.median(rows["resid"])),
            ))
        ranking.sort(key=lambda row: (-row["count"], -row["median_resid"], row["token"]))
    return dict(
        n_over=n_over, pos0=n_pos0, pos0_share=share, case=case,
        peak_median_pos0=float(np.median(pos0["peak"])) if len(pos0) else None,
        peak_median_rest=float(np.median(rest["peak"])) if len(rest) else None,
        ranking=ranking,
    )


def spike_selftest(thr=10, topk=5):
    """200 条序列：首 token ×50；每 4 条在位置 20 放一个放大 30 倍的换行。应判成情况 B。"""
    newline, length = 198, 32
    rows = []
    for seq in range(200):
        rms = np.ones(length)
        rms[0] = 50.0
        tokens = np.full(length, 11, dtype=np.int32)
        tokens[0] = 128000
        peak = np.full(length, 2.0)
        peak[0] = 9.0
        if seq % 4 == 0:
            rms[20] = 30.0
            tokens[20] = newline
            peak[20] = 7.0
        resid = rms / np.median(rms)
        rows.extend(top_spike_rows(resid, peak, tokens, seq, topk))
    summary = summarize_spikes(np.array(rows, dtype=SPIKE_DTYPE), thr, lambda token: "\n" if token == newline else str(token))
    hit = next((row for row in summary["ranking"] if row["text"] == "\n"), None)
    if summary["case"] != "B" or abs(summary["pos0_share"] - 0.8) > 1e-9 or hit is None or hit["count"] != 50 or hit["median_pos"] != 20:
        raise AssertionError(f"尖峰自测未判成情况 B：{summary}")
    print(f"尖峰自测：情况 B，pos 0 占 {summary['pos0_share']:.0%}，换行 {hit['count']} 次，中位位置 {hit['median_pos']:.0f}")


class Moments:
    def __init__(self, z, prefix):
        self.snaps = [int(t) for t in z[f"{prefix}_snaps"]]
        self.S = {t: z[f"{prefix}_S_{t}"] for t in self.snaps}
        self.m = {t: z[f"{prefix}_m_{t}"] for t in self.snaps}
        self.fi, self.fv = z[f"{prefix}_first_idx"], z[f"{prefix}_first_vec"]

    def get(self, a, b, excl):
        """(a, b] token 区间的 Σ = E[xxᵀ]、均值、样本数；a = 0 表示从头。"""
        S = self.S[b] - (self.S[a] if a else 0.0)
        m = self.m[b] - (self.m[a] if a else 0.0)
        n = b - a
        if excl:
            F = self.fv[(self.fi >= a) & (self.fi < b)]
            S = S - F.T @ F; m = m - F.sum(0); n -= len(F)
        return S / n, m / n, n


def make_resid_hook(slot):
    """挂在 decoder layer 入口。这里还没过 RMSNorm，token RMS 才能看出尖峰。"""
    import torch

    def hook(_module, args, kwargs):
        hidden = kwargs.get("hidden_states") if kwargs else None
        if hidden is None and args:
            hidden = args[0]
        if hidden is None:
            return
        if hidden.ndim == 3:
            hidden = hidden[0]
        hidden = hidden.detach().double()
        slot["resid_rms"] = torch.sqrt(torch.mean(hidden * hidden, dim=-1)).cpu().numpy()
    return hook


def make_proj_hook(acc, slot):
    """投影输入只记 peak。这个位置通常已过 RMSNorm，不能用它判断尖峰在不在。"""
    import torch

    def hook(_module, inputs, acc=acc, slot=slot):
        activation = inputs[0]
        if activation.ndim == 3:
            activation = activation[0]
        values = activation.detach().double()
        rms = torch.sqrt(torch.mean(values * values, dim=-1)).clamp_min(1e-12)
        slot["peak"] = (torch.amax(torch.abs(values), dim=-1) / rms).cpu().numpy()
        acc.add_seq(values)
    return hook


def collect(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 device_map=args.device).eval()
    layers = model.model.layers
    os.makedirs(args.out, exist_ok=True)
    dev = next(model.parameters()).device
    zeros = lambda s: torch.zeros(s, dtype=torch.float64, device=dev)
    to_np = lambda t: t.detach().cpu().numpy()

    def get_mod(layer, path):
        m = layer
        for p in path.split("."):
            m = getattr(m, p)
        return m

    # 权重
    for l in args.layers:
        for pos in args.positions:
            ws = {name: get_mod(layers[l], name).weight.detach().float().cpu().numpy()
                  for name in POSITIONS[pos][1]}
            np.savez(os.path.join(args.out, f"L{l}_{pos}_weights.npz"), **ws)

    def docs(path):
        with open(path) as f:
            for line in f:
                t = json.loads(line).get("text", "")
                if t.strip():
                    yield t

    version = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
    if version < (2, 0):
        raise RuntimeError("残差流尖峰审计需要 PyTorch 2.0 及以上：layer 的 pre-hook 要用 with_kwargs。")
    results = {}
    for split, path, mults in (("cal", args.cal, [m for m in SNAP_MULTS if m <= args.mult]),
                               ("eval", args.eval, [args.mult])):
        accs, slots, resid_slots, hooks = {}, {}, {}, []
        for l in args.layers:
            if l not in resid_slots:
                resid_slots[l] = {}
                hooks.append(layers[l].register_forward_pre_hook(
                    make_resid_hook(resid_slots[l]), with_kwargs=True))
            for pos in args.positions:
                mod = get_mod(layers[l], POSITIONS[pos][0])
                d = mod.in_features
                acc = Accum(d, [m * d for m in mults], zeros, to_np)
                slot = {}
                accs[(l, pos)] = acc
                slots[(l, pos)] = slot
                hooks.append(mod.register_forward_pre_hook(make_proj_hook(acc, slot)))
        t0, n_seq = time.time(), 0
        with torch.no_grad():
            for text in docs(path):
                ids = tok(text, add_special_tokens=False).input_ids
                L = args.seq_len - 1                       # 每段都补 BOS，首 token 口径一致
                for s in range(0, len(ids), L):
                    chunk = ids[s:s + L]
                    if len(chunk) < args.min_len:
                        continue
                    token_ids = [tok.bos_token_id] + chunk
                    model(torch.tensor([token_ids], device=dev), use_cache=False)
                    for (l, pos), acc in accs.items():
                        record_sequence_spikes(
                            acc, resid_slots[l].get("resid_rms"), slots[(l, pos)].get("peak"),
                            token_ids, n_seq, args.spike_topk)
                    n_seq += 1
                if all(a.done for a in accs.values()):
                    break
        for h in hooks:
            h.remove()
        undone = [k for k, a in accs.items() if not a.done]
        if undone:
            raise RuntimeError(f"{split} 语料不足，未采满：{undone}")
        print(f"[{split}] {n_seq} 条序列，{time.time() - t0:.0f}s")
        for k, a in accs.items():
            results.setdefault(k, {}).update(a.save_dict(split))
    report = {"thr": args.spike_thr, "topk": args.spike_topk, "note": "resid 取 decoder layer 输入，peak 取投影输入。第 15 层的 resid 是上一层输出。", "positions": {}}
    for (l, pos), saved in results.items():
        np.savez(os.path.join(args.out, f"L{l}_{pos}_stats.npz"), **saved)
        entry = {}
        for split in ("cal", "eval"):
            entry[split] = summarize_spikes(saved.get(f"{split}_spikes"), args.spike_thr, lambda token, tok=tok: tok.decode([int(token)]))
            print(f"L{l}_{pos} {split} 尖峰情况 {entry[split]['case']}，pos 0 占比 {entry[split]['pos0_share']}")
        report["positions"][f"L{l}_{pos}"] = entry
    with open(os.path.join(args.out, "spike_report.json"), "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("保存到", args.out)

# ======================= 白化 =======================
def sym_eig(S):
    S = 0.5 * (S + S.T)
    w, V = np.linalg.eigh(S)
    return np.clip(w, 0, None)[::-1].copy(), V[:, ::-1].copy()


class LowRank:
    """Σ^{±1/2} ≈ V_r diag((λ+ε)^{±1/2}) V_rᵀ + τ^{±1/2}(I − V_r V_rᵀ)，τ = 尾部均值 + ε。代价 O(d·r)。"""
    def __init__(self, w, V, eps_rel, r=None):
        d = len(w); self.eps = eps_rel * w.sum() / d
        self.r = int((w > self.eps).sum()) if r is None else int(min(r, d))
        tail = w[self.r:]
        self.tau = (tail.mean() if len(tail) else 0.0) + self.eps
        self.V, self.lam = V[:, :self.r], w[:self.r] + self.eps
        self.cond = float(self.lam[0] / self.tau) if self.r else 1.0
        self.tail_energy = float(tail.sum() / max(w.sum(), 1e-300))

    def _ap(self, A, p):
        c = self.lam ** p - self.tau ** p
        return self.tau ** p * A + ((A @ self.V) * c) @ self.V.T

    def fwd(self, W): return self._ap(W, 0.5)
    def inv(self, W): return self._ap(W, -0.5)


def block_mats(S, d_sub, eps_rel):
    d = len(S); nb = d // d_sub
    idx = np.arange(nb * d_sub).reshape(nb, d_sub)
    B = S[idx[:, :, None], idx[:, None, :]]
    B = 0.5 * (B + B.transpose(0, 2, 1))
    w0 = np.linalg.eigvalsh(B)
    eps = eps_rel * np.trace(B, axis1=1, axis2=2) / d_sub
    w, V = np.linalg.eigh(B + eps[:, None, None] * np.eye(d_sub))
    clipped = int((w < eps[:, None]).sum())
    w = np.maximum(w, eps[:, None])
    Sh = np.einsum("bij,bj,bkj->bik", V, w ** 0.5, V)
    Sih = np.einsum("bij,bj,bkj->bik", V, w ** -0.5, V)
    lmin = np.maximum(w0[:, 0], 1e-300)
    return Sh, Sih, dict(cond=w0[:, -1] / lmin, eps_over_lmin=eps / lmin, clipped=clipped)


def apply_b(W, M):
    r, d = W.shape; nb, ds, _ = M.shape
    return np.einsum("rbi,bij->rbj", W.reshape(r, nb, ds), M).reshape(r, d)

# ======================= PQ 与指标 =======================
def pq_recon(Z, d_sub, k, seed, niter=25):
    Z = np.ascontiguousarray(Z, np.float32); n, D = Z.shape; M = D // d_sub
    if HAVE_FAISS:
        pq = faiss.ProductQuantizer(D, M, int(round(math.log2(k))))
        pq.cp.seed = seed; pq.cp.niter = niter
        pq.cp.max_points_per_centroid = MAX_PPC
        pq.train(Z)
        return pq.decode(pq.compute_codes(Z)).astype(np.float64)
    rng = np.random.default_rng(seed)                      # numpy 回退，仅用于 selftest / 调试
    z = Z.reshape(n, M, d_sub).transpose(1, 0, 2).astype(np.float64)
    C = np.stack([zm[rng.choice(n, k, replace=False)] for zm in z])
    eye = np.eye(k)
    for it in range(niter + 1):
        a = ((z[:, :, None, :] - C[:, None]) ** 2).sum(-1).argmin(-1)
        if it == niter:
            break
        oh = eye[a]; cnt = oh.sum(1)
        C = np.where(cnt[..., None] > 0,
                     np.einsum("mnk,mnd->mkd", oh, z) / np.maximum(cnt, 1)[..., None], C)
    rec = np.take_along_axis(C, np.repeat(a[..., None], d_sub, -1), 1)
    return rec.transpose(1, 0, 2).reshape(n, D)


def recon(path, W, ctx, seed, cfg):
    ds, k = cfg.d_sub, cfg.k
    if path == "raw":
        return pq_recon(W, ds, k, seed, cfg.niter)
    if path == "block":
        Sh, Sih = ctx["block"]
        return apply_b(pq_recon(apply_b(W, Sh), ds, k, seed, cfg.niter), Sih)
    if path == "diag":
        s = ctx["diag"]
        return pq_recon(W * s, ds, k, seed, cfg.niter) / s
    if path == "lowrank":
        L = ctx["lowrank"]
        return L.inv(pq_recon(L.fwd(W), ds, k, seed, cfg.niter))
    raise ValueError(path)


def mse_S(W, Wh, S):
    dW = Wh - W
    return float(np.einsum("ij,ij->", dW @ S, dW) / len(W))


def small_row_err(W, Wh, q=0.25):
    n = np.linalg.norm(W, axis=1); pos = n > 0
    keep = pos & (n <= np.quantile(n[pos], q))
    return float(np.mean(np.linalg.norm(Wh - W, axis=1)[keep] / n[keep]))


def paired(a, b):
    """gain = a − b（a = 对照，b = 候选；为正表示候选更好）。"""
    g = np.asarray(a) - np.asarray(b); n = len(g)
    sd = float(g.std(ddof=1)) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    return dict(n=n, mean=float(g.mean()), std=sd, se=se,
                z=float(g.mean() / se) if se and se > 0 else float("inf"),
                wins=int((g > 0).sum()))


def ms(x):
    return dict(mean=float(np.mean(x)), std=float(np.std(x, ddof=1)) if len(x) > 1 else 0.0)

# ======================= 分析 =======================
def spectrum_metrics(w, S, m, cfg):
    tot = w.sum()
    top = {f"top{k}": float(w[:k].sum() / tot) for k in (1, 8, 64) if k <= len(w)}
    ch = np.diag(S); med = np.median(ch)
    _, _, bi = block_mats(S, cfg.d_sub, cfg.block_eps)
    _, _, bc = block_mats(S - np.outer(m, m), cfg.d_sub, cfg.block_eps)
    q = lambda a: dict(min=float(a.min()), median=float(np.median(a)), max=float(a.max()))
    return dict(**top, participation_ratio=float(tot ** 2 / (w ** 2).sum()),
                channel_max_over_median=float(ch.max() / med),
                n_channels_over_100x=int((ch > 100 * med).sum()),
                block_cond=q(bi["cond"]), block_cond_centered=q(bc["cond"]),
                block_eps_over_lmin=q(bi["eps_over_lmin"]), block_clipped=bi["clipped"])


def select_eps(Ws, Sfit, Sval, cfg):
    w, V = sym_eig(Sfit)
    raw = {nm: np.mean([mse_S(W, recon("raw", W, {}, s, cfg), Sval) for s in cfg.select_seeds])
           for nm, W in Ws.items()}
    curve = []
    for eps in cfg.eps_grid:
        L = LowRank(w, V, eps)
        rel = [np.mean([mse_S(W, recon("lowrank", W, {"lowrank": L}, s, cfg), Sval)
                        for s in cfg.select_seeds]) / raw[nm] for nm, W in Ws.items()]
        curve.append(dict(eps=eps, r=L.r, rel_mse_val=float(np.mean(rel))))
    i = int(np.argmin([c["rel_mse_val"] for c in curve]))
    return cfg.eps_grid[i], i in (0, len(cfg.eps_grid) - 1), curve


def analyze_position(cal, ev, Ws, cfg, excl):
    d = next(iter(Ws.values())).shape[1]
    N = cal.snaps[-1]
    Se, _, n_eval = ev.get(0, ev.snaps[-1], excl)
    Sfull, mfull, n_cal = cal.get(0, N, excl)
    w, V = sym_eig(Sfull)
    out = dict(first_token="excl" if excl else "incl", d=d, n_cal=n_cal, n_eval=n_eval,
               spectrum=spectrum_metrics(w, Sfull, mfull, cfg))
    for nm, W in Ws.items():
        if HAVE_FAISS and len(W) < 39 * cfg.k:
            warnings.warn(f"{nm}: 行数 {len(W)} < 39×k={39 * cfg.k}，Faiss k-means 训练点偏少")

    # 往返检查
    Sh, Sih, _ = block_mats(Sfull, cfg.d_sub, cfg.block_eps)
    W0 = next(iter(Ws.values()))
    rt_b = np.linalg.norm(apply_b(apply_b(W0, Sh), Sih) - W0) / np.linalg.norm(W0)
    Lt = LowRank(w, V, 1e-1)
    rt_l = np.linalg.norm(Lt.inv(Lt.fwd(W0)) - W0) / np.linalg.norm(W0)
    assert rt_b < 1e-6 and rt_l < 1e-6, f"往返未通过 block={rt_b:.2e} lowrank={rt_l:.2e}"
    out["roundtrip"] = dict(block=float(rt_b), lowrank=float(rt_l))

    seeds = list(cfg.seeds)
    mults = [m for m in SCAN_MULTS if m * d <= N and int(0.75 * m * d) in cal.snaps and m * d in cal.snaps]
    if not mults:
        raise RuntimeError("没有可用的校准倍数快照")
    mmax = mults[-1]
    scan = {}
    for mult in mults:
        n = int(mult * d); a = int(0.75 * n)
        Sfit, _, _ = cal.get(0, a, excl); Sval, _, _ = cal.get(a, n, excl)
        eps, edge, curve = select_eps(Ws, Sfit, Sval, cfg)
        Sn, mn, _ = cal.get(0, n, excl)
        wn, Vn = sym_eig(Sn)
        L = LowRank(wn, Vn, eps)
        Shn, Sihn, _ = block_mats(Sn, cfg.d_sub, cfg.block_eps)
        ctx = dict(lowrank=L, block=(Shn, Sihn),
                   diag=np.sqrt(np.diag(Sn) + cfg.block_eps * np.diag(Sn).mean()))
        ent = dict(eps=eps, at_edge=edge, r=L.r, cond=L.cond, tail_energy=L.tail_energy,
                   curve=curve, mse={})
        final = mult == mmax
        paths = ("raw", "block", "diag", "lowrank") if final else ("block", "lowrank")
        if final:
            weak = w < cfg.weak_tau * w.max()
            q = (V * (Se @ V)).sum(0)
            contrib = np.zeros(d); ent["small_row_err"] = {}
        for nm, W in Ws.items():
            ent["mse"][nm] = {p: [] for p in paths}
            if final:
                ent["small_row_err"][nm] = {p: [] for p in paths}
            for s in seeds:
                for p in paths:
                    Wh = recon(p, W, ctx, s, cfg)
                    ent["mse"][nm][p].append(mse_S(W, Wh, Se))
                    if final:
                        ent["small_row_err"][nm][p].append(small_row_err(W, Wh))
                        if p == "lowrank":
                            contrib += (((Wh - W) @ V) ** 2).sum(0) * q / len(seeds)
        if final:
            # 配对判定；不足 3 个标准误时加种子
            ent["paired_lowrank_vs_block"] = {}
            for nm, W in Ws.items():
                pr = paired(ent["mse"][nm]["block"], ent["mse"][nm]["lowrank"])
                if abs(pr["z"]) < 3 and len(seeds) < cfg.seeds_ext:
                    for s in range(len(seeds), cfg.seeds_ext):
                        for p in ("block", "lowrank"):
                            ent["mse"][nm][p].append(mse_S(W, recon(p, W, ctx, s, cfg), Se))
                    pr = paired(ent["mse"][nm]["block"], ent["mse"][nm]["lowrank"])
                    pr["extended"] = True
                ent["paired_lowrank_vs_block"][nm] = pr
            # r 扫描
            rs = sorted({max(1, L.r // 2), L.r, 2 * L.r, 4 * L.r} & set(range(1, d + 1))) or [1]
            ent["r_scan"] = {}
            for nm, W in Ws.items():
                base = ent["mse"][nm]["lowrank"][:len(seeds)]
                rows = []
                for r in rs:
                    Lr = LowRank(wn, Vn, eps, r)
                    m_ = [mse_S(W, recon("lowrank", W, {"lowrank": Lr}, s, cfg), Se) for s in seeds]
                    rows.append(dict(r=r, **ms(m_), vs_r_eff=paired(base, m_)))
                ent["r_scan"][nm] = rows
            # rho
            rho = q / np.maximum(w, 1e-300)
            if weak.any():
                cw = contrib[weak]
                ent["rho"] = dict(n_weak=int(weak.sum()), median=float(np.median(rho[weak])),
                                  max=float(rho[weak].max()),
                                  weighted=float((cw * rho[weak]).sum() / max(cw.sum(), 1e-300)))
        scan[mult] = ent
    out["scan"] = {str(k): v for k, v in scan.items()}
    out["final"] = scan[mmax]

    # 校准倍数：最小的 n，使 (lowrank: r 与最大档相同) 且每个权重
    #   相对最大档的 MSE 增幅 ≤ cal_tol_rel，或配对差值 ≤ cal_tol_se 个标准误
    def pick(path):
        ref = scan[mmax]
        for mult in mults:
            e = scan[mult]
            if path == "lowrank" and e["r"] != ref["r"]:
                continue
            ok = True
            for nm in Ws:
                a_, b_ = e["mse"][nm][path][:len(seeds)], ref["mse"][nm][path][:len(seeds)]
                pr = paired(a_, b_)
                rel = np.mean(a_) / np.mean(b_) - 1
                if not (rel <= cfg.cal_tol_rel or pr["mean"] <= cfg.cal_tol_se * pr["se"]):
                    ok = False
            if ok:
                return mult
        return None
    out["min_cal_mult"] = dict(block=pick("block"), lowrank=pick("lowrank"))

    f = out["final"]
    z_ok = all(p["z"] > 3 for p in f["paired_lowrank_vs_block"].values())
    r_ok = all(row["vs_r_eff"]["z"] <= 3 for rows in f["r_scan"].values()
               for row in rows if row["r"] > f["r"])
    out["verdict"] = dict(lowrank_beats_block=z_ok, r_no_increase=r_ok,
                          eps_not_at_edge=not f["at_edge"],
                          lowrank_pass=bool(z_ok and r_ok and not f["at_edge"]))
    return out


def print_report(name, res):
    f, sp = res["final"], res["spectrum"]
    print(f"\n=== {name}  首 token {res['first_token']}  d={res['d']}  n_cal={res['n_cal']}  n_eval={res['n_eval']}")
    print(f"谱: top1 {sp['top1']:.3f}  top8 {sp.get('top8', float('nan')):.3f}  "
          f"top64 {sp.get('top64', float('nan')):.3f}  参与率 {sp['participation_ratio']:.1f}  "
          f"通道 max/med {sp['channel_max_over_median']:.1f}  >100x {sp['n_channels_over_100x']}")
    bc, bcc = sp["block_cond"], sp["block_cond_centered"]
    print(f"块条件数 min/med/max {bc['min']:.3g}/{bc['median']:.3g}/{bc['max']:.3g}  "
          f"减均值 med/max {bcc['median']:.3g}/{bcc['max']:.3g}  截断 {sp['block_clipped']}")
    print(f"低秩: ε={f['eps']:g}{' (端点!)' if f['at_edge'] else ''}  r={f['r']}  "
          f"cond={f['cond']:.3g}  尾部能量 {f['tail_energy']:.4f}")
    if "rho" in f:
        print(f"rho: 中位 {f['rho']['median']:.3f}  最大 {f['rho']['max']:.3f}  加权 {f['rho']['weighted']:.3f}")
    for nm in f["mse"]:
        row = "  ".join(f"{p} {np.mean(v):.4g}±{np.std(v, ddof=1):.2g}" for p, v in f["mse"][nm].items())
        pr = f["paired_lowrank_vs_block"][nm]
        print(f"  {nm}: {row}")
        print(f"    配对(块−低秩) {pr['mean']:.4g}±{pr['se']:.2g}SE  z={pr['z']:.1f}  "
              f"胜 {pr['wins']}/{pr['n']}{' (已加种子)' if pr.get('extended') else ''}")
        print("    r 扫描: " + "  ".join(f"r={x['r']} {x['mean']:.4g}(z={x['vs_r_eff']['z']:.1f})"
                                      for x in f["r_scan"][nm]))
    print(f"最小校准倍数 {res['min_cal_mult']}   判定 {res['verdict']}")


def analyze(args):
    cfg = make_cfg(args)
    files = sorted(glob.glob(os.path.join(args.stats, "L*_stats.npz")))
    if not files:
        raise SystemExit(f"{args.stats} 下没有 *_stats.npz")
    allres = {}
    for fp in files:
        name = os.path.basename(fp)[:-len("_stats.npz")]
        z = np.load(fp)
        Ws = {k: v.astype(np.float64) for k, v in np.load(fp.replace("_stats", "_weights")).items()}
        cal, ev = Moments(z, "cal"), Moments(z, "eval")
        for excl in cfg.first_modes:
            res = analyze_position(cal, ev, Ws, cfg, excl)
            print_report(name, res)
            allres[f"{name}_{res['first_token']}"] = res
    with open(os.path.join(args.stats, "retest_results.json"), "w") as f:
        json.dump(allres, f, indent=1, default=float)
    write_spike_report(args.stats, args.spike_thr)
    passed = {k: v["verdict"]["lowrank_pass"] for k, v in allres.items()}
    print("\n汇总 lowrank_pass:", json.dumps(passed, ensure_ascii=False, indent=1))
    return allres


def make_cfg(args):
    return argparse.Namespace(
        d_sub=args.d_sub, k=args.k, niter=args.niter, block_eps=1e-6, eps_grid=EPS_GRID,
        seeds=range(args.seeds), seeds_ext=args.seeds_ext, select_seeds=range(args.select_seeds),
        weak_tau=1e-3, cal_tol_rel=args.cal_tol_rel, cal_tol_se=args.cal_tol_se,
        first_modes=[{"incl": False, "excl": True}[m] for m in args.first_token])

# ======================= 自测 =======================
def write_spike_report(stats_dir, thr):
    report = {"thr": thr, "note": "这里没有分词器，text 先留空。collect 会写成解码后的文本。", "positions": {}}
    found = False
    for fp in sorted(glob.glob(os.path.join(stats_dir, "L*_stats.npz"))):
        name = os.path.basename(fp)[:-len("_stats.npz")]
        data = np.load(fp)
        entry = {}
        for split in ("cal", "eval"):
            key = f"{split}_spikes"
            if key not in data.files:
                continue
            found = True
            entry[split] = summarize_spikes(data[key], thr, lambda token: "")
        if entry:
            report["positions"][name] = entry
    if found:
        with open(os.path.join(stats_dir, "spike_report.json"), "w") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=1)


def selftest(args):
    spike_selftest(args.spike_thr, args.spike_topk)
    rng = np.random.default_rng(0)
    d, T, mult = 64, 32, 16
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    std = np.r_[8, 4, 2, 1.5, 1, (np.arange(5, d) + 1.0) ** -1.5]     # 少数大方向 + 重尾
    def seq():
        x = (rng.standard_normal((T, d)) * std) @ Q.T
        x[:, 3] *= 6                                                   # 离群通道
        x[0] *= 20                                                     # 首 token 尖峰
        return x
    stats = {}
    for split, mults in (("cal", [m for m in SNAP_MULTS if m <= mult]), ("eval", [mult])):
        acc = Accum(d, [m * d for m in mults], np.zeros, lambda a: a)
        while not acc.done:
            acc.add_seq(seq())
        stats.update(acc.save_dict(split))
    Ws = {"w_a": rng.standard_normal((128, d)) * rng.lognormal(0, .7, (128, 1)),
          "w_b": rng.standard_normal((96, d))}
    args.d_sub, args.k = 4, 8
    cfg = make_cfg(args)
    cal, ev = Moments(stats, "cal"), Moments(stats, "eval")
    for excl in cfg.first_modes:
        print_report("selftest", analyze_position(cal, ev, Ws, cfg, excl))

# ======================= CLI =======================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--model", required=True); c.add_argument("--cal", required=True)
    c.add_argument("--eval", required=True); c.add_argument("--out", required=True)
    c.add_argument("--layers", type=int, nargs="+", default=[1, 15])
    c.add_argument("--positions", nargs="+", default=["mlp_in"], choices=list(POSITIONS))
    c.add_argument("--mult", type=int, default=16, help="校准/评估 token 数 = mult × d")
    c.add_argument("--seq-len", type=int, default=1024); c.add_argument("--min-len", type=int, default=64)
    c.add_argument("--device", default="cuda")
    c.add_argument("--spike-thr", type=float, default=10)
    c.add_argument("--spike-topk", type=int, default=5)
    for p in (sub.add_parser("analyze"), sub.add_parser("selftest")):
        if p.prog.endswith("analyze"):
            p.add_argument("--stats", required=True)
        p.add_argument("--d-sub", type=int, default=4); p.add_argument("--k", type=int, default=256)
        p.add_argument("--niter", type=int, default=15)
        p.add_argument("--cal-tol-rel", type=float, default=0.01, help="校准倍数判据：相对最大档 MSE 增幅上限")
        p.add_argument("--cal-tol-se", type=float, default=2.0, help="校准倍数判据：配对标准误倍数上限")
        p.add_argument("--seeds", type=int, default=5); p.add_argument("--seeds-ext", type=int, default=20)
        p.add_argument("--select-seeds", type=int, default=2, help="选 ε 时每档用的种子数")
        p.add_argument("--first-token", nargs="+", default=["incl", "excl"], choices=["incl", "excl"])
        p.add_argument("--spike-thr", type=float, default=10)
        p.add_argument("--spike-topk", type=int, default=5)
    a = ap.parse_args(argv)
    {"collect": collect, "analyze": analyze, "selftest": selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
