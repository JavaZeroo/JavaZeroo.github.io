"""DSA 的两阶段训练教学示例：CPU、小型 MQA、合成周期序列。

运行：python dsa_training_demo.py
仅自检：python dsa_training_demo.py --check-only
依赖：PyTorch。无需下载权重或数据。

对应博客 /2026/11/12/deepseek-dsa/ 。这是独立教学实现，不是 DeepSeek
训练代码：用 MQA 和绝对位置嵌入代替 MLA/RoPE，不含 FP8、分布式与稀疏
kernel；仍构造稠密分数矩阵，不能用来测 DSA 加速比或模型能力。
"""

import argparse
import math

import torch
from torch import nn
from torch.nn import functional as F


class Indexer(nn.Module):
    def __init__(self, dim=48, latent=24, heads=4, head_dim=8):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        self.q = nn.Linear(latent, heads * head_dim, bias=False)
        self.k = nn.Sequential(nn.Linear(dim, head_dim, bias=False),
                               nn.LayerNorm(head_dim))
        self.w = nn.Linear(dim, heads, bias=False)

    def forward(self, h, qr):
        # 同时切断 hidden 和共享 query latent 两条输入的梯度。
        h, qr = h.detach(), qr.detach()
        b, length, _ = h.shape
        q = self.q(qr).reshape(b, length, self.heads, self.head_dim)
        k = self.k(h)
        dots = torch.einsum("bthd,bsd->bths", q, k)
        weights = self.w(h) / math.sqrt(self.heads * self.head_dim)
        return (dots.relu() * weights.unsqueeze(-1)).sum(dim=2)


class TinyLM(nn.Module):
    def __init__(self, vocab=32, length=32, dim=48, heads=4, head_dim=12):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        self.embedding = nn.Embedding(vocab, dim)
        self.position = nn.Parameter(torch.randn(length, dim) * 0.02)
        self.norm = nn.LayerNorm(dim)
        self.query_latent = nn.Sequential(nn.Linear(dim, 24), nn.LayerNorm(24))
        self.q = nn.Linear(24, heads * head_dim, bias=False)
        self.k = nn.Linear(dim, head_dim, bias=False)
        self.v = nn.Linear(dim, head_dim, bias=False)
        self.out = nn.Linear(heads * head_dim, dim, bias=False)
        self.readout = nn.Linear(dim, vocab)
        self.indexer = Indexer(dim=dim)

    def parameter_groups(self):
        main = [p for n, p in self.named_parameters() if not n.startswith("indexer.")]
        return main, list(self.indexer.parameters())

    def forward(self, tokens, *, mode="dense", topk=8):
        b, length = tokens.shape
        residual = self.embedding(tokens) + self.position[:length]
        h = self.norm(residual)
        qr = self.query_latent(h)
        q = self.q(qr).reshape(b, length, self.heads, self.head_dim)
        logits = torch.einsum("bthd,bsd->bths", q, self.k(h))
        logits = logits / math.sqrt(self.head_dim)
        causal = torch.ones(length, length, device=h.device, dtype=torch.bool).tril()
        valid = causal.unsqueeze(0).expand(b, -1, -1)
        index_logits = None
        if mode != "dense":
            index_logits = self.indexer(h, qr).masked_fill(~valid, -torch.inf)
        if mode == "sparse":
            # top-k 的离散下标不参与反向传播。
            with torch.no_grad():
                indices = index_logits.topk(min(topk, length), dim=-1).indices
                chosen = torch.zeros_like(valid).scatter_(-1, indices, True)
                # 早期行合法历史少于 k，topk 可能返回 -inf 占位；必须再交 causal。
                valid = valid & chosen
        a = logits.masked_fill(~valid.unsqueeze(2), -torch.inf).softmax(dim=-1)
        context = torch.einsum("bths,bsd->bthd", a, self.v(h))
        output = self.readout(residual + self.out(context.flatten(2)))
        # 教师是当前主注意力的概率；各头平均后沿 key 维仍归一化。
        teacher = a.detach().mean(dim=2)
        return output, index_logits, teacher, valid


def index_kl(scores, teacher, valid):
    log_r = scores.float().masked_fill(~valid, -torch.inf).log_softmax(dim=-1)
    # 非法位置 target=0，先将 log_r 填0，避免 0*(-inf) 引发 NaN。
    log_r = log_r.masked_fill(~valid, 0.0)
    return F.kl_div(log_r, teacher.detach().float(), reduction="none").sum(-1).mean()


def lm_loss(logits, targets):
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten())


def sample(batch=8, length=32, vocab=32):
    # 每条序列重复随机生成的8个符号；最初的符号无法从历史预测。
    seed = torch.randint(vocab, (batch, 8))
    sequence = seed.repeat(1, math.ceil((length + 1) / 8))[:, :length + 1]
    return sequence[:, :-1], sequence[:, 1:]


def self_check(model):
    """验证目标/梯度/因果语义，不把演示脚本误当作性能基准。"""
    main, index = model.parameter_groups()
    tokens, targets = sample(batch=2)
    logits, scores, p, valid = model(tokens, mode="sparse")
    ce, kl = lm_loss(logits, targets), index_kl(scores, p, valid)
    assert torch.isfinite(ce) and torch.isfinite(kl)
    torch.testing.assert_close(p.sum(-1), torch.ones_like(p[..., 0]))
    assert not valid.triu(1).any(), "未来位置进入支持集"
    assert (valid.sum(-1) <= 8).all() and (valid.sum(-1) > 0).all()
    assert all(g is None for g in torch.autograd.grad(ce, index, allow_unused=True,
                                                     retain_graph=True))
    assert all(g is None for g in torch.autograd.grad(kl, main, allow_unused=True,
                                                     retain_graph=True))
    assert any(g is not None and g.abs().sum() > 0 for g in
               torch.autograd.grad(ce, main, allow_unused=True, retain_graph=True))
    assert any(g is not None and g.abs().sum() > 0 for g in
               torch.autograd.grad(kl, index, allow_unused=True))
    with torch.no_grad():
        dense = model(tokens)[0]
        full = model(tokens, mode="sparse", topk=tokens.shape[1])[0]
        torch.testing.assert_close(dense, full)
        # 改写后半序列不应影响前半输出或前半选中集合。
        changed = tokens.clone()
        changed[:, 16:] = (changed[:, 16:] + 1) % 32
        before = model(tokens, mode="sparse")
        after = model(changed, mode="sparse")
        torch.testing.assert_close(before[0][:, :16], after[0][:, :16])
        assert torch.equal(before[3][:, :16], after[3][:, :16])
    print("checks: causal mask, normalized teacher, gradient isolation, k=N equivalence: OK")


@torch.no_grad()
def probe(model, tokens, targets, topk=8):
    # 仅在小规模验证集重算 dense 教师，监控全局保留质量。
    dense_logits, scores, dense_p, valid = model(tokens, mode="warmup")
    sparse_logits, sparse_scores, sparse_p, selected = model(tokens, mode="sparse", topk=topk)
    rows = torch.arange(tokens.shape[1]) + 1 > topk
    retained = (dense_p * selected).sum(-1)[:, rows].mean()
    return {
        "dense_ce": float(lm_loss(dense_logits, targets)),
        "sparse_ce": float(lm_loss(sparse_logits, targets)),
        "full_kl": float(index_kl(scores, dense_p, valid)),
        "selected_kl": float(index_kl(sparse_scores, sparse_p, selected)),
        "retained_mass": float(retained),
    }


def run(args):
    torch.manual_seed(7)
    torch.set_num_threads(2)
    model = TinyLM()
    self_check(model)
    if args.check_only:
        return
    main, index = model.parameter_groups()
    probe_x, probe_y = sample(batch=32)

    # 教学准备阶段。实际 V3.2 从已训练好的 V3.1-Terminus checkpoint 起步。
    dense_optimizer = torch.optim.AdamW(main, lr=3e-3)
    for _ in range(args.dense_steps):
        x, y = sample()
        loss = lm_loss(model(x)[0], y)
        dense_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        dense_optimizer.step()
    print("dense checkpoint:", probe(model, probe_x, probe_y))

    # A：主干冻结，保持 dense attention；只有 indexer 接收全局 KL 梯度。
    for p in main:
        p.requires_grad_(False)
        p.grad = None
    before = [p.detach().clone() for p in main]
    index_optimizer = torch.optim.AdamW(index, lr=3e-3)
    for _ in range(args.warmup_steps):
        x, _ = sample()
        _, scores, teacher, valid = model(x, mode="warmup")
        kl = index_kl(scores, teacher, valid)
        index_optimizer.zero_grad(set_to_none=True)
        kl.backward()
        index_optimizer.step()
    assert all(torch.equal(a, b) for a, b in zip(before, main)), "热身改动了主干"
    print("after warm-up:", probe(model, probe_x, probe_y))

    # B：独立优化器；示例学习率仅供这个小模型，非官方超参。
    for p in main:
        p.requires_grad_(True)
    main_optimizer = torch.optim.AdamW(main, lr=1e-3)
    for group in index_optimizer.param_groups:
        group["lr"] = 1e-3
    for _ in range(args.sparse_steps):
        x, y = sample()
        logits, scores, teacher, valid = model(x, mode="sparse")
        ce = lm_loss(logits, y)
        kl = index_kl(scores, teacher, valid)
        main_optimizer.zero_grad(set_to_none=True)
        index_optimizer.zero_grad(set_to_none=True)
        # 两个损失虽然相加，计算图已分离：CE只到主干，KL只到indexer。
        (ce + kl).backward()
        main_optimizer.step()
        index_optimizer.step()
    print("after sparse adaptation:", probe(model, probe_x, probe_y))
    self_check(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--dense-steps", type=int, default=120)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--sparse-steps", type=int, default=120)
    run(parser.parse_args())
