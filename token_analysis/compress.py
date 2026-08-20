"""비주얼 토큰 압축 방법 모음.

모든 함수는 (F, N, D) 텐서를 받는다 — F 프레임, 프레임당 N 토큰, D 차원.
반환도 (F', N', D) — LLaVA 계열은 토큰이 1D로 삽입되므로 F'·N'이 얼마든 동작한다.

방법 축:
  토큰 수 축소: random / pool_avg / pool_max / pca_select / tome / kmeans
  프레임 수 축소: temporal_pool / framediff
  정보량 축소(토큰 수 유지): pca_recon  ← rank ablation, 다른 방법들과 직교하는 축
"""

import math

import torch


def _grid(n):
    s = int(math.isqrt(n))
    assert s * s == n, f"토큰 수 {n}이 정사각 격자가 아님"
    return s


def none(x, keep):
    return x


def random_drop(x, keep):
    F, N, D = x.shape
    k = max(1, int(N * keep))
    idx = torch.stack([torch.randperm(N, device=x.device)[:k].sort().values for _ in range(F)])
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, D))


def _pool2d(x, keep, mode):
    F, N, D = x.shape
    s = _grid(N)
    target = max(1, int(round(math.sqrt(N * keep))))
    g = x.view(F, s, s, D).permute(0, 3, 1, 2)  # F, D, s, s
    if mode == "avg":
        g = torch.nn.functional.adaptive_avg_pool2d(g, target)
    else:
        g = torch.nn.functional.adaptive_max_pool2d(g, target)
    return g.permute(0, 2, 3, 1).reshape(F, target * target, D)


def pool_avg(x, keep):
    return _pool2d(x, keep, "avg")


def pool_max(x, keep):
    return _pool2d(x, keep, "max")


def temporal_pool(x, keep):
    """인접 프레임 평균으로 프레임 수를 줄임 (프레임당 토큰은 유지)."""
    F, N, D = x.shape
    t = max(1, int(F * keep))
    group = F / t
    outs = []
    for i in range(t):
        a, b = int(i * group), max(int((i + 1) * group), int(i * group) + 1)
        outs.append(x[a:b].mean(dim=0))
    return torch.stack(outs)


def framediff(x, keep):
    """변화가 큰 프레임만 유지 (첫 프레임은 항상 유지) — temporal 중요도 기반."""
    F, N, D = x.shape
    t = max(1, int(F * keep))
    if t >= F:
        return x
    diff = (x[1:] - x[:-1]).norm(dim=-1).mean(dim=-1)  # F-1
    keep_idx = diff.topk(t - 1).indices + 1 if t > 1 else torch.tensor([], dtype=torch.long, device=x.device)
    idx = torch.cat([torch.tensor([0], device=x.device), keep_idx]).sort().values
    return x[idx]


def pca_select(x, keep):
    """프레임별 PCA leverage score 상위 토큰만 유지."""
    F, N, D = x.shape
    k = max(1, int(N * keep))
    outs = []
    for f in range(F):
        feat = x[f].float()
        c = feat - feat.mean(dim=0, keepdim=True)
        q = min(k, min(c.shape) - 1, 32)
        _, _, v = torch.pca_lowrank(c, q=max(q, 2))
        lever = (c @ v).pow(2).sum(dim=-1)  # 주성분 공간에서의 에너지
        idx = lever.topk(k).indices.sort().values
        outs.append(x[f][idx])
    return torch.stack(outs)


def pca_recon(x, keep):
    """토큰 수는 유지, 상위 주성분 rank만 남겨 재구성 — 정보량 축소 축."""
    F, N, D = x.shape
    q = max(2, int(min(N, D) * keep))
    outs = []
    for f in range(F):
        feat = x[f].float()
        mu = feat.mean(dim=0, keepdim=True)
        c = feat - mu
        _, _, v = torch.pca_lowrank(c, q=q)
        outs.append(((c @ v) @ v.T + mu).to(x.dtype))
    return torch.stack(outs)


def tome(x, keep):
    """ToMe식 bipartite soft matching을 프레임별로 반복 적용해 k개까지 병합."""
    F, N, D = x.shape
    k = max(1, int(N * keep))
    outs = []
    for f in range(F):
        feat = x[f]
        size = torch.ones(feat.shape[0], 1, device=feat.device, dtype=feat.dtype)
        while feat.shape[0] > k:
            n = feat.shape[0]
            r = min(n // 2, n - k)
            a, b = feat[0::2], feat[1::2]
            sa, sb = size[0::2], size[1::2]
            an = torch.nn.functional.normalize(a.float(), dim=-1)
            bn = torch.nn.functional.normalize(b.float(), dim=-1)
            sim = an @ bn.T  # (na, nb)
            best_val, best_dst = sim.max(dim=-1)
            merge_src = best_val.topk(r).indices
            keep_mask = torch.ones(a.shape[0], dtype=torch.bool, device=feat.device)
            keep_mask[merge_src] = False
            # 병합: src를 대응 dst(b쪽)에 크기 가중 평균으로 흡수
            b = b.clone()
            sb = sb.clone()
            for s_i in merge_src.tolist():
                d_i = best_dst[s_i].item()
                tot = sa[s_i] + sb[d_i]
                b[d_i] = (b[d_i] * sb[d_i] + a[s_i] * sa[s_i]) / tot
                sb[d_i] = tot
            feat = torch.cat([a[keep_mask], b])
            size = torch.cat([sa[keep_mask], sb])
        outs.append(feat)
    return torch.stack(outs)


def kmeans(x, keep, iters=8):
    """프레임별 k-means 센터로 대체."""
    F, N, D = x.shape
    k = max(1, int(N * keep))
    outs = []
    for f in range(F):
        feat = x[f].float()
        centers = feat[torch.randperm(N, device=x.device)[:k]].clone()
        for _ in range(iters):
            assign = torch.cdist(feat, centers).argmin(dim=-1)
            for c_i in range(k):
                m = assign == c_i
                if m.any():
                    centers[c_i] = feat[m].mean(dim=0)
        outs.append(centers.to(x.dtype))
    return torch.stack(outs)


METHODS = {
    "none": none,
    "random": random_drop,
    "pool_avg": pool_avg,
    "pool_max": pool_max,
    "temporal_pool": temporal_pool,
    "framediff": framediff,
    "pca_select": pca_select,
    "pca_recon": pca_recon,
    "tome": tome,
    "kmeans": kmeans,
}


def compress(x, method, keep):
    """x: (F, N, D), keep: 유지 비율 (0~1]. 반환 (F', N', D)."""
    with torch.no_grad():
        return METHODS[method](x, keep)
