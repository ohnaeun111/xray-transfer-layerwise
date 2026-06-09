import torch
import torch.nn as nn

from config import LAYERS, MODELS, IMG_SIZE


# ─────────────────────────────────────────────────────────────
# 공통 베이스
# ─────────────────────────────────────────────────────────────
class BackboneWrapper(nn.Module):
    """모든 backbone wrapper의 부모. extract()를 하위 클래스가 구현."""
    def __init__(self, model, embed_dim):
        super().__init__()
        self.model = model
        self.embed_dim = embed_dim
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def extract(self, x) -> dict:
        """x:[B,3,H,W] -> {layer_label: [B, embed_dim]} (CLS token)."""
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────
# timm 계열 (ImageNet ViT, DINOv2)
# ─────────────────────────────────────────────────────────────
class TimmWrapper(BackboneWrapper):
    """
    timm ViT는 get_intermediate_layers(x, n, return_class_token=True)를
    지원한다. n에 0-base block 인덱스 집합을 주면 각 block 출력의
    (patch_tokens, class_token) 튜플 리스트를 돌려준다.
    """
    def __init__(self, model_id):
        import timm
        model = timm.create_model(model_id, pretrained=True, num_classes=0, img_size=IMG_SIZE)
        embed_dim = model.embed_dim
        super().__init__(model, embed_dim)
        # config의 L3.. 라벨을 인덱스 오름차순으로 정렬해 보관
        self._labels = sorted(LAYERS, key=lambda k: LAYERS[k])
        self._indices = [LAYERS[k] for k in self._labels]

    @torch.no_grad()
    def extract(self, x):
        outs = self.model.get_intermediate_layers(
            x, n=self._indices, return_prefix_tokens=True, norm=True,
        )
        # outs: 길이 len(indices), 각 원소 (patch_tokens[B,N,D], prefix[B,1,D])
        feats = {}
        for label, (_, prefix) in zip(self._labels, outs):
            feats[label] = prefix[:, 0, :].float().cpu()  # CLS = prefix[:, 0]
        return feats


# ─────────────────────────────────────────────────────────────
# HuggingFace 계열 (DINOv3)
# ─────────────────────────────────────────────────────────────
class HFWrapper(BackboneWrapper):
    """
    HF DINOv3는 output_hidden_states=True 로 부르면 hidden_states 튜플을
    돌려준다. hidden_states[0]은 embedding 출력, [i]는 i번째 block 출력.
    각 hidden state는 [B, num_tokens, dim] 이고 token 0 = CLS.
    (CLS 다음 register 4개가 오지만 CLS만 쓰므로 무관.)
    """
    def __init__(self, model_id):
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_id)
        embed_dim = model.config.hidden_size
        super().__init__(model, embed_dim)
        self._labels = sorted(LAYERS, key=lambda k: LAYERS[k])
        # hidden_states[0]=embeddings 이므로 block i 출력은 hidden_states[i+1]
        self._hs_index = {k: LAYERS[k] + 1 for k in self._labels}

    @torch.no_grad()
    def extract(self, x):
        out = self.model(pixel_values=x, output_hidden_states=True)
        hs = out.hidden_states           # tuple, 길이 = num_blocks+1
        feats = {}
        for label in self._labels:
            cls = hs[self._hs_index[label]][:, 0, :]   # token 0 = CLS
            feats[label] = cls.float().cpu()
        return feats


# ─────────────────────────────────────────────────────────────
# 팩토리
# ─────────────────────────────────────────────────────────────
def load_backbone(model_key: str, device="cuda") -> BackboneWrapper:
    cfg = MODELS[model_key]
    loader = cfg["loader"]
    if loader == "timm":
        w = TimmWrapper(cfg["id"])
    elif loader == "hf":
        w = HFWrapper(cfg["id"])
    else:
        raise ValueError(f"알 수 없는 loader: {loader}")
    return w.to(device)


if __name__ == "__main__":
    # python models.py 로 실행하면 등록된 모델들의 layer 추출 shape를 더미로 검증
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dummy = torch.randn(2, 3, IMG_SIZE, IMG_SIZE, device=device)
    for key in MODELS:
        print(f"\n=== {key} ({MODELS[key]['pretty']}) ===")
        try:
            wb = load_backbone(key, device)
            feats = wb.extract(dummy)
            for label in sorted(feats, key=lambda k: LAYERS[k]):
                print(f"  {label}: {tuple(feats[label].shape)}")
            print(f"  embed_dim = {wb.embed_dim}")
        except Exception as e:
            print(f"  [실패] {type(e).__name__}: {e}")
