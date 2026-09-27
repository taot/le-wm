import torch
import stable_pretraining as spt
from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg

D = 192
# 和 config/train/model/lewm.yaml 相同的组件，只是 image_size 缩小、depth 截短
encoder = spt.backbone.utils.vit_hf(size="tiny", patch_size=14,
    image_size=56, pretrained=False, use_mask_token=False)
predictor = ARPredictor(num_frames=3, depth=2, heads=8, mlp_dim=512,
    input_dim=D, hidden_dim=D, output_dim=D)
model = JEPA(encoder, predictor,
    Embedder(input_dim=2, emb_dim=D),
    MLP(input_dim=D, output_dim=D, hidden_dim=512, norm_fn=torch.nn.BatchNorm1d),
    MLP(input_dim=D, output_dim=D, hidden_dim=512, norm_fn=torch.nn.BatchNorm1d))

B, T, H = 2, 4, 3  # batch, 时间步, history_size
info = {"pixels": torch.rand(B, T, 3, 56, 56), "action": torch.rand(B, T, 2)}
out = model.encode(info)
emb, act_emb = out["emb"], out["act_emb"]
assert emb.shape == (B, T, D)

pred = model.predict(emb[:, :H], act_emb[:, :H])  # (B, H, D)
tgt = emb[:, 1:]                                 # num_preds=1
# 和 train.py 里 lejepa_forward 完全一致的两项 loss
pred_loss = (pred - tgt).pow(2).mean()
sigreg_loss = SIGReg()(emb.transpose(0, 1))      # SIGReg 吃 (T, B, D)
loss = pred_loss + 0.09 * sigreg_loss
loss.backward()
print(f"pred={pred_loss.item():.4f} sigreg={sigreg_loss.item():.4f} total={loss.item():.4f}")
