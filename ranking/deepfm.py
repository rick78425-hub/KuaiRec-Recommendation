"""DeepFM V1/V2 comparison only; V3 remains the default recommender."""
import numpy as np
import pandas as pd
import torch
from torch import nn

class DeepFM(nn.Module):

    def __init__(self, n_num, cat_sizes, emb_dim=8):
        super().__init__()
        self.linear_num = nn.Linear(n_num, 1, bias=False)
        self.num_emb = nn.Parameter(torch.randn(n_num, emb_dim) * 0.01)
        self.linear_cat = nn.ModuleList([nn.Embedding(n, 1) for n in cat_sizes])
        self.cat_emb = nn.ModuleList([nn.Embedding(n, emb_dim) for n in cat_sizes])
        self.bias = nn.Parameter(torch.zeros(1))
        for emb in self.cat_emb:
            nn.init.normal_(emb.weight, std=0.01)
        for emb in self.linear_cat:
            nn.init.zeros_(emb.weight)
        input_dim = (n_num + len(cat_sizes)) * emb_dim + n_num
        self.dnn = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x_num, x_cat):
        num_emb = x_num.unsqueeze(-1) * self.num_emb.unsqueeze(0)
        cat_emb = torch.stack([emb(x_cat[:, i]) for i, emb in enumerate(self.cat_emb)], dim=1)
        embeddings = torch.cat([num_emb, cat_emb], dim=1)
        linear = self.linear_num(x_num) + self.bias
        linear += sum((emb(x_cat[:, i]) for i, emb in enumerate(self.linear_cat)))
        fm = 0.5 * (embeddings.sum(dim=1).square() - embeddings.square().sum(dim=1))
        fm = fm.sum(dim=1, keepdim=True)
        dnn_input = torch.cat([embeddings.flatten(1), x_num], dim=1)
        deep = self.dnn(dnn_input)
        return (linear + fm + deep).squeeze(1)

class DeepFMComparison:
    def __init__(self, artifact_dir, version="v1"):
        import joblib
        from pathlib import Path
        if version not in {"v1","v2"}:
            raise ValueError("Expected v1 or v2")
        path=Path(artifact_dir)
        self.state=joblib.load(path/"feature_state.joblib")
        self.model=DeepFM(len(self.state["num_cols"]),self.state["cat_sizes"],self.state["emb_dim"])
        self.model.load_state_dict(torch.load(path/f"deepfm_{version}.pt",map_location="cpu",weights_only=True))
        self.model.eval()

    def transform(self, frame):
        state=self.state
        numeric=frame[state["num_cols"]].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan).fillna(0)
        for col in state["log_cols"]:
            numeric[col]=np.log1p(numeric[col].clip(lower=0))
        numeric=numeric.astype("float32")
        numeric=((numeric-state["num_mean"])/state["num_std"]).to_numpy(dtype=np.float32)
        cats=[]
        for col in state["cat_cols"]:
            values=pd.to_numeric(frame[col].astype("object"),errors="coerce").fillna(-1).astype("int32")
            cats.append(values.map(state["cat_vocab"][col]).fillna(0).to_numpy(dtype=np.int64))
        return numeric,np.column_stack(cats)

    def predict(self, frame, batch_size=4096):
        numeric,cats=self.transform(frame)
        parts=[]
        with torch.inference_mode():
            for start in range(0,len(frame),batch_size):
                logits=self.model(torch.from_numpy(numeric[start:start+batch_size]),torch.from_numpy(cats[start:start+batch_size]))
                parts.append(torch.sigmoid(logits).numpy())
        return np.concatenate(parts) if parts else np.array([],dtype=np.float32)
