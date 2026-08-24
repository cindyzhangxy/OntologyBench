# ontologybench/evaluation/backends/pylate.py

import torch
from typing import Dict, List

from pylate import models
from pylate.utils.tensor import convert_to_tensor


def colbert_maxsim(q_emb: torch.Tensor, d_emb: torch.Tensor) -> torch.Tensor:
    """
    q_emb: (Lq, d)
    d_emb: (D, Ld, d)
    returns: (D,)
    """
    q_emb = torch.nn.functional.normalize(q_emb, dim=-1)
    d_emb = torch.nn.functional.normalize(d_emb, dim=-1)

    # (Lq, D, Ld)
    sim = torch.einsum("qd,dmd->qmd", q_emb, d_emb)

    # max over doc tokens, sum over query tokens
    return sim.max(dim=-1).values.sum(dim=0)


class PyLateReranker:
    """
    ColBERT reranker operating on a fixed candidate set.
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        self.device = device
        self.model = models.ColBERT(model_name_or_path=model_name)
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def rerank(
        self,
        query_text: str,
        candidate_docs: Dict[str, str],
        top_k: int,
    ) -> List[str]:
        """
        Returns ranked doc_ids.
        """

        # Encode query (token-level)
        q_emb = self.model.encode_queries(
            [query_text],
            convert_to_tensor=True,
        )[0].to(self.device)  # (Lq, d)

        doc_ids = list(candidate_docs.keys())
        doc_texts = list(candidate_docs.values())

        d_emb = self.model.encode_documents(
            doc_texts,
            convert_to_tensor=True,
        ).to(self.device)  # (D, Ld, d)

        scores = colbert_maxsim(q_emb, d_emb)
        top_idx = torch.topk(scores, k=min(top_k, len(scores))).indices.tolist()

        return [doc_ids[i] for i in top_idx]
