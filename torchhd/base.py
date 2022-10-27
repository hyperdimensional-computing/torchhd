import torch
from typing import List
from torch import Tensor


class VSA_Model(Tensor):
    @classmethod
    def empty_hv(
        cls,
        num_vectors: int,
        dimensions: int,
        *,
        dtype=None,
        device=None,
    ) -> "VSA_Model":
        raise NotImplementedError

    @classmethod
    def identity_hv(
        cls,
        num_vectors: int,
        dimensions: int,
        *,
        dtype=None,
        device=None,
    ) -> "VSA_Model":
        raise NotImplementedError

    @classmethod
    def random_hv(
        cls,
        num_vectors: int,
        dimensions: int,
        *,
        dtype=None,
        device=None,
        generator=None,
    ) -> "VSA_Model":
        raise NotImplementedError

    def bundle(self, other: "VSA_Model") -> "VSA_Model":
        raise NotImplementedError

    def multibundle(self) -> "VSA_Model":
        if self.dim() < 2:
            class_name = self.__class__.__name__
            raise RuntimeError(
                f"{class_name} needs to have at least two dimensions for multibundle, got size: {tuple(self.shape)}"
            )

        n = self.size(-2)
        if n == 1:
            return self.unsqueeze(-2)

        tensors: List[VSA_Model] = torch.unbind(self, dim=-2)
        print(type(tensors[0]))

        output = tensors[0].bundle(tensors[1])
        for i in range(2, n):
            output = output.bundle(tensors[i])

        return output

    def bind(self, other: "VSA_Model") -> "VSA_Model":
        raise NotImplementedError

    def multibind(self) -> "VSA_Model":
        if self.dim() < 2:
            class_name = self.__class__.__name__
            raise RuntimeError(
                f"{class_name} data needs to have at least two dimensions for multibind, got size: {tuple(self.shape)}"
            )

        n = self.size(-2)
        if n == 1:
            return self.unsqueeze(-2)

        tensors: List[VSA_Model] = torch.unbind(self, dim=-2)

        output = tensors[0].bind(tensors[1])
        for i in range(2, n):
            output = output.bind(tensors[i])

        return output

    def inverse(self) -> "VSA_Model":
        raise NotImplementedError

    def negative(self) -> "VSA_Model":
        raise NotImplementedError

    def permute(self, n: int = 1) -> "VSA_Model":
        raise NotImplementedError

    def dot_similarity(self, others: "VSA_Model") -> Tensor:
        raise NotImplementedError

    def cos_similarity(self, others: "VSA_Model") -> Tensor:
        raise NotImplementedError
