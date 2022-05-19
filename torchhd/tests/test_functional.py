import pytest
import torch

from .. import functional


class TestIdentity_hv:
    def test_shape(self):
        hv = functional.identity_hv(13, 2556)

        assert hv.dim() == 2
        assert hv.size(0) == 13
        assert hv.size(1) == 2556

    def test_value(self):
        hv = functional.identity_hv(4, 85)
        assert (hv == 1).min().item()

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hv = functional.identity_hv(3, 52, device=device)

        assert hv.device == device

    def test_dtype(self):
        hv = functional.identity_hv(3, 52)
        assert hv.dtype == torch.get_default_dtype()

        hv = functional.identity_hv(3, 52, dtype=torch.int)
        assert hv.dtype == torch.int

        hv = functional.identity_hv(3, 52, dtype=torch.bool)
        assert hv.dtype == torch.bool

        hv = functional.identity_hv(3, 52, dtype=torch.float)
        assert hv.dtype == torch.float

    def test_requires_grad(self):
        hv = functional.identity_hv(3, 52, requires_grad=False)
        assert hv.requires_grad == False

        hv = functional.identity_hv(3, 52, requires_grad=True)
        assert hv.requires_grad == True

    def test_integration(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hv = functional.identity_hv(
            6, 10000, dtype=torch.float16, requires_grad=True, device=device
        )

        assert hv.dim() == 2
        assert hv.size(0) == 6
        assert hv.size(1) == 10000
        assert hv.requires_grad == True
        assert hv.dtype == torch.float16
        assert hv.device == device


class TestRandom_hv:
    def test_shape(self):
        hv = functional.random_hv(13, 2556)

        assert hv.dim() == 2
        assert hv.size(0) == 13
        assert hv.size(1) == 2556

    def test_value(self):
        generator = torch.Generator()
        generator.manual_seed(2147483644)
        hv = functional.random_hv(4, 10000, generator=generator)

        assert ((hv == -1) | (hv == 1)).min().item()

        sim = functional.cosine_similarity(hv[0], hv[3].unsqueeze(0))
        assert sim.abs().item() < 0.015
        sim = functional.cosine_similarity(hv[2], hv[3].unsqueeze(0))
        assert sim.abs().item() < 0.015

    def test_generator(self):
        generator = torch.Generator()
        generator.manual_seed(2147483644)
        hv1 = functional.random_hv(20, 10000, generator=generator)

        generator = torch.Generator()
        generator.manual_seed(2147483644)
        hv2 = functional.random_hv(20, 10000, generator=generator)

        assert (hv1 == hv2).min().item()

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hv = functional.random_hv(3, 52, device=device)

        assert hv.device == device

    def test_dtype(self):
        hv = functional.random_hv(3, 52)
        assert hv.dtype == torch.get_default_dtype()

        hv = functional.random_hv(3, 52, dtype=torch.int)
        assert hv.dtype == torch.int

        hv = functional.random_hv(3, 52, dtype=torch.bool)
        assert hv.dtype == torch.bool

        hv = functional.random_hv(3, 52, dtype=torch.float)
        assert hv.dtype == torch.float

    def test_requires_grad(self):
        hv = functional.random_hv(3, 52, dtype=torch.long, requires_grad=False)
        assert hv.requires_grad == False

        hv = functional.random_hv(3, 52, dtype=torch.float, requires_grad=True)
        assert hv.requires_grad == True

        with pytest.raises(RuntimeError):
            hv = functional.random_hv(3, 52, dtype=torch.long, requires_grad=True)

    def test_integration(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hv = functional.random_hv(
            6, 10000, dtype=torch.float, requires_grad=True, device=device
        )

        assert hv.dim() == 2
        assert hv.size(0) == 6
        assert hv.size(1) == 10000
        assert hv.requires_grad == True
        assert hv.dtype == torch.float
        assert hv.device == device


class TestLevel_hv:
    def test_shape(self):
        hv = functional.level_hv(13, 2556)

        assert hv.dim() == 2
        assert hv.size(0) == 13
        assert hv.size(1) == 2556

    def test_value(self):
        generator = torch.Generator()
        generator.manual_seed(2147483644)
        hv = functional.level_hv(50, 10000, generator=generator)

        assert ((hv == -1) | (hv == 1)).min().item(), "values are either -1 or +1"

        sim = functional.cosine_similarity(hv[0], hv[49].unsqueeze(0))
        assert sim.abs().item() < 0.015
        sim = functional.cosine_similarity(hv[0], hv[1].unsqueeze(0))
        assert sim.abs().item() > 0.98
        sim = functional.cosine_similarity(hv[0], hv[24].unsqueeze(0))
        assert sim.abs().item() > 0.47
        sim = functional.cosine_similarity(hv[0], hv[24].unsqueeze(0))
        assert sim.abs().item() < 0.52
        sim = functional.cosine_similarity(hv[40], hv[41].unsqueeze(0))
        assert sim.abs().item() > 0.98

    def test_generator(self):
        generator = torch.Generator()
        generator.manual_seed(2147483644)
        hv1 = functional.level_hv(60, 10000, generator=generator)

        generator = torch.Generator()
        generator.manual_seed(2147483644)
        hv2 = functional.level_hv(60, 10000, generator=generator)

        assert (hv1 == hv2).min().item()

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hv = functional.level_hv(3, 52, device=device)

        assert hv.device == device

    def test_dtype(self):
        hv = functional.level_hv(3, 52)
        assert hv.dtype == torch.get_default_dtype()

        hv = functional.level_hv(3, 52, dtype=torch.int)
        assert hv.dtype == torch.int

        hv = functional.level_hv(3, 52, dtype=torch.bool)
        assert hv.dtype == torch.bool

        hv = functional.level_hv(3, 52, dtype=torch.float)
        assert hv.dtype == torch.float

    def test_requires_grad(self):
        hv = functional.level_hv(3, 52, dtype=torch.long, requires_grad=False)
        assert hv.requires_grad == False

        hv = functional.level_hv(3, 52, dtype=torch.float, requires_grad=True)
        assert hv.requires_grad == True

        with pytest.raises(RuntimeError):
            hv = functional.level_hv(3, 52, dtype=torch.long, requires_grad=True)

    def test_integration(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hv = functional.level_hv(
            6, 10000, dtype=torch.float, requires_grad=True, device=device
        )

        assert hv.dim() == 2
        assert hv.size(0) == 6
        assert hv.size(1) == 10000
        assert hv.requires_grad == True
        assert hv.dtype == torch.float
        assert hv.device == device


class TestCircular_hv:
    def test_shape(self):
        hv = functional.circular_hv(13, 2556)

        assert hv.dim() == 2
        assert hv.size(0) == 13
        assert hv.size(1) == 2556

    def test_value(self):
        generator = torch.Generator()
        generator.manual_seed(2147483644)
        hv = functional.circular_hv(50, 10000, generator=generator)

        assert ((hv == -1) | (hv == 1)).min().item(), "values are either -1 or +1"

        sim = functional.cosine_similarity(hv[0], hv[25].unsqueeze(0))
        assert sim.abs().item() < 0.015
        sim = functional.cosine_similarity(hv[0], hv[1].unsqueeze(0))
        assert sim.abs().item() > 0.95
        sim = functional.cosine_similarity(hv[0], hv[49].unsqueeze(0))
        assert sim.abs().item() > 0.95
        sim = functional.cosine_similarity(hv[0], hv[12].unsqueeze(0))
        assert sim.abs().item() > 0.47
        sim = functional.cosine_similarity(hv[0], hv[37].unsqueeze(0))
        assert sim.abs().item() > 0.47
        sim = functional.cosine_similarity(hv[0], hv[12].unsqueeze(0))
        assert sim.abs().item() < 0.54
        sim = functional.cosine_similarity(hv[0], hv[37].unsqueeze(0))
        assert sim.abs().item() < 0.54
        sim = functional.cosine_similarity(hv[40], hv[41].unsqueeze(0))
        assert sim.abs().item() > 0.96

    def test_generator(self):
        generator = torch.Generator()
        generator.manual_seed(2147483644)
        hv1 = functional.circular_hv(60, 10000, generator=generator)

        generator = torch.Generator()
        generator.manual_seed(2147483644)
        hv2 = functional.circular_hv(60, 10000, generator=generator)

        assert (hv1 == hv2).min().item()

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hv = functional.circular_hv(3, 52, device=device)

        assert hv.device == device

    def test_dtype(self):
        hv = functional.circular_hv(3, 52)
        assert hv.dtype == torch.get_default_dtype()

        hv = functional.circular_hv(3, 52, dtype=torch.int)
        assert hv.dtype == torch.int

        hv = functional.circular_hv(3, 52, dtype=torch.bool)
        assert hv.dtype == torch.bool

        hv = functional.circular_hv(3, 52, dtype=torch.float)
        assert hv.dtype == torch.float

    def test_requires_grad(self):
        hv = functional.circular_hv(3, 52, dtype=torch.long, requires_grad=False)
        assert hv.requires_grad == False

        hv = functional.circular_hv(3, 52, dtype=torch.float, requires_grad=True)
        assert hv.requires_grad == True

        with pytest.raises(RuntimeError):
            hv = functional.circular_hv(3, 52, dtype=torch.long, requires_grad=True)

    def test_integration(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hv = functional.circular_hv(
            6, 10000, dtype=torch.float, requires_grad=True, device=device
        )

        assert hv.dim() == 2
        assert hv.size(0) == 6
        assert hv.size(1) == 10000
        assert hv.requires_grad == True
        assert hv.dtype == torch.float
        assert hv.device == device


class TestBind:
    def test_value(self):
        hv = functional.random_hv(2, 100)
        res = functional.bind(hv[0], hv[1])

        assert res.dtype == hv.dtype
        assert res.dim() == 1
        assert res.size(0) == 100
        assert ((hv == -1) | (hv == 1)).min().item(), "values are either -1 or +1"

    def test_out(self):
        hv = functional.random_hv(2, 100)
        buffer = torch.empty(100)
        res = functional.bind(hv[0], hv[1], out=buffer)

        assert res.data_ptr() == buffer.data_ptr()
        assert res.dtype == hv.dtype
        assert res.dim() == 1
        assert res.size(0) == 100
        assert ((hv == -1) | (hv == 1)).min().item(), "values are either -1 or +1"

    def test_dtype(self):
        hv = functional.random_hv(2, 100, dtype=torch.long)
        buffer = torch.empty(100, dtype=torch.long)
        res = functional.bind(hv[0], hv[1], out=buffer)

        assert res.data_ptr() == buffer.data_ptr()
        assert res.dtype == hv.dtype
        assert res.dim() == 1
        assert res.size(0) == 100
        assert ((hv == -1) | (hv == 1)).min().item(), "values are either -1 or +1"

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hv = functional.random_hv(2, 100, device=device)
        res = functional.bind(hv[0], hv[1])

        assert res.dtype == hv.dtype
        assert res.dim() == 1
        assert res.size(0) == 100
        assert ((hv == -1) | (hv == 1)).min().item(), "values are either -1 or +1"
        assert res.device == device


class TestBundle:
    def test_value(self):
        hv = functional.random_hv(2, 100)
        res = functional.bundle(hv[0], hv[1])

        assert res.dtype == hv.dtype
        assert res.dim() == 1
        assert res.size(0) == 100
        assert ((hv <= 2) & (hv >= -2)).min().item(), "values are between -2 and +2"

    def test_out(self):
        hv = functional.random_hv(2, 100)
        buffer = torch.empty(100)
        res = functional.bundle(hv[0], hv[1], out=buffer)

        assert res.data_ptr() == buffer.data_ptr()
        assert res.dtype == hv.dtype
        assert res.dim() == 1
        assert res.size(0) == 100
        assert ((hv <= 2) & (hv >= -2)).min().item(), "values are between -2 and +2"

    def test_dtype(self):
        hv = functional.random_hv(2, 100, dtype=torch.long)
        buffer = torch.empty(100, dtype=torch.long)
        res = functional.bundle(hv[0], hv[1], out=buffer)

        assert res.data_ptr() == buffer.data_ptr()
        assert res.dtype == hv.dtype
        assert res.dim() == 1
        assert res.size(0) == 100
        assert ((hv <= 2) & (hv >= -2)).min().item(), "values are between -2 and +2"

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hv = functional.random_hv(2, 100, device=device)
        res = functional.bundle(hv[0], hv[1])

        assert res.dtype == hv.dtype
        assert res.dim() == 1
        assert res.size(0) == 100
        assert ((hv <= 2) & (hv >= -2)).min().item(), "values are between -2 and +2"
        assert res.device == device


class TestPermute:
    def test_value(self):
        hv = functional.random_hv(2, 100)
        res = functional.permute(hv[0])

        assert res.dtype == hv.dtype
        assert res.dim() == 1
        assert res.size(0) == 100
        assert ((hv == -1) | (hv == 1)).min().item(), "values are either -1 or +1"
        assert torch.sum(res == hv[0]) != res.size(
            0
        ), "all element must not be the same"

        one_shift = functional.permute(hv[0])
        two_shift = functional.permute(hv[0], shifts=2)
        assert torch.sum(one_shift == two_shift) != res.size(
            0
        ), "all element must not be the same"

    def test_dtype(self):
        hv = functional.random_hv(2, 100, dtype=torch.long)
        res = functional.permute(hv[0])

        assert res.dtype == hv.dtype
        assert res.dim() == 1
        assert res.size(0) == 100
        assert ((hv == -1) | (hv == 1)).min().item(), "values are either -1 or +1"

    def test_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hv = functional.random_hv(2, 100, device=device)
        res = functional.permute(hv[0])

        assert res.dtype == hv.dtype
        assert res.dim() == 1
        assert res.size(0) == 100
        assert ((hv == -1) | (hv == 1)).min().item(), "values are either -1 or +1"
        assert res.device == device
