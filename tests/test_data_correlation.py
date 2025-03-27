import pytest
import torch
from kaira.data.correlation import WynerZivCorrelationModel, WynerZivCorrelationDataset

@pytest.fixture
def source_tensor():
    return torch.randint(0, 2, (100, 20)).float()

def test_wyner_ziv_correlation_model_gaussian():
    model = WynerZivCorrelationModel(correlation_type="gaussian", correlation_params={"sigma": 0.5})
    source = torch.randn(1000)
    correlated = model(source)
    assert correlated.shape == source.shape
    assert torch.allclose(correlated.mean(), source.mean(), atol=0.1)
    assert torch.allclose(correlated.std(), (source.std()**2 + 0.5**2)**0.5, atol=0.1)

def test_wyner_ziv_correlation_model_binary():
    model = WynerZivCorrelationModel(correlation_type="binary", correlation_params={"crossover_prob": 0.2})
    source = torch.randint(0, 2, (1000,)).float()
    correlated = model(source)
    assert correlated.shape == source.shape
    assert torch.all((correlated == 0) | (correlated == 1))
    expected_corr = 1 - 2 * 0.2
    empirical_corr = 1 - 2 * ((source != correlated).float().mean().item())
    assert abs(empirical_corr - expected_corr) < 0.05

def test_wyner_ziv_correlation_model_custom():
    def custom_transform(x):
        return x * 2
    model = WynerZivCorrelationModel(correlation_type="custom", correlation_params={"transform_fn": custom_transform})
    source = torch.randn(1000)
    correlated = model(source)
    assert correlated.shape == source.shape
    assert torch.allclose(correlated, source * 2)

def test_wyner_ziv_correlation_dataset(source_tensor):
    dataset = WynerZivCorrelationDataset(source=source_tensor, correlation_type="binary", correlation_params={"crossover_prob": 0.15})
    assert len(dataset) == 100
    x, y = dataset[0]
    assert x.shape == torch.Size([20])
    assert y.shape == torch.Size([20])
    assert torch.all((x == 0) | (x == 1))
    assert torch.all((y == 0) | (y == 1))
    batch_x, batch_y = dataset[10:20]
    assert batch_x.shape == torch.Size([10, 20])
    assert batch_y.shape == torch.Size([10, 20])

def test_wyner_ziv_correlation_model_unknown_type(source_tensor):
    model = WynerZivCorrelationModel(correlation_type="unknown")
    with pytest.raises(ValueError, match="Unknown correlation type"):
        model(source_tensor)

def test_wyner_ziv_correlation_model_missing_transform_fn(source_tensor):
    model = WynerZivCorrelationModel(correlation_type="custom", correlation_params={})
    with pytest.raises(ValueError, match="requires 'transform_fn' parameter"):
        model(source_tensor)
