import pytest
import torch

from kaira.models import BaseModel, ModelRegistry


class DummyModel(BaseModel):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.hidden_size = hidden_size
    
    def forward(self, x):
        return x
    
    def bandwidth_ratio(self):
        return 1.0


def test_model_registry_register():
    """Test registering a model with the ModelRegistry."""
    # Clear existing registrations for this test
    original_models = ModelRegistry._models.copy()
    ModelRegistry._models.clear()
    
    try:
        # Register a new model
        ModelRegistry.register("dummy", DummyModel)
        assert "dummy" in ModelRegistry._models
        assert ModelRegistry._models["dummy"] == DummyModel
    finally:
        # Restore original models
        ModelRegistry._models = original_models


def test_model_registry_register_decorator():
    """Test using register_model decorator."""
    original_models = ModelRegistry._models.copy()
    ModelRegistry._models.clear()
    
    try:
        # Define and register a model using decorator
        @ModelRegistry.register_model("decorator_test")
        class TestModel(BaseModel):
            def forward(self, x):
                return x
            
            def bandwidth_ratio(self):
                return 1.0
        
        # Check registration
        assert "decorator_test" in ModelRegistry._models
        assert ModelRegistry._models["decorator_test"] == TestModel
        
        # Test with default name
        @ModelRegistry.register_model()
        class ImplicitNameModel(BaseModel):
            def forward(self, x):
                return x
            
            def bandwidth_ratio(self):
                return 1.0
        
        # Should use class name (lowercase)
        assert "implicitnamemodel" in ModelRegistry._models
    finally:
        # Restore original models
        ModelRegistry._models = original_models


def test_model_registry_create():
    """Test creating a model instance from the registry."""
    original_models = ModelRegistry._models.copy()
    ModelRegistry._models.clear()
    
    try:
        # Register a model and create an instance
        ModelRegistry.register("test_param", DummyModel)
        model = ModelRegistry.create("test_param", hidden_size=128)
        
        # Verify the instance
        assert isinstance(model, DummyModel)
        assert model.hidden_size == 128
        
        # Test with non-existent model
        with pytest.raises(ValueError):
            ModelRegistry.create("nonexistent_model")
    finally:
        # Restore original models
        ModelRegistry._models = original_models


def test_model_registry_list_models():
    """Test listing registered models."""
    original_models = ModelRegistry._models.copy()
    ModelRegistry._models.clear()
    
    try:
        ModelRegistry.register("model1", DummyModel)
        ModelRegistry.register("model2", DummyModel)
        
        models = ModelRegistry.list_models()
        assert "model1" in models
        assert "model2" in models
        assert len(models) == 2
    finally:
        # Restore original models
        ModelRegistry._models = original_models


def test_model_registry_get_model_info():
    """Test getting model info."""
    original_models = ModelRegistry._models.copy()
    ModelRegistry._models.clear()
    
    try:
        ModelRegistry.register("info_test", DummyModel)
        
        # Get info
        info = ModelRegistry.get_model_info("info_test")
        assert info["name"] == "info_test"
        assert info["class"] == DummyModel.__name__
        assert "signature" in info
        
        # Test with non-existent model
        with pytest.raises(ValueError):
            ModelRegistry.get_model_info("nonexistent")
    finally:
        # Restore original models
        ModelRegistry._models = original_models
