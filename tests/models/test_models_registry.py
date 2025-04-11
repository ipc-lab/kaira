import pytest

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

        # Test registering a model with the same name (should raise ValueError)
        with pytest.raises(ValueError, match="Model 'dummy' is already registered"):
            ModelRegistry.register("dummy", DummyModel)

        # Test registering a non-BaseModel class (should raise TypeError)
        class NotAModel:
            pass

        with pytest.raises(TypeError, match="Model class NotAModel must inherit from BaseModel"):
            ModelRegistry.register("not_a_model", NotAModel)
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
        with pytest.raises(KeyError, match="Model 'nonexistent_model' not found in registry"):
            ModelRegistry.create("nonexistent_model")

        # Test with invalid constructor arguments
        with pytest.raises(TypeError, match="Failed to create model 'test_param'"):
            ModelRegistry.create("test_param", invalid_param=42)
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


def test_model_registry_get():
    """Test getting a model class from the registry."""
    original_models = ModelRegistry._models.copy()
    ModelRegistry._models.clear()

    try:
        # Register a model
        ModelRegistry.register("get_test", DummyModel)

        # Get the model class
        model_class = ModelRegistry.get("get_test")
        assert model_class == DummyModel

        # Test with non-existent model
        with pytest.raises(KeyError, match="Model 'nonexistent' not found in registry"):
            ModelRegistry.get("nonexistent")
    finally:
        # Restore original models
        ModelRegistry._models = original_models


def test_model_registry_comprehensive():
    """Comprehensive test to ensure full coverage of the ModelRegistry class."""
    original_models = ModelRegistry._models.copy()
    ModelRegistry._models.clear()

    try:
        # Register a model class
        @ModelRegistry.register_model()
        class CompModel(BaseModel):
            def __init__(self, param1=10, param2=20):
                super().__init__()
                self.param1 = param1
                self.param2 = param2

            def forward(self, x):
                return x * self.param1 + self.param2

        # Test all aspects of the registry
        # 1. Registration worked
        assert "compmodel" in ModelRegistry._models

        # 2. Get works
        model_class = ModelRegistry.get("compmodel")
        assert model_class == CompModel

        # 3. Create works with default params
        instance = ModelRegistry.create("compmodel")
        assert instance.param1 == 10
        assert instance.param2 == 20

        # 4. Create works with custom params
        custom_instance = ModelRegistry.create("compmodel", param1=30, param2=40)
        assert custom_instance.param1 == 30
        assert custom_instance.param2 == 40

        # 5. List models includes our model
        models_list = ModelRegistry.list_models()
        assert "compmodel" in models_list

    finally:
        # Restore original models
        ModelRegistry._models = original_models
