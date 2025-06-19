==================
Training Models
==================

The Kaira framework provides a comprehensive command-line interface for training communication system models through the ``kaira-train`` console script. This tool offers flexible configuration options and supports various communication-specific parameters.

Overview
========

The ``kaira-train`` command provides:

- **Model Training**: Train any registered communication model
- **Flexible Configuration**: Support for YAML, JSON, and command-line parameters
- **Communication-Specific Features**: SNR ranges, channel types, and noise modeling
- **Hugging Face Hub Integration**: Upload trained models for sharing and distribution
- **Integration**: Works with Hydra configuration management
- **Monitoring**: Built-in logging, evaluation, and checkpointing

Installation
============

The ``kaira-train`` command is automatically available after installing Kaira:

.. code-block:: bash

    pip install -e .

Verify installation:

.. code-block:: bash

    kaira-train --help

Quick Start
===========

List Available Models
----------------------

.. code-block:: bash

    kaira-train --list-models

Basic Training
--------------

.. code-block:: bash

    # Train with default settings
    kaira-train --model deepjscc --output-dir ./results

    # Train with custom parameters
    kaira-train --model deepjscc \\
      --output-dir ./results \\
      --epochs 20 \\
      --batch-size 64 \\
      --learning-rate 1e-3

Advanced Training
-----------------

.. code-block:: bash

    # Communication-specific parameters
    kaira-train --model channel_code \\
      --snr-min 0 \\
      --snr-max 15 \\
      --channel-uses 128 \\
      --channel-type awgn

    # Using configuration files
    kaira-train --model deepjscc \\
      --config-file ./configs/training_example.yaml

    # Resume from checkpoint
    kaira-train --model deepjscc \\
      --resume-from-checkpoint ./results/checkpoint-1000

    # Train and upload to Hugging Face Hub
    kaira-train --model deepjscc \\
      --push-to-hub --hub-model-id username/my-model

Command-Line Reference
======================

Core Arguments
--------------

.. list-table::
   :header-rows: 1
   :widths: 20 10 15 55

   * - Argument
     - Type
     - Default
     - Description
   * - ``--list-models``
     - flag
     - \-
     - List all available models
   * - ``--model``
     - str
     - \-
     - Model name to train (required)
   * - ``--config-file``
     - path
     - \-
     - YAML or JSON configuration file
   * - ``--output-dir``
     - path
     - ``./training_results``
     - Output directory for results

Training Parameters
-------------------

.. list-table::
   :header-rows: 1
   :widths: 20 10 15 55

   * - Argument
     - Type
     - Default
     - Description
   * - ``--epochs``
     - float
     - 10.0
     - Number of training epochs
   * - ``--batch-size``
     - int
     - 32
     - Training batch size per device
   * - ``--eval-batch-size``
     - int
     - 32
     - Evaluation batch size per device
   * - ``--learning-rate``
     - float
     - 1e-4
     - Learning rate
   * - ``--warmup-steps``
     - int
     - 1000
     - Number of warmup steps

Communication Parameters
------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 10 15 55

   * - Argument
     - Type
     - Default
     - Description
   * - ``--snr-min``
     - float
     - 0.0
     - Minimum SNR value
   * - ``--snr-max``
     - float
     - 20.0
     - Maximum SNR value
   * - ``--noise-variance-min``
     - float
     - 0.1
     - Minimum noise variance
   * - ``--noise-variance-max``
     - float
     - 2.0
     - Maximum noise variance
   * - ``--channel-uses``
     - int
     - \-
     - Number of channel uses
   * - ``--code-length``
     - int
     - \-
     - Code length
   * - ``--info-length``
     - int
     - \-
     - Information length
   * - ``--channel-type``
     - str
     - ``awgn``
     - Channel simulation type

Performance Options
-------------------

.. list-table::
   :header-rows: 1
   :widths: 20 10 15 55

   * - Argument
     - Type
     - Default
     - Description
   * - ``--device``
     - str
     - ``auto``
     - Device (auto/cpu/cuda)
   * - ``--fp16``
     - flag
     - False
     - Mixed precision training
   * - ``--dataloader-num-workers``
     - int
     - 0
     - Number of dataloader workers
   * - ``--seed``
     - int
     - 42
     - Random seed

Hugging Face Hub Options
------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 10 15 55

   * - Argument
     - Type
     - Default
     - Description
   * - ``--push-to-hub``
     - flag
     - False
     - Upload trained model to Hugging Face Hub
   * - ``--hub-model-id``
     - str
     - \-
     - Model ID for Hugging Face Hub (e.g., 'username/model-name')
   * - ``--hub-token``
     - str
     - \-
     - Hugging Face Hub authentication token (or set HF_TOKEN env var)
   * - ``--hub-private``
     - flag
     - False
     - Make the Hub repository private
   * - ``--hub-strategy``
     - str
     - ``end``
     - When to upload to Hub: 'end' (after training) or 'checkpoint' (during training)

Configuration Files
===================

Kaira supports both YAML (Hydra format) and JSON configuration files for comprehensive parameter specification.

Hydra YAML Format (Recommended)
--------------------------------

.. code-block:: yaml

    # @package _global_

    defaults:
      - _self_

    # Model configuration
    model:
      _target_: kaira.models.DeepJSCCModel
      type: deepjscc
      input_dim: 512
      channel_uses: 64
      hidden_dim: 256

    # Training configuration
    training:
      output_dir: ./training_results
      num_train_epochs: 10
      per_device_train_batch_size: 32
      learning_rate: 1e-4
      snr_min: 0.0
      snr_max: 20.0
      channel_type: awgn
      do_eval: true

    # Hydra configuration
    hydra:
      run:
        dir: ${training.output_dir}/hydra_outputs/${now:%Y-%m-%d_%H-%M-%S}

JSON Format
-----------

.. code-block:: json

    {
      "model": {
        "type": "deepjscc",
        "input_dim": 512,
        "channel_uses": 64,
        "hidden_dim": 256
      },
      "training": {
        "output_dir": "./training_results",
        "num_train_epochs": 10,
        "per_device_train_batch_size": 32,
        "learning_rate": 1e-4,
        "snr_min": 0.0,
        "snr_max": 20.0,
        "channel_type": "awgn",
        "do_eval": true
      }
    }

Training Examples
=================

Deep Joint Source-Channel Coding
---------------------------------

.. code-block:: bash

    kaira-train --model deepjscc \\
      --output-dir ./deepjscc_results \\
      --epochs 15 \\
      --batch-size 64 \\
      --learning-rate 1e-4 \\
      --snr-min 0 \\
      --snr-max 20 \\
      --channel-uses 64 \\
      --do-eval \\
      --eval-steps 500

Channel Coding
--------------

.. code-block:: bash

    kaira-train --model channel_code \\
      --output-dir ./channel_code_results \\
      --epochs 20 \\
      --code-length 128 \\
      --info-length 64 \\
      --snr-min -5 \\
      --snr-max 15 \\
      --channel-type awgn

Configuration-Based Training
----------------------------

.. code-block:: bash

    kaira-train --model deepjscc --config-file ./configs/training_example.yaml

Training with Hub Upload
------------------------

.. code-block:: bash

    # Train and upload to Hugging Face Hub
    kaira-train --model deepjscc \\
      --output-dir ./deepjscc_results \\
      --epochs 15 \\
      --push-to-hub \\
      --hub-model-id username/deepjscc-model

    # Train and upload to private repository
    kaira-train --model deepjscc \\
      --output-dir ./deepjscc_results \\
      --epochs 20 \\
      --push-to-hub \\
      --hub-model-id username/private-deepjscc \\
      --hub-private

Checkpoint Resume
-----------------

.. code-block:: bash

    kaira-train --model deepjscc \\
      --resume-from-checkpoint ./deepjscc_results/checkpoint-2000 \\
      --output-dir ./deepjscc_results_continued

Model Integration
=================

Registering Custom Models
--------------------------

Models must be registered with the ModelRegistry to be accessible:

.. code-block:: python

    from kaira.models import ModelRegistry, BaseModel

    @ModelRegistry.register_model("my_custom_model")
    class MyCustomModel(BaseModel):
        def __init__(self, input_dim=256, **kwargs):
            super().__init__()
            self.input_dim = input_dim
            # Model implementation

Model Requirements
------------------

Training models should:

- Inherit from ``BaseModel``
- Handle data generation internally (for communication models)
- Support the standard training interface
- Implement proper forward/loss computation

Data Handling
=============

Communication models in Kaira typically generate synthetic data on-the-fly based on their configuration. The training script supports:

- **Synthetic Data**: Models generate data internally
- **External Datasets**: Optional dataset loading
- **Custom Data Paths**: Specify training/evaluation data

.. code-block:: bash

    # External dataset (if supported by model)
    kaira-train --model deepjscc \\
      --dataset custom_dataset \\
      --train-data-path ./data/train \\
      --eval-data-path ./data/eval

Monitoring and Logging
======================

Output Structure
----------------

.. code-block:: text

    training_results/
    ├── checkpoints/
    │   ├── checkpoint-1000/
    │   ├── checkpoint-2000/
    │   └── checkpoint-3000/
    ├── logs/
    │   └── training.log
    ├── config.json
    └── pytorch_model.bin

Integration with Monitoring Tools
---------------------------------

Configure monitoring in YAML:

.. code-block:: yaml

    training:
      logging_dir: ${training.output_dir}/logs
      report_to: ["wandb", "tensorboard"]
      run_name: my_experiment

Hugging Face Hub Integration
============================

Kaira supports uploading trained models to the Hugging Face Hub, making it easy to share and distribute your communication system models.

Features
--------

- **Automatic Upload**: Upload models to Hugging Face Hub after training
- **Flexible Strategies**: Upload at the end of training or during checkpointing
- **Private Repositories**: Support for private model repositories
- **Rich Model Cards**: Automatically generated model cards with training details
- **Authentication**: Multiple authentication methods (token, environment variable)

Hub Arguments
-------------

.. list-table::
   :header-rows: 1
   :widths: 20 10 15 55

   * - Argument
     - Type
     - Default
     - Description
   * - ``--push-to-hub``
     - flag
     - False
     - Enable Hub upload
   * - ``--hub-model-id``
     - str
     - \-
     - Model ID (username/model-name)
   * - ``--hub-token``
     - str
     - \-
     - Authentication token
   * - ``--hub-private``
     - flag
     - False
     - Make repository private
   * - ``--hub-strategy``
     - str
     - ``end``
     - Upload strategy: ``end`` or ``checkpoint``

Quick Start
-----------

Basic upload:

.. code-block:: bash

    kaira-train --model deepjscc --push-to-hub --hub-model-id username/my-model

Private repository:

.. code-block:: bash

    kaira-train --model deepjscc --push-to-hub \\
      --hub-model-id username/my-model --hub-private

With authentication token:

.. code-block:: bash

    kaira-train --model deepjscc --push-to-hub \\
      --hub-model-id username/my-model --hub-token your_token_here

Upload Strategies
-----------------

**End Strategy (default)**

Uploads the model only after training is completed:

.. code-block:: bash

    kaira-train --model deepjscc --push-to-hub \\
      --hub-model-id username/my-model --hub-strategy end

**Checkpoint Strategy**

Uploads the model during training at each checkpoint:

.. code-block:: bash

    kaira-train --model deepjscc --push-to-hub \\
      --hub-model-id username/my-model --hub-strategy checkpoint

Authentication
--------------

**Method 1: Environment Variable (Recommended)**

.. code-block:: bash

    export HF_TOKEN=your_huggingface_token
    kaira-train --model deepjscc --push-to-hub --hub-model-id username/my-model

**Method 2: Command Line Argument**

.. code-block:: bash

    kaira-train --model deepjscc --push-to-hub \\
      --hub-model-id username/my-model --hub-token your_token_here

**Method 3: Hugging Face CLI**

.. code-block:: bash

    huggingface-cli login
    kaira-train --model deepjscc --push-to-hub --hub-model-id username/my-model

Configuration File Integration
------------------------------

You can also specify Hub upload options in Hydra configuration files:

.. code-block:: yaml

    # training_config.yaml
    training:
      output_dir: "./results"
      num_train_epochs: 10
      push_to_hub: true
      hub_model_id: "username/my-model"
      hub_private: false
      hub_strategy: "end"

Then run:

.. code-block:: bash

    kaira-train --model deepjscc --config-file training_config.yaml

Generated Content
-----------------

For each uploaded model, the system automatically creates:

1. **pytorch_model.bin** - Model weights (state_dict)
2. **README.md** - Auto-generated model card with training details
3. **config.json** - Model configuration and metadata

Example model card content:

.. code-block:: markdown

    # my-model

    This model was trained using the Kaira framework for communication systems.

    ## Model Information

    - Framework: Kaira
    - Model Type: deepjscc
    - Training Configuration: ./results

    ## Usage

    ```python
    import torch
    from kaira.models import ModelRegistry

    # Load the model
    model_class = ModelRegistry.get_model_cls('deepjscc')
    model = model_class()

    # Load the trained weights
    state_dict = torch.load('pytorch_model.bin')
    model.load_state_dict(state_dict)
    ```

    ## Training Details

    - Epochs: 10.0
    - Batch Size: 32
    - Learning Rate: 0.0001
    - SNR Range: 0.0 to 20.0 dB

Hub Examples
------------

**Research Model Sharing**

.. code-block:: bash

    kaira-train \\
      --model channel_code \\
      --snr-min -5 \\
      --snr-max 25 \\
      --epochs 50 \\
      --push-to-hub \\
      --hub-model-id research-lab/channel-code-5g \\
      --verbose

**Private Development**

.. code-block:: bash

    kaira-train \\
      --model deepjscc \\
      --epochs 100 \\
      --batch-size 64 \\
      --push-to-hub \\
      --hub-model-id company/internal-deepjscc-v2 \\
      --hub-private

**Checkpoint Monitoring**

.. code-block:: bash

    kaira-train \\
      --model feedback_channel \\
      --epochs 200 \\
      --save-steps 1000 \\
      --push-to-hub \\
      --hub-model-id username/feedback-channel-experiment \\
      --hub-strategy checkpoint

Requirements
------------

The Hub upload functionality requires the ``huggingface_hub`` package:

.. code-block:: bash

    pip install huggingface_hub>=0.16.0

This dependency is automatically included in the updated ``requirements.txt``.

Hub Troubleshooting
-------------------

**"Hub model ID required"**
   Ensure you provide ``--hub-model-id`` when using ``--push-to-hub``

**"Authentication failed"**
   Check your token with ``huggingface-cli whoami`` and ensure token has write permissions

**"Repository not found"**
   The repository will be created automatically; check your username spelling

**"Network timeout"**
   Large models may take time to upload; check your internet connection

Use ``--verbose`` flag for detailed upload information:

.. code-block:: bash

    kaira-train --model deepjscc --push-to-hub --hub-model-id username/my-model --verbose

Advanced Features
=================

Mixed Precision Training
------------------------

.. code-block:: bash

    kaira-train --model deepjscc --fp16

Custom Device Selection
-----------------------

.. code-block:: bash

    # Force CPU
    kaira-train --model deepjscc --device cpu

    # Force CUDA
    kaira-train --model deepjscc --device cuda

Evaluation Strategies
---------------------

.. code-block:: bash

    # Evaluate every epoch
    kaira-train --model deepjscc --eval-strategy epoch

    # Disable evaluation
    kaira-train --model deepjscc --eval-strategy no

    # Custom evaluation frequency
    kaira-train --model deepjscc --eval-strategy steps --eval-steps 100

Troubleshooting
===============

Common Issues
-------------

**Model Not Found**

.. code-block:: bash

    Error: Unknown model 'model_name'

- Check available models: ``kaira-train --list-models``
- Ensure model is registered in ModelRegistry

**Configuration Errors**

.. code-block:: bash

    Error: OmegaConf is required for YAML configuration files

- Install OmegaConf: ``pip install omegaconf``

**Training Dataset Required**

.. code-block:: bash

    Error: Trainer: training requires a train_dataset

- Communication models should handle data generation internally
- Check model implementation for dataset requirements

**CUDA Out of Memory**

.. code-block:: bash

    RuntimeError: CUDA out of memory

- Reduce batch size: ``--batch-size 16``
- Use CPU: ``--device cpu``
- Enable mixed precision: ``--fp16``

Debugging
---------

Enable verbose output:

.. code-block:: bash

    kaira-train --model deepjscc --verbose

Check model parameters:

.. code-block:: bash

    kaira-train --list-models  # See available models

Validate configuration:

.. code-block:: bash

    python -c "
    from omegaconf import OmegaConf
    config = OmegaConf.load('configs/training_example.yaml')
    print(OmegaConf.to_yaml(config))
    "

API Reference
=============

For programmatic usage, see:

- :class:`kaira.training.TrainingArguments`: Training configuration
- :class:`kaira.training.Trainer`: Training implementation
- :class:`kaira.models.ModelRegistry`: Model management

See Also
========

- :doc:`api_reference`: API documentation
- :doc:`benchmarks`: Performance evaluation
- :doc:`best_practices`: Development best practices
