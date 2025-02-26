Modulation Schemes and Visualization
====================================

This tutorial introduces Kaira's modulation package, focusing on constellation visualization
tools for analyzing digital modulation schemes.

.. contents:: Table of Contents
   :depth: 2
   :local:

Introduction to Digital Modulation
----------------------------------

Digital modulation is the process of encoding information onto carrier signals for
transmission through physical channels. Kaira provides implementations of common
modulation schemes and tools to analyze and visualize their properties.

Modulation Architecture
-----------------------

Kaira's modulation system is built on two core base classes:

- :class:`~kaira.core.BaseModulator`: Abstract base class for all modulators
- :class:`~kaira.core.BaseDemodulator`: Abstract base class for all demodulators

These core classes define the interfaces that all modulation implementations must follow,
ensuring consistent behavior across different schemes.

Basic Usage
-----------

Here's a simple example of using a QAM modulator:

.. code-block:: python

    import torch
    from kaira.modulations import QAMModulator, QAMDemodulator
    
    # Create a 16-QAM modulator
    mod = QAMModulator(bits_per_symbol=4)  # 16-QAM
    
    # Generate random bits
    bits = torch.randint(0, 2, (1000,), dtype=torch.float32)
    
    # Modulate bits into symbols
    symbols = mod.modulate(bits)
    
    # Create demodulator and recover bits
    demod = QAMDemodulator(bits_per_symbol=4)
    recovered_bits = demod.demodulate(symbols)
    
    # Check bit error rate
    ber = torch.mean((bits != recovered_bits).float()).item()
    print(f"BER: {ber}")

Using ConstellationVisualizer
-----------------------------

The :class:`~kaira.modulations.ConstellationVisualizer` class provides
advanced tools for visualizing and analyzing modulation constellations.

Basic Constellation Plot
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from kaira.modulations import QAMModulator, ConstellationVisualizer
    
    # Create a 16-QAM modulator
    mod = QAMModulator(bits_per_symbol=4)  # 16-QAM
    
    # Create visualizer and generate basic plot
    viz = ConstellationVisualizer(modulator=mod)
    fig = viz.plot_basic(show_labels=True)
    fig.savefig('16qam_constellation.png')

Decision Regions
^^^^^^^^^^^^^^^^

Visualize decision boundaries between constellation points:

.. code-block:: python

    # Plot decision regions for the constellation
    fig = viz.plot_decision_regions(resolution=200)

Noise Effects
^^^^^^^^^^^^^

Analyze the effects of AWGN on the constellation:

.. code-block:: python

    # Visualize constellation with noise at 15 dB SNR
    fig = viz.plot_with_noise(snr_db=15.0, n_points=1000)

Bit Error Rate Analysis
^^^^^^^^^^^^^^^^^^^^^^^

Estimate BER performance across different SNR levels:

.. code-block:: python

    # Generate BER curve
    fig = viz.plot_ber_estimation(snr_db_range=[0, 5, 10, 15, 20])

Phase Rotation Animation
^^^^^^^^^^^^^^^^^^^^^^^^

Create animations to visualize phase rotation effects:

.. code-block:: python

    # Generate phase rotation animation
    anim = viz.animate_phase_rotation(n_frames=100, rotation_cycles=1.0)
    
    # Display in Jupyter notebook
    from IPython.display import HTML
    HTML(anim.to_jshtml())
    
    # Or save as video
    # anim.save('phase_rotation.mp4', writer='ffmpeg')

Eye Diagrams
^^^^^^^^^^^^

Generate eye diagrams to analyze intersymbol interference:

.. code-block:: python

    # Create eye diagrams with Root Raised Cosine pulse shaping
    fig = viz.plot_eye_diagram(
        snr_db=20.0,
        pulse_type='rrc',
        beta=0.35
    )

Comparing Modulation Schemes
----------------------------

Kaira provides tools to compare different modulation schemes:

.. code-block:: python

    from kaira.modulations import benchmark_modulation_schemes
    from kaira.modulations import BPSKModulator, QPSKModulator, QAMModulator
    
    # Create modulators to compare
    modulators = [
        BPSKModulator(),
        QPSKModulator(),
        QAMModulator(bits_per_symbol=4)  # 16-QAM
    ]
    
    # Compare BER performance
    snr_range = range(0, 21, 2)  # 0 to 20 dB
    fig = benchmark_modulation_schemes(
        modulators, 
        snr_db_range=snr_range,
        labels=['BPSK', 'QPSK', '16-QAM']
    )

Advanced Visualization and Analysis
-----------------------------------

For more advanced use cases, the ConstellationVisualizer provides additional
methods:

- :meth:`~kaira.modulations.ConstellationVisualizer.plot_bit_reliability`: Analyze bit reliability with Log-Likelihood Ratio (LLR) heatmaps
- :meth:`~kaira.modulations.ConstellationVisualizer.plot_trajectory`: Visualize trajectories between consecutive symbols