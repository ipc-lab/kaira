Modulation
==========

Digital modulation schemes and constellation analysis. These examples demonstrate
the implementation and analysis of various digital modulation techniques using
the Kaira library.

.. toctree::
   :maxdepth: 1
   :caption: Examples:

   plot_psk_modulation
   plot_qam_modulation
   plot_pam_modulation
   plot_differential_modulation
   plot_constellation_comparison

Example Descriptions:
-----------------------------------

- **plot_psk_modulation.py**: Demonstrates Phase-Shift Keying (PSK) modulation,
  focusing on BPSK and QPSK implementations. Shows constellation diagrams and
  BER analysis under AWGN conditions.

- **plot_qam_modulation.py**: Explores Quadrature Amplitude Modulation (QAM) with
  different orders (4-QAM, 16-QAM, 64-QAM). Analyzes constellation patterns and
  performance characteristics.

- **plot_pam_modulation.py**: Illustrates Pulse Amplitude Modulation (PAM) with
  various orders, showing amplitude levels and their impact on performance.

- **plot_differential_modulation.py**: Demonstrates differential modulation schemes
  (DBPSK and DQPSK), highlighting their robustness to phase ambiguity and
  performance in fading channels.

- **plot_constellation_comparison.py**: Provides a comprehensive comparison of
  different modulation schemes, analyzing their spectral efficiency, power
  requirements, and performance trade-offs.
