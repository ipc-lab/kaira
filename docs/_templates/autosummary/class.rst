{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:

   {% block methods %}
   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :toctree:
   {% for item in methods %}
      ~{{ fullname }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
      :toctree:
   {% for item in attributes %}
      ~{{ fullname }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

{% if objname in ["AWGNChannel", "BaseChannel", "BinaryErasureChannel", "BinarySymmetricChannel", "BinaryZChannel",
                 "ChannelRegistry", "FlatFadingChannel", "GaussianChannel", "IdealChannel", "IdentityChannel",
                 "LambdaChannel", "LaplacianChannel", "NonlinearChannel", "PerfectChannel", "PhaseNoiseChannel", "PoissonChannel"] %}
.. rubric:: {{ _('Examples using') }} ``{{ objname }}``

.. include:: ../gen_modules/backreferences/{{module}}.{{objname}}.examples
   :optional:
{% endif %}
