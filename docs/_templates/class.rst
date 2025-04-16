{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. inheritance-diagram:: {{ objname }}
   :parts: 1
   :private-bases:
   :caption: Inheritance diagram for {{ objname }}
   :top-classes: abc.ABC torch.nn.Module

.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:
   :inherited-members:
   :special-members: __call__, __add__, __mul__
   :exclude-members: forward

   {% block methods %}
   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :nosignatures:
   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% if examples %}
   .. rubric:: {{ _('Examples') }}

   {{ examples }}
   {% endif %}

   {% block backreferences %}
   .. include:: ../gen_modules/backreferences/{{module}}.{{objname}}.examples
   {% endblock %}
