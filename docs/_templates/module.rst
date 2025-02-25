{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}
   :members:
   :show-inheritance:
   
   {% if classes %}
   .. rubric:: Classes

   .. autosummary::
      :nosignatures:
      :toctree:
   {% for class in classes %}
      {{ class }}
   {%- endfor %}
   {% endif %}
   
   {% if functions %}
   .. rubric:: Functions

   .. autosummary::
      :nosignatures:
      :toctree:
   {% for function in functions %}
      {{ function }}
   {%- endfor %}
   {% endif %}
   
   {% if attributes %}
   .. rubric:: Attributes

   .. autosummary::
      :nosignatures:
      :toctree:
   {% for attribute in attributes %}
      {{ attribute }}
   {%- endfor %}
   {% endif %}
   
   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
      :nosignatures:
      :toctree:
   {% for method in methods %}
      {{ method }}
   {%- endfor %}
   {% endif %}
