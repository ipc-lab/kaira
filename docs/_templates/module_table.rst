{{ name | escape | underline}}

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Component
     - Description
   {% for item in members %}
   {% if item.__doc__ %}
   * - :{{ item.__module__ }}.{{ item.__name__ }}:`~{{ item.__module__ }}.{{ item.__name__ }}`
     - {{ item.__doc__.split('\n')[0] | escape }}
   {% else %}
   * - :{{ item.__module__ }}.{{ item.__name__ }}:`~{{ item.__module__ }}.{{ item.__name__ }}`
     - *No description available*
   {% endif %}
   {% endfor %}

   {% if attributes %}
   .. rubric:: Attributes

   .. list-table::
      :widths: 30 70
      :header-rows: 1

      * - Attribute
        - Description
      {% for attribute in attributes %}
      {% if attribute.__doc__ %}
      * - :{{ attribute.__module__ }}.{{ attribute.__name__ }}:`~{{ attribute.__module__ }}.{{ attribute.__name__ }}`
        - {{ attribute.__doc__.split('\n')[0] | escape }}
      {% else %}
      * - :{{ attribute.__module__ }}.{{ attribute.__name__ }}:`~{{ attribute.__module__ }}.{{ attribute.__name__ }}`
        - *No description available*
      {% endif %}
      {% endfor %}
   {% endif %}

   {% if methods %}
   .. rubric:: Methods

   .. list-table::
      :widths: 30 70
      :header-rows: 1

      * - Method
        - Description
      {% for method in methods %}
      {% if method.__doc__ %}
      * - :{{ method.__module__ }}.{{ method.__name__ }}:`~{{ method.__module__ }}.{{ method.__name__ }}`
        - {{ method.__doc__.split('\n')[0] | escape }}
      {% else %}
      * - :{{ method.__module__ }}.{{ method.__name__ }}:`~{{ method.__module__ }}.{{ method.__name__ }}`
        - *No description available*
      {% endif %}
      {% endfor %}
   {% endif %}
