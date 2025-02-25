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
