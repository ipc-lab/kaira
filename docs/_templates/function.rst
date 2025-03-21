{{ fullname | escape | underline}}
.. currentmodule:: {{ module }}
.. autofunction:: {{ objname }}

{% block backreferences %}
.. include:: ../gen_modules/backreferences/{{module}}.{{objname}}.examples
{% endblock %}
