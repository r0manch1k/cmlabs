PACKAGEDIR = cmlabs

format-docs:
	docformatter -i -r --black $(PACKAGEDIR)

.PHONY: format-docs

tests:
	pytest -s

.PHONY: tests

autodoc:
	sphinx-apidoc -f -o docs/source cmlabs

.PHONY: autodoc