PACKAGEDIR = cmlabs

.PHONY: format-docs
format-docs:
	docformatter -i -r --black $(PACKAGEDIR)


.PHONY: tests
tests:
	pytest -s

.PHONY: tests-interpolate
tests-interpolate:
	pytest -s cmlabs/interpolate

.PHONY: tests-differentiate
tests-differentiate:
	pytest -s cmlabs/differentiate

.PHONY: tests-integrate
tests-integrate:
	pytest -s cmlabs/integrate

.PHONY: tests-linalg
tests-linalg:
	pytest -s cmlabs/linalg

.PHONY: tests-optimize
tests-optimize:
	pytest -s cmlabs/optimize


.PHONY: autodoc
autodoc:
	sphinx-apidoc -f -o docs/source cmlabs
