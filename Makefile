PACKAGEDIR = cmlabs

format-docs:
	docformatter -i -r --black $(PACKAGEDIR)

.PHONY: format-docs