
clean:
	rm -rf openscope_predictive_coding/data/templates

templates:
	python openscope_predictive_coding/scripts/stimulus/generate_stimulus_templates.py
	python openscope_predictive_coding/scripts/stimulus/generate_occlusion_template.py