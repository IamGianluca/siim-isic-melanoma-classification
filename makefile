format:
	isort -rc . && \
	black -l 79 .

submit:
	kaggle competitions submit -c siim-isic-melanoma-classification -f subs/submission.csv -m "$(COMMENT)"

pep8_checks:
	flake8 siim-isic-melanoma-classification/ test/ pipe/

type_checks:
	mypy siim-isic-melanoma-classification pipe tests --ignore-missing-imports

unittest:
	make type_checks && \
	pytest -xsvv --lf -p no:warnings --cov siim-isic-melanoma-classification --cov-report term-missing --ignore tests/integration

test:
	make type_checks && \
	pytest -xsvv --lf -p no:warnings --cov siim-isic-melanoma-classification --cov-report term-missing
