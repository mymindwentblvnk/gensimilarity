env:
	virtualenv -p python3 venv
	. venv/bin/activate && pip install -r requirements.txt && python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')" && deactivate

jupyter:
	. venv/bin/activate && jupyter notebook && deactivate
