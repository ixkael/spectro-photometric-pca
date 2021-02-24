
# Spectro-photometric-pca


## Set-up and testing

After downloading and unzipping, you could use pip and run the tests:
```
$ pip install -r requirements.txt
$ pip install -e .
$ pytest .
```

However, it is much better to deploy Poetry,
```
poetry install
poetry run pytest --cov
```
and then use its virtual environment to run a notebook:
```
poetry shell
jupyter notebook
```

## Notebooks and scripts


## People
* [Boris Leistedt](https://github.com/ixkael) (Imperial College London)

## License, etc.
