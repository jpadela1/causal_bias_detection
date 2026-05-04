COMPAS data
===========

The first time you run main_compas.py the script will download the
ProPublica COMPAS CSV from

    https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores.csv

and cache it here as `compas-scores.csv` (~5 MB). On subsequent runs it
loads from this folder.

If you are behind a firewall, download the file manually from the URL
above (or its mirror at https://github.com/propublica/compas-analysis)
and place it at:

    data/compas-scores.csv
