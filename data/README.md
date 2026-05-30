COMPAS data
===========

The first time you run main_compas.py the script will download the
ProPublica COMPAS CSV from

 https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv


and cache it here as `compas-scores-two-years.csv` (~5 MB). On subsequent runs it
loads from this folder.

If you are behind a firewall, download the file manually from the URL  above (or its mirror at https://github.com/propublica/compas-analysis 
- using the compas-scores.csv will not be allow you to duplicate the information from the article or in the paper. Use
- the compas-scores-two-years.csv because it also includes the follow up information on recidivism) and place it at:

    data/compas-scores-two-years.csv
