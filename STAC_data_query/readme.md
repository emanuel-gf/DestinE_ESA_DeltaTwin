## How to use

the argument should be passed as 

### Test STAC QUERY locally. 

To test it locally, please run the following command:

Go to the folder holding main.py
```bash
cd STAC_data_query/models
```
Run 
```bash
python main.py 3.2833 45.3833 11.2 50.1833 20
```

To ask for help: 

```bash
cd STAC_data_query/models

python main.py --help

```
### Test DeltaTwin locally

To test it locally, run:

Go to the main STAC_data_query folder, containing: manifest.json, workflow.yml, models/, inputs_file.json

```bash
cd STAC_data_query

deltatwin run start_local -i inputs_file.json
```

### Publish Delta Twin

1. Make Sure you have login beforehand:
```bash
deltatwin login -a https://api.deltatwin.destine.eu/ username password 
```