# Spend Report AI API
- It is an API to process the extract info from invoices using AI.

## To setup this
- Clone the repository
```bash
git clone https://ArchKey@dev.azure.com/ArchKey/RPA%20AI/_git/Spend%20Report%20AI%20API
```
- In a terminal, navigate to `Spend Report AI API` folder
- Activate your desired virtual environment
- In the terminal, type `pip install -r requirements.txt`
- Run this command `pre-commit install`
- Create a `env/temp.env.yaml` file with `sys_name: '<your_name>'` which is added in .gitignore.
- You need access to key vault to run the app.
- Run your app with `python app.py`


## Testing the Simple API
- Open the Postman app or site
- Enter URL `http://127.0.0.1:8000/v1/process_invoice_details` and post the request with necessary payload

## Manual execution of pre-commit hooks
- `pre-commit run --all-files`

## Databases and APIs
- It uses the Azure Cosmos DB for storing intermediate data
- It connects with SDP for fetching data required and writes response back
- It uses Azure Open AI for LLM services

## Configurations
- config.yaml is used for user config changes.
- env/X.env.yaml is env specific config file for DB and API connections where X is [local, dev].
- App settings in environment variables are added for this web app to work in Azure
    - LOG_LEVEL, WEB_APP_ENV

## Points to Remember
- DO NOT commit code to Git with env=local in config.yaml
- DO NOT deploy code to Azure apps with env=local in config.yaml
