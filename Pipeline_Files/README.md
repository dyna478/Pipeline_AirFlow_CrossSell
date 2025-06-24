# Cross-Selling Automobile Insurance

This project contains scripts for cross-selling automobile insurance to existing health insurance clients.

## Overview

The main workflow consists of:

1. Processing daily CSV files with health insurance client data
2. Identifying clients suitable for automobile insurance cross-selling
3. Generating predictions using a machine learning model
4. Sending qualified leads to the TelusHolding API
5. Generating daily reports

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Testing without Airflow

You can test the entire workflow without installing Airflow using the provided test scripts:

### Basic testing without real API

```bash
python run_test_without_airflow.py
```

### Testing with real API connection

The system connects to TelusHolding API using POST requests and API key authentication.
Set the TELUS_API_KEY environment variable before running the API tests:

```bash
# Windows
set TELUS_API_KEY=your_api_key_here
set TELUS_USE_TEST=true

# Linux/Mac
export TELUS_API_KEY=your_api_key_here
export TELUS_USE_TEST=true
```

Then run the API test:

```bash
python test_api_connection.py
```

For full workflow testing with real API:

```bash
python real_api_test.py
```

## API Configuration

The system supports both test and production environments:

- **Test API**: https://telusholding.cloud/LEADSMANAGER/API/lead_insertV2_SAND.php
- **Production API**: http://192.168.144.10/API/lead_insertV2.php

Toggle between them using the `TELUS_USE_TEST` environment variable.

## Field Mappings

The system uses the following field mappings when sending data to the API:

- `firstname` → Lead's first name
- `lastname` → Lead's last name
- `emailaddress` → Email address
- `phonenumber` → Phone number
- `city` → City
- `gender` → Gender (M/F)
- `agent` → Agent code
- `promo_code` → Promotion code
- `category` → Category (AUTO)
- `power` → Engine power
- `energy` → Fuel type

## Generating a Dummy Model

If you don't have a model file, you can generate a dummy one:

```bash
python create_dummy_model.py
```

## Logs and Reports

- Reports are saved to `./reports/`
- Output data is saved to `./data/output/` 