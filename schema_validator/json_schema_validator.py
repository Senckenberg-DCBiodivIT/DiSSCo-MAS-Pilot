import json
import jsonschema
from jsonschema  import validate
from jsonschema.exceptions import ValidationError

with open ('response.json') as f:
    document = json.load(f)

with open('schema.json') as f:
    schema = json.load(f)

try:
    validate(instance=document, schema=schema)
except ValidationError as e:
    print(f"JSON validation error: {e.message}")
    print(f"Error path: {e.path}")
    print(f"Schema path: {e.schema_path}")
    print(f"Validator: {e.validator}")
    print(f"Validator value: {e.validator_value}")


